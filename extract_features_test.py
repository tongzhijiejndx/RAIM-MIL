import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import re

from utils import cv_imread_safe, get_ultrasound_fan_mask


CONFIG = {
    'DATA_ROOT': 'data/test',
    'SAVE_DIR': 'data/features_test_noroi',
    'WEIGHT_PATH': 'model/RadImageNet_ResNet50.pth',
    'EXCEL_PATH': 'all_with_scores.xlsx',

    'PHYSICAL_PATCH_SIZE': 96,
    'MODEL_INPUT_SIZE': 224,
    'STRIDE': 24,

    'FAN_THRESHOLD': 15,
    'MASK_RATIO_LIMIT': 0.85,
    'BG_MEAN_LIMIT': 10,
    'TOP_CUT_RATIO': 0.15,
    'BOTTOM_CUT_RATIO': 0.20,

    'STRICT_REQUIRE_ROI': False,

    'MAX_PATCHES_PER_IMAGE': None,

    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'BATCH_SIZE_EXTRACT': 64,
}

transform = transforms.Compose([
    transforms.Resize((CONFIG['MODEL_INPUT_SIZE'], CONFIG['MODEL_INPUT_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
CLASS_MAP = {'normal': 0, 'early': 1, 'middle': 2, 'late': 3}


def get_patient_key(folder_name):
    match = re.match(r"(\d+)", folder_name)
    if match:
        return int(match.group(1))
    return folder_name


def get_radimagenet_model(weight_path, device):
    print(f"loading RadImageNet: {weight_path}")
    model = models.resnet50(weights=None)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"files not found: {weight_path}")

    checkpoint = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model


def append_csv_row(csv_path, row_dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_row = pd.DataFrame([row_dict])
    file_exists = os.path.exists(csv_path)
    df_row.to_csv(csv_path, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')


def safe_read_image(img_path, image_error_log_path=None):
    try:
        img = cv_imread_safe(img_path)
    except Exception as e:
        if image_error_log_path is not None:
            append_csv_row(image_error_log_path, {
                'img_path': img_path,
                'reason': 'read_exception',
                'detail': str(e),
            })
        return None

    if img is None:
        if image_error_log_path is not None:
            append_csv_row(image_error_log_path, {
                'img_path': img_path,
                'reason': 'read_none',
                'detail': 'cv_imread_safe returned None',
            })
        return None
    return img


def build_fan_mask(img):
    mask, _ = get_ultrasound_fan_mask(img, fan_threshold=CONFIG['FAN_THRESHOLD'])
    if mask is None:
        return None

    y_indices, _ = np.where(mask > 0)
    if len(y_indices) == 0:
        return None

    y_min, y_max = np.min(y_indices), np.max(y_indices)
    fan_height = y_max - y_min
    if fan_height <= 0:
        return None

    top_cutoff_y = int(y_min + fan_height * CONFIG['TOP_CUT_RATIO'])
    mask[:top_cutoff_y, :] = 0

    bottom_cutoff_y = int(y_max - fan_height * CONFIG.get('BOTTOM_CUT_RATIO', 0.0))
    mask[bottom_cutoff_y:, :] = 0

    if cv2.countNonZero(mask) == 0:
        return None

    return mask


def build_candidate_patch_records(img, fan_mask):
    H, W = img.shape[:2]
    P = CONFIG['PHYSICAL_PATCH_SIZE']
    S = CONFIG['STRIDE']

    records = []
    for y in range(0, H - P + 1, S):
        for x in range(0, W - P + 1, S):
            patch_mask = fan_mask[y:y + P, x:x + P]
            fan_ratio = cv2.countNonZero(patch_mask) / float(P * P)
            if fan_ratio <= CONFIG['MASK_RATIO_LIMIT']:
                continue

            patch_img = img[y:y + P, x:x + P]
            if patch_img.mean() <= CONFIG['BG_MEAN_LIMIT']:
                continue

            patch_pil = Image.fromarray(cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB))
            records.append({
                'tensor': transform(patch_pil),
                'patch_box': [x, y, x + P, y + P],
            })

    return records


def finalize_records(records, mode_tag):
    if len(records) == 0:
        return [], [], [], mode_tag

    if CONFIG['MAX_PATCHES_PER_IMAGE'] is not None and len(records) > CONFIG['MAX_PATCHES_PER_IMAGE']:
        records = records[:CONFIG['MAX_PATCHES_PER_IMAGE']]

    patches = [r['tensor'] for r in records]
    patch_boxes = [r['patch_box'] for r in records]

    patch_labels = [0.0 for _ in records]

    return patches, patch_labels, patch_boxes, mode_tag


@torch.no_grad()
def extract_features_from_patches(patches, model):
    if len(patches) == 0:
        return None

    input_tensor = torch.stack(patches).to(CONFIG['DEVICE'])
    features_list = []
    for i in range(0, len(input_tensor), CONFIG['BATCH_SIZE_EXTRACT']):
        batch = input_tensor[i:i + CONFIG['BATCH_SIZE_EXTRACT']]
        features_list.append(model(batch).cpu())
    return torch.cat(features_list, dim=0)


def process_one_image(img_path, model, image_error_log_path=None, image_skip_log_path=None):
    img = safe_read_image(img_path, image_error_log_path=image_error_log_path)
    if img is None:
        return None

    fan_mask = build_fan_mask(img)
    if fan_mask is None:
        if image_skip_log_path is not None:
            append_csv_row(image_skip_log_path, {
                'img_path': img_path,
                'reason': 'fan_mask_invalid_after_top_bottom_cut',
                'detail': '',
            })
        return None

    candidate_records = build_candidate_patch_records(
        img=img,
        fan_mask=fan_mask,
    )

    patches, patch_labels, patch_boxes, fallback_mode = finalize_records(
        candidate_records,
        mode_tag='fan_fullbag_test_noroi',
    )

    if len(patches) == 0:
        if image_skip_log_path is not None:
            append_csv_row(image_skip_log_path, {
                'img_path': img_path,
                'reason': 'no_valid_patches_after_fullfan_filter',
                'detail': '',
            })
        return None

    feats = extract_features_from_patches(patches, model)
    if feats is None or feats.size(0) == 0:
        if image_skip_log_path is not None:
            append_csv_row(image_skip_log_path, {
                'img_path': img_path,
                'reason': 'feature_extraction_failed',
                'detail': '',
            })
        return None

    out = {
        'img_feats': feats,
        'patch_labels': torch.tensor(patch_labels, dtype=torch.float32),
        'patch_boxes': torch.tensor(patch_boxes, dtype=torch.int32),
        'num_patches': int(len(patches)),
        'img_path': img_path,
        'fallback_mode': fallback_mode,
        'num_candidates_after_basic_filter': int(len(candidate_records)),
    }
    return out


def load_clinical_dict(excel_path):
    clin_dict = {}
    num_clin_feats = 8

    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        num_clin_feats = len(df.columns) - 1
        print(f"detected features num: {num_clin_feats}")
        for _, row in df.iterrows():
            p_id = row.iloc[0]
            vals = pd.to_numeric(row.iloc[1:], errors='coerce').fillna(0).values
            clin_dict[p_id] = vals
    else:
        print(f"no file detected: {excel_path}，all 0。")

    return clin_dict, num_clin_feats


def export_all_patients():
    os.makedirs(CONFIG['SAVE_DIR'], exist_ok=True)

    image_error_log_path = os.path.join(CONFIG['SAVE_DIR'], 'image_read_errors.csv')
    image_skip_log_path = os.path.join(CONFIG['SAVE_DIR'], 'image_skip_logs.csv')
    patient_skip_log_path = os.path.join(CONFIG['SAVE_DIR'], 'patient_skip_logs.csv')
    summary_csv_path = os.path.join(CONFIG['SAVE_DIR'], 'export_summary.csv')

    clin_dict, num_clin_feats = load_clinical_dict(CONFIG['EXCEL_PATH'])

    resnet = get_radimagenet_model(CONFIG['WEIGHT_PATH'], CONFIG['DEVICE'])

    if not os.path.isdir(CONFIG['DATA_ROOT']):
        raise FileNotFoundError(f"DATA_ROOT not found: {CONFIG['DATA_ROOT']}")

    classes = [
        d for d in os.listdir(CONFIG['DATA_ROOT'])
        if os.path.isdir(os.path.join(CONFIG['DATA_ROOT'], d))
    ]

    summary_rows = []

    for cls_name in classes:
        label_int = CLASS_MAP.get(cls_name.lower())
        if label_int is None:
            continue

        print(f"\nclass: {cls_name} --> Label: {label_int}")
        cls_dir = os.path.join(CONFIG['DATA_ROOT'], cls_name)
        patient_dirs = [
            d for d in os.listdir(cls_dir)
            if os.path.isdir(os.path.join(cls_dir, d))
        ]

        for p_name in tqdm(patient_dirs, desc=f"Export {cls_name}"):
            p_path = os.path.join(cls_dir, p_name)
            p_key = get_patient_key(p_name)
            p_clin = clin_dict.get(p_key, np.zeros(num_clin_feats, dtype=np.float32))
            p_clin = torch.tensor(p_clin, dtype=torch.float32)

            img_files = [
                os.path.join(p_path, f)
                for f in os.listdir(p_path)
                if f.lower().endswith(VALID_EXTENSIONS)
            ]
            img_files = sorted(img_files)

            if len(img_files) == 0:
                append_csv_row(patient_skip_log_path, {
                    'patient_id': p_name,
                    'class_name': cls_name,
                    'reason': 'no_images',
                    'detail': p_path,
                })
                continue

            patient_feats = []
            patient_patch_labels = []
            patient_patch_boxes = []
            patient_patch_img_paths = []
            patient_fallback_modes = []
            valid_image_count = 0
            total_patches = 0

            for img_f in img_files:
                res = process_one_image(
                    img_path=img_f,
                    model=resnet,
                    image_error_log_path=image_error_log_path,
                    image_skip_log_path=image_skip_log_path,
                )
                if res is None:
                    continue

                valid_image_count += 1
                patient_feats.append(res['img_feats'])
                patient_patch_labels.append(res['patch_labels'])
                patient_patch_boxes.append(res['patch_boxes'])
                patient_patch_img_paths.extend([res['img_path']] * res['img_feats'].size(0))
                patient_fallback_modes.extend([res['fallback_mode']] * res['img_feats'].size(0))
                total_patches += int(res['num_patches'])

            if len(patient_feats) == 0:
                append_csv_row(patient_skip_log_path, {
                    'patient_id': p_name,
                    'class_name': cls_name,
                    'reason': 'no_valid_images_after_fullfan_export',
                    'detail': p_path,
                })
                continue

            final_feats = torch.cat(patient_feats, dim=0)
            final_patch_labels = torch.cat(patient_patch_labels, dim=0)
            final_patch_boxes = torch.cat(patient_patch_boxes, dim=0)

            save_name = f"{cls_name}_{p_name}.pt"
            save_path = os.path.join(CONFIG['SAVE_DIR'], save_name)

            torch.save({
                'img_feats': final_feats,
                'clin_feats': p_clin,
                'label': label_int,
                'patch_labels': final_patch_labels,
                'patient_id': p_name,
                'patch_boxes': final_patch_boxes,
                'patch_image_paths': patient_patch_img_paths,
                'patch_fallback_modes': patient_fallback_modes,
                'meta': {
                    'class_name': cls_name,
                    'num_images_total': len(img_files),
                    'num_images_valid': valid_image_count,
                    'num_patches_total': int(total_patches),
                    'use_roi_to_filter_bag': False,
                    'strict_require_roi': bool(CONFIG['STRICT_REQUIRE_ROI']),
                    'top_cut_ratio': float(CONFIG['TOP_CUT_RATIO']),
                    'bottom_cut_ratio': float(CONFIG['BOTTOM_CUT_RATIO']),
                    'patch_size': int(CONFIG['PHYSICAL_PATCH_SIZE']),
                    'stride': int(CONFIG['STRIDE']),
                }
            }, save_path)

            summary_rows.append({
                'file_name': save_name,
                'patient_id': p_name,
                'class_name': cls_name,
                'label': label_int,
                'num_images_total': len(img_files),
                'num_images_valid': valid_image_count,
                'num_patches_total': int(total_patches),
                'use_roi_to_filter_bag': False,
                'strict_require_roi': bool(CONFIG['STRICT_REQUIRE_ROI']),
                'top_cut_ratio': float(CONFIG['TOP_CUT_RATIO']),
                'bottom_cut_ratio': float(CONFIG['BOTTOM_CUT_RATIO']),
                'save_path': save_path,
            })

    if len(summary_rows) > 0:
        pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        print(f"summary has been saved: {summary_csv_path}")
    else:
        print("no one has been extracted")


if __name__ == '__main__':
    export_all_patients()