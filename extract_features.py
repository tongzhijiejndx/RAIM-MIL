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

from utils import cv_imread_safe, get_ultrasound_fan_mask, load_json_boxes

CONFIG = {
    'DATA_ROOT': 'data/train',
    'SAVE_DIR': 'data/features_train',
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

    'USE_ROI_TO_FILTER_BAG': False,

    'ROI_KEEP_OVERLAP_RATIO': 0.30,
    'ROI_KEEP_OVERLAP_RATIO_FALLBACKS': [0.20, 0.10, 0.05],

    # ===== ROI prior label 规则 =====
    'PATCH_POS_OVERLAP_RATIO': 0.55,
    'PATCH_POS_ROI_COVERAGE_RATIO': 0.55,

    'ROI_DILATE_PIXELS': 12,

    'PATCH_POS_MIN_INTER_AREA': 1.0,

    'PATCH_POS_USE_CENTER_RULE': True,

    'FORCE_TOPK_IF_EMPTY': 2,

    'STRICT_REQUIRE_ROI': True,

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
        raise FileNotFoundError(f"Can't find file: {weight_path}")

    checkpoint = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model


def compute_intersection_metrics(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    patch_overlap = float(inter_area / area_a) if area_a > 0 else 0.0
    roi_coverage = float(inter_area / area_b) if area_b > 0 else 0.0
    union = area_a + area_b - inter_area
    iou = float(inter_area / union) if union > 0 else 0.0

    return float(inter_area), patch_overlap, roi_coverage, iou


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


def load_roi_boxes_for_image(json_path, h, w):
    if not os.path.exists(json_path):
        return []

    raw_boxes = load_json_boxes(json_path)
    cleaned = []
    for b in raw_boxes:
        x1, y1, x2, y2 = b
        x1 = max(0, min(int(round(x1)), w - 1))
        y1 = max(0, min(int(round(y1)), h - 1))
        x2 = max(0, min(int(round(x2)), w))
        y2 = max(0, min(int(round(y2)), h))
        if x2 > x1 and y2 > y1:
            cleaned.append([x1, y1, x2, y2])
    return cleaned


def dilate_box(box, max_w, max_h, pad):
    x1, y1, x2, y2 = box
    x1 = max(0, int(round(x1 - pad)))
    y1 = max(0, int(round(y1 - pad)))
    x2 = min(max_w, int(round(x2 + pad)))
    y2 = min(max_h, int(round(y2 + pad)))
    return [x1, y1, x2, y2]


def point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)


def build_candidate_patch_records(img, fan_mask, roi_boxes):
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

            patch_box = (x, y, x + P, y + P)

            best_inter_area = 0.0
            best_patch_overlap = 0.0
            best_roi_coverage = 0.0
            best_iou = 0.0
            best_keep_score = 0.0

            for roi in roi_boxes:
                inter_area, patch_overlap, roi_coverage, iou = compute_intersection_metrics(patch_box, roi)
                keep_score = max(patch_overlap, roi_coverage)
                if keep_score > best_keep_score:
                    best_inter_area = inter_area
                    best_patch_overlap = patch_overlap
                    best_roi_coverage = roi_coverage
                    best_iou = iou
                    best_keep_score = keep_score

            patch_pil = Image.fromarray(cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB))
            records.append({
                'tensor': transform(patch_pil),
                'patch_box': [x, y, x + P, y + P],
                'keep_score': float(best_keep_score),
                'patch_overlap': float(best_patch_overlap),
                'roi_coverage': float(best_roi_coverage),
                'iou': float(best_iou),
                'inter_area': float(best_inter_area),
            })

    return records


def assign_patch_label(record, roi_boxes, img_w, img_h):
    patch_box = record['patch_box']
    x1, y1, x2, y2 = patch_box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    for roi in roi_boxes:
        roi_d = dilate_box(
            roi,
            max_w=img_w,
            max_h=img_h,
            pad=CONFIG['ROI_DILATE_PIXELS']
        )

        inter_area, patch_overlap, roi_coverage, iou = compute_intersection_metrics(
            patch_box, roi_d
        )

        if CONFIG.get('PATCH_POS_USE_CENTER_RULE', True):
            if point_in_box(cx, cy, roi_d):
                return 1.0

        if inter_area >= CONFIG.get('PATCH_POS_MIN_INTER_AREA', 1.0):
            return 1.0

        if patch_overlap >= CONFIG['PATCH_POS_OVERLAP_RATIO']:
            return 1.0
        if roi_coverage >= CONFIG['PATCH_POS_ROI_COVERAGE_RATIO']:
            return 1.0

    return 0.0


def finalize_records(records, mode_tag, roi_boxes, img_w, img_h):
    if len(records) == 0:
        return [], [], [], [], [], [], mode_tag

    if CONFIG['MAX_PATCHES_PER_IMAGE'] is not None and len(records) > CONFIG['MAX_PATCHES_PER_IMAGE']:
        order = np.argsort(np.asarray([r['keep_score'] for r in records]))[::-1][:CONFIG['MAX_PATCHES_PER_IMAGE']]
        records = [records[i] for i in order]

    patches = [r['tensor'] for r in records]
    patch_labels = [assign_patch_label(r, roi_boxes, img_w, img_h) for r in records]
    patch_boxes = [r['patch_box'] for r in records]
    keep_scores = [r['keep_score'] for r in records]
    patch_overlaps = [r['patch_overlap'] for r in records]
    roi_coverages = [r['roi_coverage'] for r in records]

    return patches, patch_labels, patch_boxes, keep_scores, patch_overlaps, roi_coverages, mode_tag


def select_records_with_fallback(candidate_records, roi_boxes, img_w, img_h):
    if len(candidate_records) == 0:
        return [], [], [], [], [], [], 'no_candidates_after_basic_filter'

    primary_thr = float(CONFIG['ROI_KEEP_OVERLAP_RATIO'])
    selected = [r for r in candidate_records if r['keep_score'] >= primary_thr]
    if len(selected) > 0:
        return finalize_records(selected, f'primary_keep_{primary_thr:.2f}', roi_boxes, img_w, img_h)

    for thr in CONFIG['ROI_KEEP_OVERLAP_RATIO_FALLBACKS']:
        thr = float(thr)
        selected = [r for r in candidate_records if r['keep_score'] >= thr]
        if len(selected) > 0:
            return finalize_records(selected, f'fallback_keep_{thr:.2f}', roi_boxes, img_w, img_h)

    k = max(1, int(CONFIG['FORCE_TOPK_IF_EMPTY']))
    order = np.argsort(np.asarray([r['keep_score'] for r in candidate_records]))[::-1][:k]
    selected = [candidate_records[i] for i in order]
    return finalize_records(selected, f'force_topk_{k}', roi_boxes, img_w, img_h)


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


def process_one_image(img_path, json_path, model, image_error_log_path=None, image_skip_log_path=None):
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

    h, w = img.shape[:2]
    roi_boxes = load_roi_boxes_for_image(json_path, h, w)

    if len(roi_boxes) == 0:
        reason = 'missing_or_empty_roi_json'
        if CONFIG['STRICT_REQUIRE_ROI']:
            if image_skip_log_path is not None:
                append_csv_row(image_skip_log_path, {
                    'img_path': img_path,
                    'reason': reason,
                    'detail': json_path,
                })
            return None

    candidate_records = build_candidate_patch_records(
        img=img,
        fan_mask=fan_mask,
        roi_boxes=roi_boxes,
    )

    if CONFIG.get('USE_ROI_TO_FILTER_BAG', False):
        patches, patch_labels, patch_boxes, keep_scores, patch_overlaps, roi_coverages, fallback_mode = select_records_with_fallback(
            candidate_records,
            roi_boxes=roi_boxes,
            img_w=w,
            img_h=h,
        )
    else:
        patches, patch_labels, patch_boxes, keep_scores, patch_overlaps, roi_coverages, fallback_mode = finalize_records(
            candidate_records,
            mode_tag='fan_fullbag_no_roi_filter',
            roi_boxes=roi_boxes,
            img_w=w,
            img_h=h,
        )

    if len(patches) == 0:
        if image_skip_log_path is not None:
            append_csv_row(image_skip_log_path, {
                'img_path': img_path,
                'reason': 'no_valid_patches_after_fullfan_filter',
                'detail': json_path,
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
        'roi_keep_scores': torch.tensor(keep_scores, dtype=torch.float32),
        'patch_overlap_ratios': torch.tensor(patch_overlaps, dtype=torch.float32),
        'roi_coverages': torch.tensor(roi_coverages, dtype=torch.float32),
        'num_patches': int(len(patches)),
        'num_core_patches': int(sum(1 for x in patch_labels if x > 0.5)),
        'img_path': img_path,
        'json_path': json_path,
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
        print(f"Detected features num: {num_clin_feats}")
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
        raise FileNotFoundError(f"DATA_ROOT no found: {CONFIG['DATA_ROOT']}")

    classes = [
        d for d in os.listdir(CONFIG['DATA_ROOT'])
        if os.path.isdir(os.path.join(CONFIG['DATA_ROOT'], d))
    ]

    summary_rows = []

    for cls_name in classes:
        label_int = CLASS_MAP.get(cls_name.lower())
        if label_int is None:
            continue

        print(f"\nclass {cls_name} --> Label: {label_int}")
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
            patient_keep_scores = []
            patient_patch_overlaps = []
            patient_roi_coverages = []
            patient_patch_img_paths = []
            patient_fallback_modes = []
            valid_image_count = 0
            total_core_patches = 0
            total_patches = 0

            for img_f in img_files:
                json_f = os.path.splitext(img_f)[0] + '.json'
                res = process_one_image(
                    img_path=img_f,
                    json_path=json_f,
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
                patient_keep_scores.append(res['roi_keep_scores'])
                patient_patch_overlaps.append(res['patch_overlap_ratios'])
                patient_roi_coverages.append(res['roi_coverages'])
                patient_patch_img_paths.extend([res['img_path']] * res['img_feats'].size(0))
                patient_fallback_modes.extend([res['fallback_mode']] * res['img_feats'].size(0))
                total_patches += int(res['num_patches'])
                total_core_patches += int(res['num_core_patches'])

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
            final_keep_scores = torch.cat(patient_keep_scores, dim=0)
            final_patch_overlaps = torch.cat(patient_patch_overlaps, dim=0)
            final_roi_coverages = torch.cat(patient_roi_coverages, dim=0)

            save_name = f"{cls_name}_{p_name}.pt"
            save_path = os.path.join(CONFIG['SAVE_DIR'], save_name)

            torch.save({
                'img_feats': final_feats,
                'clin_feats': p_clin,
                'label': label_int,
                'patch_labels': final_patch_labels,
                'patient_id': p_name,
                'patch_boxes': final_patch_boxes,
                'roi_keep_scores': final_keep_scores,
                'patch_overlap_ratios': final_patch_overlaps,
                'roi_coverages': final_roi_coverages,
                'patch_image_paths': patient_patch_img_paths,
                'patch_fallback_modes': patient_fallback_modes,
                'meta': {
                    'class_name': cls_name,
                    'num_images_total': len(img_files),
                    'num_images_valid': valid_image_count,
                    'num_patches_total': int(total_patches),
                    'num_core_patches': int(total_core_patches),
                    'use_roi_to_filter_bag': bool(CONFIG['USE_ROI_TO_FILTER_BAG']),
                    'top_cut_ratio': float(CONFIG['TOP_CUT_RATIO']),
                    'bottom_cut_ratio': float(CONFIG['BOTTOM_CUT_RATIO']),
                    'roi_keep_overlap_ratio': float(CONFIG['ROI_KEEP_OVERLAP_RATIO']),
                    'roi_keep_overlap_ratio_fallbacks': list(CONFIG['ROI_KEEP_OVERLAP_RATIO_FALLBACKS']),
                    'patch_pos_overlap_ratio': float(CONFIG['PATCH_POS_OVERLAP_RATIO']),
                    'patch_pos_roi_coverage_ratio': float(CONFIG['PATCH_POS_ROI_COVERAGE_RATIO']),
                    'roi_dilate_pixels': int(CONFIG['ROI_DILATE_PIXELS']),
                    'patch_pos_min_inter_area': float(CONFIG['PATCH_POS_MIN_INTER_AREA']),
                    'patch_pos_use_center_rule': bool(CONFIG['PATCH_POS_USE_CENTER_RULE']),
                    'force_topk_if_empty': int(CONFIG['FORCE_TOPK_IF_EMPTY']),
                    'strict_require_roi': bool(CONFIG['STRICT_REQUIRE_ROI']),
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
                'num_core_patches': int(total_core_patches),
                'use_roi_to_filter_bag': bool(CONFIG['USE_ROI_TO_FILTER_BAG']),
                'top_cut_ratio': float(CONFIG['TOP_CUT_RATIO']),
                'bottom_cut_ratio': float(CONFIG['BOTTOM_CUT_RATIO']),
                'roi_keep_overlap_ratio': float(CONFIG['ROI_KEEP_OVERLAP_RATIO']),
                'patch_pos_overlap_ratio': float(CONFIG['PATCH_POS_OVERLAP_RATIO']),
                'patch_pos_roi_coverage_ratio': float(CONFIG['PATCH_POS_ROI_COVERAGE_RATIO']),
                'roi_dilate_pixels': int(CONFIG['ROI_DILATE_PIXELS']),
                'patch_pos_min_inter_area': float(CONFIG['PATCH_POS_MIN_INTER_AREA']),
                'patch_pos_use_center_rule': bool(CONFIG['PATCH_POS_USE_CENTER_RULE']),
                'force_topk_if_empty': int(CONFIG['FORCE_TOPK_IF_EMPTY']),
                'save_path': save_path,
            })

    if len(summary_rows) > 0:
        pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        print(f"summary has been saved: {summary_csv_path}")
    else:
        print("no one has been extracted")


if __name__ == '__main__':
    export_all_patients()