import pandas as pd
import numpy as np
import cv2
import json
import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def cv_imread_safe(file_path):
    try:
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def get_ultrasound_fan_mask(img_cv2, fan_threshold=15):
    if img_cv2 is None: return None, None

    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, fan_threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c_max = max(contours, key=cv2.contourArea)
        mask_clean = np.zeros_like(mask)
        cv2.drawContours(mask_clean, [c_max], -1, 255, -1)
        mask = mask_clean
    else:
        return None, None

    return mask, None


def load_json_boxes(json_path):
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        boxes = []
        for shape in data.get('shapes', []):
            if shape['shape_type'] == 'rectangle':
                points = shape['points']
                x_vals = [p[0] for p in points]
                y_vals = [p[1] for p in points]
                x1, y1 = min(x_vals), min(y_vals)
                x2, y2 = max(x_vals), max(y_vals)
                boxes.append([x1, y1, x2, y2])
        return boxes
    except Exception as e:
        print(f"Error loading json {json_path}: {e}")
        return []


def expand_box(box, max_w, max_h, ratio=1.2):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2

    new_w = w * ratio
    new_h = h * ratio

    new_x1 = max(0, int(cx - new_w / 2))
    new_y1 = max(0, int(cy - new_h / 2))
    new_x2 = min(max_w, int(cx + new_w / 2))
    new_y2 = min(max_h, int(cy + new_h / 2))

    return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]


def calculate_metrics(y_true, y_pred):
    if isinstance(y_true, torch.Tensor): y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor): y_pred = y_pred.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }


def save_validation_details(val_files, val_true, val_probs, threshold, save_path):
    detailed_list = []

    for i in range(len(val_files)):
        path = val_files[i]
        fname = os.path.basename(path)
        prob = val_probs[i]
        true_lab = val_true[i]

        pred_lab = 1 if prob >= threshold else 0

        detailed_list.append({
            'filename': fname,
            'true_label': true_lab,
            'predict_label': pred_lab,
            'prob_score': round(prob, 4),
            'result': 'Correct' if true_lab == pred_lab else 'Wrong'
        })

    df_details = pd.DataFrame(detailed_list)
    df_details.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"Validation details saved to: {save_path}")


def detect_border_text_mask(
    img_cv2,
    top_ratio=0.18,
    bottom_ratio=0.18,
    side_ratio=0.12,
    min_area=20,
    max_area=20000,
    dilate_kernel=(7, 7),
):
    if img_cv2 is None:
        return None

    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]

    border_mask = np.zeros((H, W), dtype=np.uint8)
    top_h = int(H * top_ratio)
    bottom_h = int(H * bottom_ratio)
    side_w = int(W * side_ratio)

    border_mask[:top_h, :] = 255
    border_mask[H - bottom_h:, :] = 255
    border_mask[:, :side_w] = 255
    border_mask[:, W - side_w:] = 255

    border_gray = cv2.bitwise_and(gray, gray, mask=border_mask)

    _, th = cv2.threshold(border_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k_open)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    out_mask = np.zeros_like(th, dtype=np.uint8)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area < min_area or area > max_area:
            continue

        comp_mask = (labels == i).astype(np.uint8) * 255
        overlap = cv2.countNonZero(cv2.bitwise_and(comp_mask, border_mask))

        if overlap / max(area, 1) < 0.9:
            continue

        if w > W * 0.5 and h > H * 0.2:
            continue

        out_mask[labels == i] = 255

    dk = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_kernel)
    out_mask = cv2.dilate(out_mask, dk, iterations=1)

    return out_mask


def blackout_text_regions(img_cv2, text_mask, fill_value=0):
    if img_cv2 is None:
        return None
    if text_mask is None:
        return img_cv2.copy()

    out = img_cv2.copy()
    out[text_mask > 0] = fill_value
    return out