import os
import sys
import copy
import json

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import pandas as pd
import torch

from image_only.image_only_models import build_image_only_model
from multimodal.multimodal_models import ClinicalMLP
from multimodal.multimodal_utils import (
    CONFIG,
    ALL_TASKS,
    get_task_label,
    load_all_test_pts,
    load_summary_df,
    get_thresholds_from_summary,
    compute_binary_metrics_with_ci,
    plot_save_roc_curve,
    save_roc_raw_data,
    merge_task_predictions,
    compute_group_metrics,
    save_group_confusion_matrix,
)

from multimodal.multimodal_ci_utils import (
    compute_binary_metrics_with_ci_v2,
    compute_group_metrics_with_ci,
)

EXPERIMENT_NAME = "LateFusion"

LOCAL_CONFIG = copy.deepcopy(CONFIG)
LOCAL_CONFIG["LOG_ROOT"] = os.path.join(CUR_DIR, "logs_multimodal")
LOCAL_CONFIG["SAVE_DIR"] = os.path.join(CUR_DIR, "test_results_multimodal")
LOCAL_CONFIG["SUMMARY_CSV"] = os.path.join(LOCAL_CONFIG["LOG_ROOT"], "FINAL_MULTIMODAL_SUMMARY_LATE_FUSION.csv")
LOCAL_CONFIG["THRESHOLD_MODE"] = "from_summary_csv"


def load_task_checkpoints(log_root: str, experiment_name: str, task_name: str):
    ckpt_dir = os.path.join(log_root, task_name, experiment_name)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    import glob
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "fold*_best.pth")))
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No fold checkpoints found in: {ckpt_dir}")
    return ckpts


def normalize_clin_array(x: np.ndarray, mean_arr: np.ndarray, std_arr: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    std_arr = np.where(np.asarray(std_arr, dtype=np.float32) == 0, 1.0, np.asarray(std_arr, dtype=np.float32))
    return (x - np.asarray(mean_arr, dtype=np.float32)) / std_arr


def build_models_from_ckpt(ckpt):
    image_model = build_image_only_model(
        model_name=ckpt["image_model_name"],
        **ckpt["image_model_kwargs"],
    ).to(LOCAL_CONFIG["DEVICE"])
    image_model.load_state_dict(ckpt["image_model_state_dict"], strict=True)
    image_model.eval()

    clinical_model = ClinicalMLP(
        **ckpt["clinical_model_kwargs"],
    ).to(LOCAL_CONFIG["DEVICE"])
    clinical_model.load_state_dict(ckpt["clinical_model_state_dict"], strict=True)
    clinical_model.eval()

    return image_model, clinical_model


def predict_one_checkpoint(image_model, clinical_model, ckpt, pt_path: str):
    data = torch.load(pt_path, map_location="cpu")

    img_feats = data["img_feats"]
    if not isinstance(img_feats, torch.Tensor):
        img_feats = torch.tensor(img_feats, dtype=torch.float32)
    else:
        img_feats = img_feats.float()

    if img_feats.ndim == 2:
        img_feats = img_feats.unsqueeze(0)

    if img_feats.size(1) > LOCAL_CONFIG["MAX_PATCHES"]:
        img_feats = img_feats[:, :LOCAL_CONFIG["MAX_PATCHES"], :]

    clin_feats = data.get("clin_feats", None)
    if clin_feats is None:
        raise ValueError(f"{pt_path} missing clin_feats")

    if isinstance(clin_feats, torch.Tensor):
        clin_feats = clin_feats.detach().cpu().numpy()
    clin_feats = np.asarray(clin_feats, dtype=np.float32).reshape(-1)
    clin_feats = normalize_clin_array(clin_feats, ckpt["clin_mean"], ckpt["clin_std"])
    clin_feats = torch.tensor(clin_feats, dtype=torch.float32).unsqueeze(0)

    img_feats = img_feats.to(LOCAL_CONFIG["DEVICE"], non_blocking=LOCAL_CONFIG["NON_BLOCKING"])
    clin_feats = clin_feats.to(LOCAL_CONFIG["DEVICE"], non_blocking=LOCAL_CONFIG["NON_BLOCKING"])

    with torch.no_grad():
        bag_logits, A, extras = image_model(img_feats)
        image_prob = torch.softmax(bag_logits, dim=1)[0, 1].item()

        clin_logits = clinical_model(clin_feats)
        clin_prob = torch.softmax(clin_logits, dim=1)[0, 1].item()

    alpha = float(ckpt["best_alpha"])
    fused_prob = alpha * float(image_prob) + (1.0 - alpha) * float(clin_prob)
    return float(fused_prob), float(image_prob), float(clin_prob), alpha


def evaluate_one_experiment(test_pt_files, threshold_map):
    exp_save_dir = os.path.join(LOCAL_CONFIG["SAVE_DIR"], EXPERIMENT_NAME)
    os.makedirs(exp_save_dir, exist_ok=True)

    task_prediction_dfs = {}
    summary_rows = []

    for current_task in ALL_TASKS:
        ckpt_files = load_task_checkpoints(
            log_root=LOCAL_CONFIG["LOG_ROOT"],
            experiment_name=EXPERIMENT_NAME,
            task_name=current_task,
        )

        fold_models = []
        fold_ckpts = []
        for ckpt_path in ckpt_files:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            image_model, clinical_model = build_models_from_ckpt(ckpt)
            fold_models.append((image_model, clinical_model))
            fold_ckpts.append(ckpt)

        threshold = float(threshold_map.get(current_task, 0.5))

        rows_for_metrics = []
        rows_for_all = []

        for pt_path in test_pt_files:
            data = torch.load(pt_path, map_location="cpu")
            orig_label = int(data["label"])
            task_label = get_task_label(orig_label, current_task)

            fold_fused_probs = []
            fold_img_probs = []
            fold_clin_probs = []
            fold_alphas = []

            for (image_model, clinical_model), ckpt in zip(fold_models, fold_ckpts):
                fused_p, img_p, clin_p, alpha = predict_one_checkpoint(
                    image_model, clinical_model, ckpt, pt_path
                )
                fold_fused_probs.append(fused_p)
                fold_img_probs.append(img_p)
                fold_clin_probs.append(clin_p)
                fold_alphas.append(alpha)

            mean_prob = float(np.mean(fold_fused_probs))
            hard_pred = 1 if mean_prob >= threshold else 0

            row = {
                "pt_path": pt_path,
                "file_name": os.path.basename(pt_path),
                "orig_label": orig_label,
                "task_label": task_label,
                "mean_prob": mean_prob,
                "mean_image_prob": float(np.mean(fold_img_probs)),
                "mean_clin_prob": float(np.mean(fold_clin_probs)),
                "mean_alpha": float(np.mean(fold_alphas)),
                "hard_pred": hard_pred,
            }
            rows_for_all.append(row)

            if task_label is not None:
                rows_for_metrics.append(row)

        df_all = pd.DataFrame(rows_for_all)
        df_metric = pd.DataFrame(rows_for_metrics)
        task_prediction_dfs[current_task] = df_all

        pred_csv = os.path.join(exp_save_dir, f"{current_task}_test_predictions.csv")
        df_all.to_csv(pred_csv, index=False, encoding="utf-8-sig")

        y_true = df_metric["task_label"].astype(int).to_numpy()
        y_prob = df_metric["mean_prob"].astype(float).to_numpy()

        metrics = compute_binary_metrics_with_ci_v2(y_true, y_prob, threshold=threshold)

        save_roc_raw_data(
            y_true, y_prob,
            os.path.join(exp_save_dir, f"{current_task}_test_roc_raw.csv"),
        )
        plot_save_roc_curve(
            y_true, y_prob,
            os.path.join(exp_save_dir, f"{current_task}_test_roc.png"),
            title=f"Test ROC | {EXPERIMENT_NAME} | {current_task}",
            auc_score=metrics["AUC"],
        )

        summary_rows.append({
            "Task": current_task,
            "Experiment": EXPERIMENT_NAME,
            "Threshold": threshold,
            **metrics,
        })

        print(
            f"[{EXPERIMENT_NAME}][{current_task}] "
            f"AUC={metrics['AUC']:.4f} | Acc={metrics['Acc']:.4f} | "
            f"Sens={metrics['Sens']:.4f} | Spec={metrics['Spec']:.4f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(exp_save_dir, "FINAL_TEST_MULTIMODAL_SUMMARY.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    df_3class = merge_task_predictions(
        task_prediction_dfs["Task1_0_vs_123"],
        task_prediction_dfs["Task2_1_vs_23"],
    )
    df_3class.to_csv(
        os.path.join(exp_save_dir, "test_predictions_3class.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    metrics_3class = compute_group_metrics_with_ci(
        df_3class["final_group_true"].to_numpy(),
        df_3class["final_group_pred"].to_numpy(),
    )
    metrics_3class_df = pd.DataFrame([{
        "Experiment": EXPERIMENT_NAME,
        **metrics_3class,
    }])
    metrics_3class_df.to_csv(
        os.path.join(exp_save_dir, "FINAL_TEST_MULTIMODAL_3CLASS_SUMMARY.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    save_group_confusion_matrix(
        df_3class["final_group_true"].to_numpy(),
        df_3class["final_group_pred"].to_numpy(),
        os.path.join(exp_save_dir, "test_3class_confusion_matrix.png"),
    )

    print("\n[3-Class]")
    print(metrics_3class_df[["Experiment", "Acc", "Weighted_F1", "Cohen_Kappa"]])

    return summary_df, metrics_3class_df


def run_all_tests():
    os.makedirs(LOCAL_CONFIG["SAVE_DIR"], exist_ok=True)

    print("========== Runtime Config (Late Fusion Test) ==========")
    print(json.dumps({
        "TEST_FEAT_DIR": LOCAL_CONFIG["TEST_FEAT_DIR"],
        "LOG_ROOT": LOCAL_CONFIG["LOG_ROOT"],
        "SAVE_DIR": LOCAL_CONFIG["SAVE_DIR"],
        "TASKS": ALL_TASKS,
        "BOOTSTRAP_N": LOCAL_CONFIG["BOOTSTRAP_N"],
        "BOOTSTRAP_ALPHA": LOCAL_CONFIG["BOOTSTRAP_ALPHA"],
        "BOOTSTRAP_SEED": LOCAL_CONFIG["BOOTSTRAP_SEED"],
        "THRESHOLD_MODE": LOCAL_CONFIG["THRESHOLD_MODE"],
        "SUMMARY_CSV": LOCAL_CONFIG["SUMMARY_CSV"],
    }, ensure_ascii=False, indent=2))
    print("=======================================================")

    test_pt_files = load_all_test_pts(LOCAL_CONFIG["TEST_FEAT_DIR"])

    summary_df = load_summary_df(LOCAL_CONFIG["SUMMARY_CSV"]) if LOCAL_CONFIG["THRESHOLD_MODE"] == "from_summary_csv" else None
    threshold_map, threshold_source = get_thresholds_from_summary(summary_df, EXPERIMENT_NAME, ALL_TASKS)

    print("[Threshold Map]")
    print(json.dumps(threshold_map, ensure_ascii=False, indent=2))

    evaluate_one_experiment(test_pt_files, threshold_map)


if __name__ == "__main__":
    run_all_tests()