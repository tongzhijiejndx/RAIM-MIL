import os
import sys
import copy
import json

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

import numpy as np
import pandas as pd
import torch

from image_only.image_only_models import build_image_only_model
from image_only.image_only_utils import (
    CONFIG,
    ALL_TASKS,
    get_task_label,
    load_all_test_pts,
    load_summary_df,
    get_thresholds_from_summary,
    load_task_checkpoints,
    compute_binary_metrics_with_ci,
    plot_save_roc_curve,
    save_roc_raw_data,
    merge_task_predictions,
    compute_group_metrics,
    save_group_confusion_matrix,
)

from image_only.image_only_ci_utils import (
    compute_binary_metrics_with_ci_v2,
    compute_group_metrics_with_ci,
)

EXPERIMENT_NAME = "CLAM"

LOCAL_CONFIG = copy.deepcopy(CONFIG)
LOCAL_CONFIG["LOG_ROOT"] = os.path.join(CUR_DIR, "logs_image_only")
LOCAL_CONFIG["SAVE_DIR"] = os.path.join(CUR_DIR, "test_results_image_only")
LOCAL_CONFIG["SUMMARY_CSV"] = os.path.join(LOCAL_CONFIG["LOG_ROOT"], "FINAL_IMAGE_ONLY_SUMMARY_ALL.csv")
LOCAL_CONFIG["THRESHOLD_MODE"] = "from_summary_csv"


def predict_one_checkpoint(model, pt_path: str):
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

    img_feats = img_feats.to(LOCAL_CONFIG["DEVICE"], non_blocking=LOCAL_CONFIG["NON_BLOCKING"])

    with torch.no_grad():
        bag_logits, A, extra = model(img_feats)
        prob = torch.softmax(bag_logits, dim=1)[0, 1].item()

    return float(prob)


def build_model_from_ckpt(ckpt):
    model_kwargs = ckpt["model_kwargs"]
    model = build_image_only_model(
        model_name=ckpt["model_name"],
        **model_kwargs,
    ).to(LOCAL_CONFIG["DEVICE"])
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


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
        for ckpt_path in ckpt_files:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model = build_model_from_ckpt(ckpt)
            fold_models.append(model)

        threshold = float(threshold_map.get(current_task, 0.5))

        rows_for_metrics = []
        rows_for_all = []

        for pt_path in test_pt_files:
            data = torch.load(pt_path, map_location="cpu")
            orig_label = int(data["label"])
            task_label = get_task_label(orig_label, current_task)

            fold_probs = []
            for model in fold_models:
                p = predict_one_checkpoint(model, pt_path)
                fold_probs.append(p)

            mean_prob = float(np.mean(fold_probs))
            hard_pred = 1 if mean_prob >= threshold else 0

            row = {
                "pt_path": pt_path,
                "file_name": os.path.basename(pt_path),
                "orig_label": orig_label,
                "task_label": task_label,
                "mean_prob": mean_prob,
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
        os.path.join(exp_save_dir, "FINAL_TEST_IMAGE_ONLY_SUMMARY.csv"),
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
        os.path.join(exp_save_dir, "FINAL_TEST_IMAGE_ONLY_3CLASS_SUMMARY.csv"),
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

    print("========== Runtime Config (CLAM Test) ==========")
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
    print("================================================")

    test_pt_files = load_all_test_pts(LOCAL_CONFIG["TEST_FEAT_DIR"])

    summary_df = load_summary_df(LOCAL_CONFIG["SUMMARY_CSV"]) if LOCAL_CONFIG["THRESHOLD_MODE"] == "from_summary_csv" else None
    threshold_map, threshold_source = get_thresholds_from_summary(summary_df, EXPERIMENT_NAME, ALL_TASKS)

    print("[Threshold Map]")
    print(json.dumps(threshold_map, ensure_ascii=False, indent=2))

    evaluate_one_experiment(test_pt_files, threshold_map)


if __name__ == "__main__":
    run_all_tests()
