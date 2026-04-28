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
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from image_only.image_only_models import build_image_only_model
from image_only.image_only_utils import (
    CONFIG,
    ALL_TASKS,
    get_task_policy,
    get_task_label,
    seed_everything,
    load_all_train_pt_files,
    build_task_file_list,
    build_train_loader,
    build_val_loader,
    find_best_threshold,
    plot_save_roc_curve,
    save_roc_raw_data,
    format_mean_std,
    FocalLoss,
    DummyScaler,
    train_one_epoch,
    evaluate_one_epoch,
    maybe_compile,
)
from data_loader import FeatureBagDataset


EXPERIMENT_NAME = "TransMIL"
MODEL_NAME = "transmil"

LOCAL_CONFIG = copy.deepcopy(CONFIG)
LOCAL_CONFIG["LOG_ROOT"] = os.path.join(CUR_DIR, "logs_image_only")
LOCAL_CONFIG["EPOCHS"] = 25


def run_all():
    seed_everything(LOCAL_CONFIG["SEED"])
    os.makedirs(LOCAL_CONFIG["LOG_ROOT"], exist_ok=True)

    print("========== Runtime Config (TransMIL Train) ==========")
    print(json.dumps(LOCAL_CONFIG, ensure_ascii=False, indent=2))
    print("====================================================")

    all_pt_files = load_all_train_pt_files(LOCAL_CONFIG["TRAIN_FEAT_DIR"])
    final_summary_rows = []

    for current_task in ALL_TASKS:
        policy = get_task_policy(current_task)
        task_files, task_labels = build_task_file_list(all_pt_files, current_task)

        if len(task_files) == 0:
            raise RuntimeError(f"No usable samples for task: {current_task}")

        y = np.asarray(task_labels).astype(int)

        print(f"\n{'=' * 60}")
        print(f"TASK: {current_task}")
        print(f"{'=' * 60}")
        print(f"usable samples = {len(task_files)} | class-0 = {(y == 0).sum()} | class-1 = {(y == 1).sum()}")

        exp_save_dir = os.path.join(LOCAL_CONFIG["LOG_ROOT"], current_task, EXPERIMENT_NAME)
        os.makedirs(exp_save_dir, exist_ok=True)

        skf = StratifiedKFold(
            n_splits=LOCAL_CONFIG["FOLDS"],
            shuffle=True,
            random_state=LOCAL_CONFIG["SEED"],
        )

        fold_rows = []
        oof_true, oof_prob, oof_name = [], [], []
        fold_thresholds = []

        fold_splits = list(skf.split(task_files, y))
        fold_pbar = tqdm(
            enumerate(fold_splits, start=1),
            total=len(fold_splits),
            desc=f"{EXPERIMENT_NAME} | {current_task}",
            unit="fold",
            leave=True,
            dynamic_ncols=True,
            ascii=True,
        )

        for fold, (train_idx, val_idx) in fold_pbar:
            fold_pbar.set_postfix_str(f"Fold {fold}/{len(fold_splits)}")
            train_files = [task_files[i] for i in train_idx]
            val_files = [task_files[i] for i in val_idx]
            y_train_fold = y[train_idx]

            train_dataset = FeatureBagDataset(
                pt_files=train_files,
                clin_data_dict=None,
                is_train=True,
                strict_clin_match=False,
            )
            val_dataset = FeatureBagDataset(
                pt_files=val_files,
                clin_data_dict=None,
                is_train=False,
                strict_clin_match=False,
            )

            train_loader = build_train_loader(train_dataset, y_train_fold, policy)
            val_loader = build_val_loader(val_dataset)

            model = build_image_only_model(
                model_name=MODEL_NAME,
                in_dim=2048,
                n_classes=2,
                L=512,
                dropout=0.25,
                num_layers=2,
                num_heads=8,
                ff_dim=1024,
                max_tokens=4096,
                use_layernorm=True,
                use_l2norm=False,
            ).to(LOCAL_CONFIG["DEVICE"])
            model = maybe_compile(model)

            class_weights = torch.tensor(
                [1.0, float(policy["class1_weight"])],
                dtype=torch.float32,
                device=LOCAL_CONFIG["DEVICE"],
            )
            criterion_bag = FocalLoss(alpha=class_weights, gamma=2.0, reduction="mean")

            optimizer = optim.Adam(
                model.parameters(),
                lr=LOCAL_CONFIG["LR"],
                weight_decay=LOCAL_CONFIG["WEIGHT_DECAY"],
            )

            scaler = torch.cuda.amp.GradScaler() if LOCAL_CONFIG["AMP"] and str(LOCAL_CONFIG["DEVICE"]).startswith("cuda") else DummyScaler()

            best_score = -1.0
            best_epoch = -1
            best_ckpt = None
            history_rows = []

            epoch_pbar = tqdm(
                range(1, LOCAL_CONFIG["EPOCHS"] + 1),
                total=LOCAL_CONFIG["EPOCHS"],
                desc=f"{current_task} | Fold {fold}",
                unit="epoch",
                leave=False,
                dynamic_ncols=True,
                ascii=True,
            )

            for epoch in epoch_pbar:
                train_loss = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    scaler=scaler,
                    criterion_bag=criterion_bag,
                    current_task=current_task,
                    use_clam_inst_loss=False,
                    clam_inst_weight=0.0,
                )

                val_out = evaluate_one_epoch(
                    model=model,
                    loader=val_loader,
                    criterion_bag=criterion_bag,
                    current_task=current_task,
                    use_clam_inst_loss=False,
                    clam_inst_weight=0.0,
                )

                val_auc = float(val_out["val_auc"])
                val_loss = float(val_out["val_loss"])
                best_threshold = float(val_out["best_threshold"])
                metric_dict = val_out["metric_dict"]

                history_rows.append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                    "val_acc": metric_dict["acc"],
                    "val_sens": metric_dict["sens"],
                    "val_spec": metric_dict["spec"],
                    "val_prec": metric_dict["prec"],
                    "val_f1": metric_dict["f1"],
                    "val_bal_acc": metric_dict["bal_acc"],
                    "best_threshold": best_threshold,
                })

                score = val_auc if policy["best_model_mode"] == "auc" else metric_dict["acc"]


                epoch_pbar.set_postfix({
                    "AUC": f"{val_auc:.4f}" if "val_auc" in locals() else f"{val_out['val_auc']:.4f}",
                    "Acc": f"{metric_dict['acc']:.4f}",
                    "F1": f"{metric_dict['f1']:.4f}",
                    "Th": f"{best_threshold:.3f}" if "best_threshold" in locals() else f"{val_out['best_threshold']:.3f}",
                })

                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_ckpt = {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "best_score": float(best_score),
                        "best_threshold": float(best_threshold),
                        "task_name": current_task,
                        "experiment_name": EXPERIMENT_NAME,
                        "model_name": MODEL_NAME,
                        "model_kwargs": {
                            "in_dim": 2048,
                            "n_classes": 2,
                            "L": 512,
                            "dropout": 0.25,
                            "num_layers": 2,
                            "num_heads": 8,
                            "ff_dim": 1024,
                            "max_tokens": 4096,
                            "use_layernorm": True,
                            "use_l2norm": False,
                        },
                    }

            if best_ckpt is None:
                raise RuntimeError(f"Fold {fold} failed to produce a checkpoint.")

            ckpt_path = os.path.join(exp_save_dir, f"fold{fold}_best.pth")
            torch.save(best_ckpt, ckpt_path)

            history_df = pd.DataFrame(history_rows)
            history_df.to_csv(
                os.path.join(exp_save_dir, f"fold{fold}_history.csv"),
                index=False,
                encoding="utf-8-sig",
            )

            best_history_row = history_df.loc[history_df["epoch"] == best_epoch].iloc[0]
            fold_thresholds.append(float(best_ckpt["best_threshold"]))

            model.load_state_dict(best_ckpt["model_state_dict"])
            model.eval()

            fold_val_probs, fold_val_true, fold_val_names = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    img_feats = batch["img_feats"].to(LOCAL_CONFIG["DEVICE"])
                    labels = batch["label"]
                    file_name = batch["file_name"][0]

                    if img_feats.size(1) > LOCAL_CONFIG["MAX_PATCHES"]:
                        img_feats = img_feats[:, :LOCAL_CONFIG["MAX_PATCHES"], :]

                    bag_logits, A, extra = model(img_feats)
                    prob = torch.softmax(bag_logits, dim=1)[0, 1].item()

                    task_true = get_task_label(int(labels.item()), current_task)
                    fold_val_probs.append(float(prob))
                    fold_val_true.append(int(task_true))
                    fold_val_names.append(file_name)

            fold_pred_df = pd.DataFrame({
                "file_name": fold_val_names,
                "y_true": fold_val_true,
                "y_prob": fold_val_probs,
            })
            fold_pred_df.to_csv(
                os.path.join(exp_save_dir, f"fold{fold}_validation_predictions.csv"),
                index=False,
                encoding="utf-8-sig",
            )

            fold_auc = roc_auc_score(fold_val_true, fold_val_probs) if len(np.unique(fold_val_true)) > 1 else 0.5
            save_roc_raw_data(
                fold_val_true,
                fold_val_probs,
                os.path.join(exp_save_dir, f"fold{fold}_validation_roc_raw.csv"),
            )
            plot_save_roc_curve(
                fold_val_true,
                fold_val_probs,
                os.path.join(exp_save_dir, f"fold{fold}_validation_roc.png"),
                title=f"{current_task} | {EXPERIMENT_NAME} | Fold {fold}",
                auc_score=fold_auc,
            )

            fold_rows.append({
                "fold": fold,
                "best_epoch": int(best_epoch),
                "auc": float(best_history_row["val_auc"]),
                "acc": float(best_history_row["val_acc"]),
                "sens": float(best_history_row["val_sens"]),
                "spec": float(best_history_row["val_spec"]),
                "prec": float(best_history_row["val_prec"]),
                "f1": float(best_history_row["val_f1"]),
                "bal_acc": float(best_history_row["val_bal_acc"]),
                "threshold": float(best_ckpt["best_threshold"]),
                "n_val": int(len(fold_val_true)),
            })

            oof_true.extend(fold_val_true)
            oof_prob.extend(fold_val_probs)
            oof_name.extend(fold_val_names)

        fold_df = pd.DataFrame(fold_rows)
        fold_df.to_csv(os.path.join(exp_save_dir, "fold_metrics.csv"), index=False, encoding="utf-8-sig")

        oof_pred_df = pd.DataFrame({
            "file_name": oof_name,
            "y_true": oof_true,
            "y_prob": oof_prob,
        })
        oof_pred_df.to_csv(os.path.join(exp_save_dir, "oof_predictions.csv"), index=False, encoding="utf-8-sig")

        if len(np.unique(np.asarray(oof_true).astype(int))) > 1:
            oof_auc = roc_auc_score(oof_true, oof_prob)
        else:
            oof_auc = 0.5

        oof_best_th, oof_metric_dict = find_best_threshold(
            oof_true,
            oof_prob,
            mode=policy["threshold_search_mode"],
            n_steps=LOCAL_CONFIG["THRESHOLD_STEPS"],
            task_name=current_task,
        )

        save_roc_raw_data(oof_true, oof_prob, os.path.join(exp_save_dir, "oof_roc_raw.csv"))
        plot_save_roc_curve(
            oof_true,
            oof_prob,
            os.path.join(exp_save_dir, "oof_roc.png"),
            title=f"OOF ROC | {current_task} | {EXPERIMENT_NAME}",
            auc_score=oof_auc,
        )

        final_summary_rows.append({
            "Task": current_task,
            "Experiment": EXPERIMENT_NAME,
            "CV_AUC_Mean": float(fold_df["auc"].mean()),
            "CV_AUC_Std": float(fold_df["auc"].std(ddof=1)),
            "CV_AUC_MeanStd": format_mean_std(fold_df["auc"].tolist()),
            "CV_Acc_Mean": float(fold_df["acc"].mean()),
            "CV_Acc_Std": float(fold_df["acc"].std(ddof=1)),
            "CV_Acc_MeanStd": format_mean_std(fold_df["acc"].tolist()),
            "CV_Sens_Mean": float(fold_df["sens"].mean()),
            "CV_Sens_Std": float(fold_df["sens"].std(ddof=1)),
            "CV_Sens_MeanStd": format_mean_std(fold_df["sens"].tolist()),
            "CV_Spec_Mean": float(fold_df["spec"].mean()),
            "CV_Spec_Std": float(fold_df["spec"].std(ddof=1)),
            "CV_Spec_MeanStd": format_mean_std(fold_df["spec"].tolist()),
            "CV_F1_Mean": float(fold_df["f1"].mean()),
            "CV_F1_Std": float(fold_df["f1"].std(ddof=1)),
            "CV_F1_MeanStd": format_mean_std(fold_df["f1"].tolist()),
            "Threshold": float(oof_best_th),
            "FoldThresholdMean": float(np.mean(fold_thresholds)) if fold_thresholds else 0.5,
            "NumSamples": int(len(oof_true)),
            "OOF_AUC": float(oof_auc),
            "OOF_Acc": float(oof_metric_dict["acc"]),
            "OOF_Sens": float(oof_metric_dict["sens"]),
            "OOF_Spec": float(oof_metric_dict["spec"]),
            "OOF_F1": float(oof_metric_dict["f1"]),
        })

    summary_df = pd.DataFrame(final_summary_rows)
    summary_csv = os.path.join(LOCAL_CONFIG["LOG_ROOT"], "FINAL_IMAGE_ONLY_SUMMARY_TRANSMIL.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print(f"\n Final summary saved to: {summary_csv}")
    print(summary_df[[
        "Task", "Experiment",
        "CV_AUC_MeanStd",
        "CV_Acc_MeanStd",
        "CV_Sens_MeanStd",
        "CV_Spec_MeanStd",
        "CV_F1_MeanStd",
        "Threshold",
    ]])


if __name__ == "__main__":
    run_all()
