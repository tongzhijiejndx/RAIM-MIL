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
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from image_only.image_only_models import build_image_only_model
from multimodal.multimodal_models import ClinicalMLP
from multimodal.multimodal_utils import (
    CONFIG,
    ALL_TASKS,
    get_task_policy,
    get_task_label,
    seed_everything,
    load_all_train_pt_files,
    build_task_file_list,
    build_clin_stats_from_train_files,
    build_clin_data_dict_from_files,
    build_train_loader,
    build_val_loader,
    maybe_compile,
    maybe_autocast,
    to_device,
    FocalLoss,
    DummyScaler,
    compute_binary_metrics,
    find_best_threshold,
    plot_save_roc_curve,
    save_roc_raw_data,
    format_mean_std,
)
from data_loader import FeatureBagDataset


EXPERIMENT_NAME = "LateFusion"
IMAGE_MODEL_NAME = "abmil"

LOCAL_CONFIG = copy.deepcopy(CONFIG)
LOCAL_CONFIG["LOG_ROOT"] = os.path.join(CUR_DIR, "logs_multimodal")
LOCAL_CONFIG["EPOCHS"] = 25
LOCAL_CONFIG["ALPHA_GRID"] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def build_image_model():
    model = build_image_only_model(
        model_name=IMAGE_MODEL_NAME,
        in_dim=2048,
        n_classes=2,
        L=512,
        D=128,
        dropout=0.25,
        use_layernorm=True,
        use_l2norm=False,
    ).to(LOCAL_CONFIG["DEVICE"])
    model = maybe_compile(model)
    return model


def build_clinical_model(clin_dim: int):
    model = ClinicalMLP(
        clin_dim=clin_dim,
        n_classes=2,
        dropout=0.20,
    ).to(LOCAL_CONFIG["DEVICE"])
    model = maybe_compile(model)
    return model


def train_one_epoch_image(model, loader, optimizer, scaler, criterion_bag, current_task):
    model.train()
    total_loss = 0.0
    n_case = 0

    for batch in loader:
        img_feats = to_device(batch["img_feats"])
        labels = to_device(batch["label"])

        target = torch.tensor(
            [get_task_label(int(labels.item()), current_task)],
            device=LOCAL_CONFIG["DEVICE"],
            dtype=torch.long,
        )

        if img_feats.size(1) > LOCAL_CONFIG["MAX_PATCHES"]:
            idx = torch.randperm(img_feats.size(1), device=img_feats.device)[:LOCAL_CONFIG["MAX_PATCHES"]]
            img_feats = img_feats[:, idx, :]

        optimizer.zero_grad(set_to_none=True)

        with maybe_autocast():
            bag_logits, A, extras = model(img_feats)
            loss = criterion_bag(bag_logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        n_case += 1

    return total_loss / max(n_case, 1)


def train_one_epoch_clinical(model, loader, optimizer, scaler, criterion_bag, current_task):
    model.train()
    total_loss = 0.0
    n_case = 0

    for batch in loader:
        clin_feats = to_device(batch["clin_feats"])
        labels = to_device(batch["label"])

        target = torch.tensor(
            [get_task_label(int(labels.item()), current_task)],
            device=LOCAL_CONFIG["DEVICE"],
            dtype=torch.long,
        )

        optimizer.zero_grad(set_to_none=True)

        with maybe_autocast():
            logits = model(clin_feats)
            loss = criterion_bag(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        n_case += 1

    return total_loss / max(n_case, 1)


@torch.no_grad()
def predict_val_probs_image(model, loader, current_task):
    model.eval()
    val_probs, val_true = [], []

    for batch in loader:
        img_feats = to_device(batch["img_feats"])
        labels = batch["label"]

        if img_feats.size(1) > LOCAL_CONFIG["MAX_PATCHES"]:
            img_feats = img_feats[:, :LOCAL_CONFIG["MAX_PATCHES"], :]

        with maybe_autocast():
            bag_logits, A, extras = model(img_feats)
            prob = torch.softmax(bag_logits, dim=1)[0, 1].item()

        val_probs.append(float(prob))
        val_true.append(int(get_task_label(int(labels.item()), current_task)))

    return val_probs, val_true


@torch.no_grad()
def predict_val_probs_clinical(model, loader, current_task):
    model.eval()
    val_probs, val_true = [], []

    for batch in loader:
        clin_feats = to_device(batch["clin_feats"])
        labels = batch["label"]

        with maybe_autocast():
            logits = model(clin_feats)
            prob = torch.softmax(logits, dim=1)[0, 1].item()

        val_probs.append(float(prob))
        val_true.append(int(get_task_label(int(labels.item()), current_task)))

    return val_probs, val_true


def search_best_late_fusion(val_true, image_probs, clin_probs, policy, alpha_grid):
    val_true = np.asarray(val_true).astype(int)
    image_probs = np.asarray(image_probs).astype(float)
    clin_probs = np.asarray(clin_probs).astype(float)

    best_alpha = 0.5
    best_threshold = 0.5
    best_metric_dict = None
    best_auc = -1.0
    best_score = -1.0

    for alpha in alpha_grid:
        fused = alpha * image_probs + (1.0 - alpha) * clin_probs

        auc = 0.5
        if len(np.unique(val_true)) > 1:
            try:
                auc = roc_auc_score(val_true, fused)
            except Exception:
                auc = 0.5

        th, metric_dict = find_best_threshold(
            val_true,
            fused,
            mode=policy["threshold_search_mode"],
            n_steps=LOCAL_CONFIG["THRESHOLD_STEPS"],
            task_name="",
        )

        score = auc if policy["best_model_mode"] == "auc" else metric_dict["acc"]

        if score > best_score:
            best_score = score
            best_auc = float(auc)
            best_alpha = float(alpha)
            best_threshold = float(th)
            best_metric_dict = metric_dict

    return best_alpha, best_threshold, best_auc, best_metric_dict


def run_all():
    seed_everything(LOCAL_CONFIG["SEED"])
    os.makedirs(LOCAL_CONFIG["LOG_ROOT"], exist_ok=True)

    print("========== Runtime Config (Late Fusion Train) ==========")
    print(json.dumps(LOCAL_CONFIG, ensure_ascii=False, indent=2))
    print("========================================================")

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
        fold_alphas = []

        fold_splits = list(skf.split(task_files, y))
        fold_pbar = tqdm(
            enumerate(fold_splits, start=1),
            total=len(fold_splits),
            desc=f"{EXPERIMENT_NAME} | {current_task}",
            unit="fold",
            leave=True,
            ascii=True,
            dynamic_ncols=False,
            ncols=110,
        )

        for fold, (train_idx, val_idx) in fold_pbar:
            fold_pbar.set_postfix_str(f"Fold {fold}/{len(fold_splits)}")

            train_files = [task_files[i] for i in train_idx]
            val_files = [task_files[i] for i in val_idx]
            y_train_fold = y[train_idx]

            clin_mean, clin_std = build_clin_stats_from_train_files(train_files)
            clin_dim = int(len(clin_mean))

            train_clin_dict = build_clin_data_dict_from_files(train_files, clin_mean, clin_std)
            val_clin_dict = build_clin_data_dict_from_files(val_files, clin_mean, clin_std)

            train_dataset = FeatureBagDataset(
                pt_files=train_files,
                clin_data_dict=train_clin_dict,
                is_train=True,
                strict_clin_match=True,
            )
            val_dataset = FeatureBagDataset(
                pt_files=val_files,
                clin_data_dict=val_clin_dict,
                is_train=False,
                strict_clin_match=True,
            )

            train_loader = build_train_loader(train_dataset, y_train_fold, policy)
            val_loader = build_val_loader(val_dataset)

            image_model = build_image_model()
            clin_model = build_clinical_model(clin_dim)

            class_weights = torch.tensor(
                [1.0, float(policy["class1_weight"])],
                dtype=torch.float32,
                device=LOCAL_CONFIG["DEVICE"],
            )
            criterion_bag = FocalLoss(alpha=class_weights, gamma=2.0, reduction="mean")

            image_optimizer = optim.Adam(
                image_model.parameters(),
                lr=LOCAL_CONFIG["LR"],
                weight_decay=LOCAL_CONFIG["WEIGHT_DECAY"],
            )
            clin_optimizer = optim.Adam(
                clin_model.parameters(),
                lr=LOCAL_CONFIG["LR"],
                weight_decay=LOCAL_CONFIG["WEIGHT_DECAY"],
            )

            image_scaler = torch.cuda.amp.GradScaler() if LOCAL_CONFIG["AMP"] and str(LOCAL_CONFIG["DEVICE"]).startswith("cuda") else DummyScaler()
            clin_scaler = torch.cuda.amp.GradScaler() if LOCAL_CONFIG["AMP"] and str(LOCAL_CONFIG["DEVICE"]).startswith("cuda") else DummyScaler()

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
                ascii=True,
                dynamic_ncols=False,
                ncols=110,
            )

            for epoch in epoch_pbar:
                train_loss_img = train_one_epoch_image(
                    model=image_model,
                    loader=train_loader,
                    optimizer=image_optimizer,
                    scaler=image_scaler,
                    criterion_bag=criterion_bag,
                    current_task=current_task,
                )
                train_loss_clin = train_one_epoch_clinical(
                    model=clin_model,
                    loader=train_loader,
                    optimizer=clin_optimizer,
                    scaler=clin_scaler,
                    criterion_bag=criterion_bag,
                    current_task=current_task,
                )

                val_probs_img, val_true_img = predict_val_probs_image(
                    image_model, val_loader, current_task
                )
                val_probs_clin, val_true_clin = predict_val_probs_clinical(
                    clin_model, val_loader, current_task
                )

                if list(val_true_img) != list(val_true_clin):
                    raise RuntimeError("Validation labels from image and clinical branches are inconsistent.")

                val_true = np.asarray(val_true_img).astype(int)
                alpha_star, threshold_star, fused_auc, metric_dict = search_best_late_fusion(
                    val_true=val_true,
                    image_probs=np.asarray(val_probs_img, dtype=float),
                    clin_probs=np.asarray(val_probs_clin, dtype=float),
                    policy=policy,
                    alpha_grid=LOCAL_CONFIG["ALPHA_GRID"],
                )

                fused_prob = alpha_star * np.asarray(val_probs_img) + (1.0 - alpha_star) * np.asarray(val_probs_clin)
                y_pred = (fused_prob >= threshold_star).astype(int)

                val_loss = 0.5 * (float(train_loss_img) + float(train_loss_clin))

                history_rows.append({
                    "epoch": epoch,
                    "train_loss_image": float(train_loss_img),
                    "train_loss_clinical": float(train_loss_clin),
                    "proxy_val_loss": float(val_loss),
                    "val_auc": float(fused_auc),
                    "val_acc": float(metric_dict["acc"]),
                    "val_sens": float(metric_dict["sens"]),
                    "val_spec": float(metric_dict["spec"]),
                    "val_prec": float(metric_dict["prec"]),
                    "val_f1": float(metric_dict["f1"]),
                    "val_bal_acc": float(metric_dict["bal_acc"]),
                    "best_alpha": float(alpha_star),
                    "best_threshold": float(threshold_star),
                })

                score = fused_auc if policy["best_model_mode"] == "auc" else metric_dict["acc"]

                epoch_pbar.set_postfix({
                    "AUC": f"{fused_auc:.4f}",
                    "Acc": f"{metric_dict['acc']:.4f}",
                    "F1": f"{metric_dict['f1']:.4f}",
                    "a": f"{alpha_star:.2f}",
                    "Th": f"{threshold_star:.3f}",
                })

                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_ckpt = {
                        "epoch": epoch,
                        "image_model_state_dict": copy.deepcopy(image_model.state_dict()),
                        "clinical_model_state_dict": copy.deepcopy(clin_model.state_dict()),
                        "best_score": float(best_score),
                        "best_threshold": float(threshold_star),
                        "best_alpha": float(alpha_star),
                        "task_name": current_task,
                        "experiment_name": EXPERIMENT_NAME,
                        "clin_dim": clin_dim,
                        "clin_mean": clin_mean.astype(np.float32),
                        "clin_std": clin_std.astype(np.float32),
                        "image_model_name": IMAGE_MODEL_NAME,
                        "image_model_kwargs": {
                            "in_dim": 2048,
                            "n_classes": 2,
                            "L": 512,
                            "D": 128,
                            "dropout": 0.25,
                            "use_layernorm": True,
                            "use_l2norm": False,
                        },
                        "clinical_model_kwargs": {
                            "clin_dim": clin_dim,
                            "n_classes": 2,
                            "dropout": 0.20,
                        },
                        "val_probs_image": list(map(float, val_probs_img)),
                        "val_probs_clinical": list(map(float, val_probs_clin)),
                        "val_true": list(map(int, val_true.tolist())),
                        "val_files": list(val_files),
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
            fold_alphas.append(float(best_ckpt["best_alpha"]))

            fused_val_probs = (
                best_ckpt["best_alpha"] * np.asarray(best_ckpt["val_probs_image"], dtype=float)
                + (1.0 - best_ckpt["best_alpha"]) * np.asarray(best_ckpt["val_probs_clinical"], dtype=float)
            )
            val_true_fold = np.asarray(best_ckpt["val_true"]).astype(int)

            fold_pred_df = pd.DataFrame({
                "file_name": [os.path.basename(x) for x in best_ckpt["val_files"]],
                "y_true": val_true_fold,
                "y_prob": fused_val_probs,
            })
            fold_pred_df.to_csv(
                os.path.join(exp_save_dir, f"fold{fold}_validation_predictions.csv"),
                index=False,
                encoding="utf-8-sig",
            )

            fold_auc = 0.5
            if len(np.unique(val_true_fold)) > 1:
                fold_auc = roc_auc_score(val_true_fold, fused_val_probs)

            save_roc_raw_data(
                val_true_fold,
                fused_val_probs,
                os.path.join(exp_save_dir, f"fold{fold}_validation_roc_raw.csv"),
            )
            plot_save_roc_curve(
                val_true_fold,
                fused_val_probs,
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
                "alpha": float(best_ckpt["best_alpha"]),
                "n_val": int(len(val_true_fold)),
            })

            oof_true.extend(val_true_fold.tolist())
            oof_prob.extend(fused_val_probs.tolist())
            oof_name.extend([os.path.basename(x) for x in best_ckpt["val_files"]])

        fold_df = pd.DataFrame(fold_rows)
        fold_df.to_csv(
            os.path.join(exp_save_dir, "fold_metrics.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        oof_pred_df = pd.DataFrame({
            "file_name": oof_name,
            "y_true": oof_true,
            "y_prob": oof_prob,
        })
        oof_pred_df.to_csv(
            os.path.join(exp_save_dir, "oof_predictions.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        oof_auc = 0.5
        if len(np.unique(np.asarray(oof_true).astype(int))) > 1:
            oof_auc = roc_auc_score(oof_true, oof_prob)

        oof_best_th, oof_metric_dict = find_best_threshold(
            oof_true,
            oof_prob,
            mode=policy["threshold_search_mode"],
            n_steps=LOCAL_CONFIG["THRESHOLD_STEPS"],
            task_name=current_task,
        )

        save_roc_raw_data(
            oof_true,
            oof_prob,
            os.path.join(exp_save_dir, "oof_roc_raw.csv"),
        )
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
            "AlphaMean": float(np.mean(fold_alphas)) if fold_alphas else 0.5,
            "NumSamples": int(len(oof_true)),
            "OOF_AUC": float(oof_auc),
            "OOF_Acc": float(oof_metric_dict["acc"]),
            "OOF_Sens": float(oof_metric_dict["sens"]),
            "OOF_Spec": float(oof_metric_dict["spec"]),
            "OOF_F1": float(oof_metric_dict["f1"]),
        })

    summary_df = pd.DataFrame(final_summary_rows)
    summary_csv = os.path.join(LOCAL_CONFIG["LOG_ROOT"], "FINAL_MULTIMODAL_SUMMARY_LATE_FUSION.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print(f"\n[OK] Final summary saved to: {summary_csv}")
    print(summary_df[[
        "Task", "Experiment",
        "CV_AUC_MeanStd",
        "CV_Acc_MeanStd",
        "CV_Sens_MeanStd",
        "CV_Spec_MeanStd",
        "CV_F1_MeanStd",
        "Threshold",
        "AlphaMean",
    ]])


if __name__ == "__main__":
    run_all()