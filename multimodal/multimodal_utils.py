import os
import glob
import json
import random
import warnings
import copy
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    recall_score,
    confusion_matrix,
    precision_score,
    f1_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import FeatureBagDataset

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message=".*scipy._lib.messagestream.MessageStream size changed.*",
    category=RuntimeWarning,
)

ALL_TASKS = [
    "Task1_0_vs_123",
    "Task2_1_vs_23",
]

TASK_POLICIES = {
    "Task1_0_vs_123": {
        "use_attn_prior": True,
        "alpha_attn": 0.10,
        "attn_warmup_epochs": 2,
        "attn_prior_eps": 0.05,
        "attn_supervision_mode": "positive_only",
        "use_neg_suppress": True,
        "beta_neg": 0.03,
        "neg_warmup_epochs": 2,
        "neg_margin": 0.15,
        "neg_supervision_mode": "positive_only",
        "neg_topk_ratio": 0.20,
        "best_model_mode": "auc",
        "threshold_search_mode": "auc",
        "class1_weight": 1.0,
        "use_weighted_sampler": False,
        "sampler_pos_multiplier": 1.0,
    },
    "Task2_1_vs_23": {
        "use_attn_prior": True,
        "alpha_attn": 0.15,
        "attn_warmup_epochs": 2,
        "attn_prior_eps": 0.05,
        "attn_supervision_mode": "both",
        "use_neg_suppress": True,
        "beta_neg": 0.05,
        "neg_warmup_epochs": 2,
        "neg_margin": 0.15,
        "neg_supervision_mode": "both",
        "neg_topk_ratio": 0.20,
        "best_model_mode": "auc",
        "threshold_search_mode": "auc",
        "class1_weight": 1.0,
        "use_weighted_sampler": False,
        "sampler_pos_multiplier": 1.0,
    },
}

CONFIG = {
    "TRAIN_FEAT_DIR": "data/features_train",
    "TEST_FEAT_DIR": "data/features_test_noroi",
    "LOG_ROOT": "multimodal/logs_multimodal",
    "SAVE_DIR": "multimodal/test_results_multimodal",
    "FOLDS": 5,
    "EPOCHS": 25,
    "LR": 1e-4,
    "WEIGHT_DECAY": 1e-3,
    "SEED": 42,
    "MAX_PATCHES": 4000,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "NUM_WORKERS": 0,
    "PIN_MEMORY": torch.cuda.is_available(),
    "PERSISTENT_WORKERS": True,
    "PREFETCH_FACTOR": 4,
    "AMP": torch.cuda.is_available(),
    "ALLOW_TF32": True,
    "NON_BLOCKING": True,
    "COMPILE_MODEL": False,
    "DETERMINISTIC": False,
    "CUDNN_BENCHMARK": True,
    "BOOTSTRAP_N": 2000,
    "BOOTSTRAP_ALPHA": 0.95,
    "BOOTSTRAP_SEED": 42,
    "THRESHOLD_STEPS": 1001,
}


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean() if self.reduction == "mean" else loss


class DummyScaler:
    def scale(self, loss):
        return loss
    def step(self, optimizer):
        optimizer.step()
    def update(self):
        return None
    def unscale_(self, optimizer):
        return None


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = CONFIG["DETERMINISTIC"]
    torch.backends.cudnn.benchmark = CONFIG["CUDNN_BENCHMARK"] and (not CONFIG["DETERMINISTIC"])

    if torch.cuda.is_available() and CONFIG["ALLOW_TF32"]:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def to_device(x):
    return x.to(CONFIG["DEVICE"], non_blocking=CONFIG["NON_BLOCKING"])


def maybe_autocast():
    return torch.cuda.amp.autocast(
        enabled=CONFIG["AMP"] and str(CONFIG["DEVICE"]).startswith("cuda")
    )


def maybe_compile(model):
    if CONFIG["COMPILE_MODEL"] and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Warning: torch.compile failed, fallback to eager mode. {e}")
    return model


def get_task_label(original_label: int, task_name: str):
    if task_name == "Task1_0_vs_123":
        return 0 if original_label == 0 else 1
    if task_name == "Task2_1_vs_23":
        if original_label == 0:
            return None
        return 0 if original_label == 1 else 1
    raise ValueError(f"Unknown task: {task_name}")


def get_task_policy(task_name: str):
    return TASK_POLICIES[task_name]


def load_all_train_pt_files(train_feat_dir: str) -> List[str]:
    pt_files = glob.glob(os.path.join(train_feat_dir, "**", "*.pt"), recursive=True)
    if not pt_files:
        pt_files = glob.glob(os.path.join(train_feat_dir, "*.pt"))
    pt_files = sorted(pt_files)
    if len(pt_files) == 0:
        raise FileNotFoundError(f"No .pt files found under: {train_feat_dir}")
    return pt_files


def load_all_test_pts(test_feat_dir: str) -> List[str]:
    pt_files = glob.glob(os.path.join(test_feat_dir, "**", "*.pt"), recursive=True)
    if not pt_files:
        pt_files = glob.glob(os.path.join(test_feat_dir, "*.pt"))
    pt_files = sorted(pt_files)
    if len(pt_files) == 0:
        raise FileNotFoundError(f"No .pt files found under: {test_feat_dir}")
    return pt_files


def build_task_file_list(all_pt_files: List[str], task_name: str):
    task_files, task_labels = [], []
    for f in all_pt_files:
        d = torch.load(f, map_location="cpu")
        orig_label = int(d["label"])
        new_label = get_task_label(orig_label, task_name)
        if new_label is None:
            continue
        task_files.append(f)
        task_labels.append(new_label)
    return task_files, task_labels


def build_clin_stats_from_train_files(train_files: List[str]):
    clin_list = []
    for f in train_files:
        d = torch.load(f, map_location="cpu")
        clin = d.get("clin_feats", None)
        if clin is None:
            raise ValueError(f"{f} missing clin_feats")
        if isinstance(clin, torch.Tensor):
            clin = clin.detach().cpu().numpy()
        clin = np.asarray(clin, dtype=np.float32).reshape(-1)
        clin_list.append(clin)

    clin_arr = np.stack(clin_list, axis=0)
    clin_mean = clin_arr.mean(axis=0).astype(np.float32)
    clin_std = clin_arr.std(axis=0).astype(np.float32)
    clin_std = np.where(clin_std == 0, 1.0, clin_std).astype(np.float32)
    return clin_mean, clin_std


def normalize_clin_array(x: np.ndarray, mean_arr: np.ndarray, std_arr: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    return (x - mean_arr) / std_arr


def build_clin_data_dict_from_files(file_list: List[str], mean_arr: np.ndarray, std_arr: np.ndarray):
    clin_data_dict = {}
    for f in file_list:
        d = torch.load(f, map_location="cpu")
        clin = d.get("clin_feats", None)
        if clin is None:
            raise ValueError(f"{f} missing clin_feats")

        if isinstance(clin, torch.Tensor):
            clin = clin.detach().cpu().numpy()
        clin = np.asarray(clin, dtype=np.float32).reshape(-1)
        clin = normalize_clin_array(clin, mean_arr, std_arr)
        clin_data_dict[os.path.basename(f)] = clin
    return clin_data_dict


def build_train_loader(dataset, train_y_fold, policy):
    if policy["use_weighted_sampler"]:
        train_y_fold = np.asarray(train_y_fold).astype(int)
        n_neg = int((train_y_fold == 0).sum())
        n_pos = int((train_y_fold == 1).sum())
        pos_w = (n_neg / max(n_pos, 1)) * policy["sampler_pos_multiplier"]
        sample_weights = np.where(train_y_fold == 1, pos_w, 1.0).astype(np.float64)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        kwargs = {
            "dataset": dataset,
            "batch_size": 1,
            "shuffle": False,
            "sampler": sampler,
            "num_workers": CONFIG["NUM_WORKERS"],
            "pin_memory": CONFIG["PIN_MEMORY"],
        }
    else:
        kwargs = {
            "dataset": dataset,
            "batch_size": 1,
            "shuffle": True,
            "num_workers": CONFIG["NUM_WORKERS"],
            "pin_memory": CONFIG["PIN_MEMORY"],
        }

    if CONFIG["NUM_WORKERS"] > 0:
        kwargs["persistent_workers"] = CONFIG["PERSISTENT_WORKERS"]
        kwargs["prefetch_factor"] = CONFIG["PREFETCH_FACTOR"]
    return DataLoader(**kwargs)


def build_val_loader(dataset):
    kwargs = {
        "dataset": dataset,
        "batch_size": 1,
        "shuffle": False,
        "num_workers": CONFIG["NUM_WORKERS"],
        "pin_memory": CONFIG["PIN_MEMORY"],
    }
    if CONFIG["NUM_WORKERS"] > 0:
        kwargs["persistent_workers"] = CONFIG["PERSISTENT_WORKERS"]
        kwargs["prefetch_factor"] = CONFIG["PREFETCH_FACTOR"]
    return DataLoader(**kwargs)


def compute_binary_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        tn, fp, fn, tp = 0, 0, 0, 0
    else:
        tn, fp, fn, tp = cm.ravel()

    spec = float(tn / max(tn + fp, 1))
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = 0.5 * (sens + spec)

    return {
        "acc": float(acc),
        "sens": float(sens),
        "spec": float(spec),
        "prec": float(prec),
        "f1": float(f1),
        "bal_acc": float(bal_acc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def task_specific_score(metric_dict, mode, task_name):
    if mode == "acc":
        return metric_dict["acc"]
    if mode == "sens":
        return metric_dict["sens"]
    if mode == "spec":
        return metric_dict["spec"]
    if mode == "f1":
        return metric_dict["f1"]
    if mode == "balanced_acc":
        return metric_dict["bal_acc"]
    return metric_dict["acc"]


def find_best_threshold(y_true, y_probs, mode="acc", n_steps=1001, task_name=""):
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)

    best_th = 0.5
    best_score = -1.0
    best_metric_dict = None

    for th in np.linspace(0.0, 1.0, n_steps):
        m = compute_binary_metrics(y_true, y_probs, threshold=th)
        score = task_specific_score(m, mode, task_name)
        if score > best_score:
            best_score = score
            best_th = float(th)
            best_metric_dict = m

    return best_th, best_metric_dict


def should_apply_local_supervision(target, supervise_mode="positive_only"):
    if supervise_mode == "none":
        return False
    if supervise_mode == "both":
        return True
    if supervise_mode == "positive_only":
        return int(target.item()) == 1
    raise ValueError(f"Unknown supervise_mode: {supervise_mode}")


def build_attention_prior(patch_labels, eps=0.05):
    y = patch_labels.squeeze().float()
    if y.numel() == 0:
        return None
    pos_mask = (y > 0.5).float()
    n = pos_mask.numel()
    n_pos = float(pos_mask.sum().item())
    if n_pos <= 0:
        return None
    q = pos_mask / max(n_pos, 1.0)
    q = (1.0 - eps) * q + eps / max(n, 1)
    q = q / q.sum().clamp_min(1e-8)
    return q


def compute_attention_prior_loss(A, patch_labels, target, epoch, warmup_epochs=5, eps=0.05, supervise_mode="positive_only"):
    if A is None or epoch < warmup_epochs or not should_apply_local_supervision(target, supervise_mode):
        return None
    a = A.squeeze(0).float()
    if a.numel() == 0:
        return None
    q = build_attention_prior(patch_labels, eps=eps)
    if q is None:
        return None
    if a.numel() != q.numel():
        raise ValueError(f"Attention length mismatch: {a.numel()} vs {q.numel()}")
    a = a / a.sum().clamp_min(1e-8)
    return -(q * torch.log(a.clamp_min(1e-8))).sum()


def compute_negative_attention_suppress_loss(A, patch_labels, target, epoch, warmup_epochs=5, margin=0.15, supervise_mode="positive_only", topk_ratio=0.20):
    if A is None or epoch < warmup_epochs or not should_apply_local_supervision(target, supervise_mode):
        return None

    a = A.squeeze(0).float()
    y = patch_labels.squeeze().float()

    if a.numel() != y.numel():
        raise ValueError(f"Attention length mismatch: {a.numel()} vs {y.numel()}")

    neg_mask = (y <= 0.5)
    if neg_mask.sum() <= 0:
        return None

    neg_attn = a[neg_mask]
    if neg_attn.numel() == 0:
        return None

    k = max(1, int(round(float(neg_attn.numel()) * float(topk_ratio))))
    k = min(k, neg_attn.numel())
    topk_vals, _ = torch.topk(neg_attn, k=k, largest=True)
    neg_focus = topk_vals.mean()
    return F.relu(neg_focus - margin)


def train_one_epoch_multimodal(
    model,
    loader,
    optimizer,
    scaler,
    criterion_bag,
    current_task,
    use_attn_prior=False,
    alpha_attn=0.0,
    attn_warmup_epochs=0,
    attn_prior_eps=0.05,
    attn_supervision_mode="positive_only",
    use_neg_suppress=False,
    beta_neg=0.0,
    neg_warmup_epochs=0,
    neg_margin=0.15,
    neg_supervision_mode="positive_only",
    neg_topk_ratio=0.20,
    epoch_idx=1,
):
    model.train()
    total_loss = 0.0
    n_case = 0

    for batch in loader:
        img_feats = to_device(batch["img_feats"])
        clin_feats = to_device(batch["clin_feats"])
        labels = to_device(batch["label"])
        patch_labels = to_device(batch["patch_labels"]).float()

        target = torch.tensor(
            [get_task_label(int(labels.item()), current_task)],
            device=CONFIG["DEVICE"],
            dtype=torch.long,
        )

        if img_feats.size(1) > CONFIG["MAX_PATCHES"]:
            idx = torch.randperm(img_feats.size(1), device=img_feats.device)[:CONFIG["MAX_PATCHES"]]
            img_feats = img_feats[:, idx, :]
            patch_labels = patch_labels[:, idx]

        optimizer.zero_grad(set_to_none=True)

        with maybe_autocast():
            bag_logits, A, _ = model(img_feats, clin_feats)
            loss_total = criterion_bag(bag_logits, target)

            if use_attn_prior:
                loss_attn = compute_attention_prior_loss(
                    A=A,
                    patch_labels=patch_labels,
                    target=target,
                    epoch=epoch_idx,
                    warmup_epochs=attn_warmup_epochs,
                    eps=attn_prior_eps,
                    supervise_mode=attn_supervision_mode,
                )
                if loss_attn is not None:
                    loss_total = loss_total + alpha_attn * loss_attn

            if use_neg_suppress:
                loss_neg = compute_negative_attention_suppress_loss(
                    A=A,
                    patch_labels=patch_labels,
                    target=target,
                    epoch=epoch_idx,
                    warmup_epochs=neg_warmup_epochs,
                    margin=neg_margin,
                    supervise_mode=neg_supervision_mode,
                    topk_ratio=neg_topk_ratio,
                )
                if loss_neg is not None:
                    loss_total = loss_total + beta_neg * loss_neg

        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss_total.item())
        n_case += 1

    return total_loss / max(n_case, 1)


@torch.no_grad()
def evaluate_one_epoch_multimodal(
    model,
    loader,
    criterion_bag,
    current_task,
    use_attn_prior=False,
    alpha_attn=0.0,
    attn_warmup_epochs=0,
    attn_prior_eps=0.05,
    attn_supervision_mode="positive_only",
    use_neg_suppress=False,
    beta_neg=0.0,
    neg_warmup_epochs=0,
    neg_margin=0.15,
    neg_supervision_mode="positive_only",
    neg_topk_ratio=0.20,
    epoch_idx=1,
):
    model.eval()
    val_probs, val_true = [], []
    val_loss_total = 0.0
    val_count = 0

    for batch in loader:
        img_feats = to_device(batch["img_feats"])
        clin_feats = to_device(batch["clin_feats"])
        labels = to_device(batch["label"])
        patch_labels = to_device(batch["patch_labels"]).float()

        target = torch.tensor(
            [get_task_label(int(labels.item()), current_task)],
            device=CONFIG["DEVICE"],
            dtype=torch.long,
        )

        if img_feats.size(1) > CONFIG["MAX_PATCHES"]:
            img_feats = img_feats[:, :CONFIG["MAX_PATCHES"], :]
            patch_labels = patch_labels[:, :CONFIG["MAX_PATCHES"]]

        with maybe_autocast():
            bag_logits, A, _ = model(img_feats, clin_feats)
            probs = torch.softmax(bag_logits, dim=1)
            loss_total = criterion_bag(bag_logits, target)

            if use_attn_prior:
                loss_attn = compute_attention_prior_loss(
                    A=A,
                    patch_labels=patch_labels,
                    target=target,
                    epoch=epoch_idx,
                    warmup_epochs=attn_warmup_epochs,
                    eps=attn_prior_eps,
                    supervise_mode=attn_supervision_mode,
                )
                if loss_attn is not None:
                    loss_total = loss_total + alpha_attn * loss_attn

            if use_neg_suppress:
                loss_neg = compute_negative_attention_suppress_loss(
                    A=A,
                    patch_labels=patch_labels,
                    target=target,
                    epoch=epoch_idx,
                    warmup_epochs=neg_warmup_epochs,
                    margin=neg_margin,
                    supervise_mode=neg_supervision_mode,
                    topk_ratio=neg_topk_ratio,
                )
                if loss_neg is not None:
                    loss_total = loss_total + beta_neg * loss_neg

        val_probs.append(float(probs[0, 1].item()))
        val_true.append(get_task_label(int(labels.item()), current_task))
        val_loss_total += float(loss_total.item())
        val_count += 1

    avg_val_loss = val_loss_total / max(val_count, 1)
    try:
        val_auc = roc_auc_score(val_true, val_probs)
    except Exception:
        val_auc = 0.5

    best_threshold, metric_dict = find_best_threshold(
        val_true,
        val_probs,
        mode=get_task_policy(current_task)["threshold_search_mode"],
        n_steps=CONFIG["THRESHOLD_STEPS"],
        task_name=current_task,
    )

    return {
        "val_probs": val_probs,
        "val_true": val_true,
        "val_loss": float(avg_val_loss),
        "val_auc": float(val_auc),
        "best_threshold": float(best_threshold),
        "metric_dict": metric_dict,
    }


def bootstrap_metric_ci(metric_fn, y_true, y_other, n_boot=1000, seed=42, alpha=0.95):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_other = np.asarray(y_other)
    n = len(y_true)

    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            val = metric_fn(y_true[idx], y_other[idx])
            if val is None:
                continue
            val = float(val)
            if not np.isnan(val):
                vals.append(val)
        except Exception:
            continue

    if len(vals) == 0:
        return np.nan, np.nan

    vals = np.asarray(vals, dtype=np.float32)
    lo = np.quantile(vals, (1 - alpha) / 2)
    hi = np.quantile(vals, 1 - (1 - alpha) / 2)
    return float(lo), float(hi)


def compute_binary_metrics_with_ci(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= float(threshold)).astype(int)

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        tn, fp, fn, tp = 0, 0, 0, 0
    else:
        tn, fp, fn, tp = cm.ravel()

    spec = float(tn / max(tn + fp, 1))
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = (sens + spec) / 2.0

    auc_lo, auc_hi = (np.nan, np.nan)
    if len(np.unique(y_true)) > 1:
        auc_lo, auc_hi = bootstrap_metric_ci(
            lambda yt, yp: roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else np.nan,
            y_true, y_prob,
            n_boot=CONFIG["BOOTSTRAP_N"],
            seed=CONFIG["BOOTSTRAP_SEED"],
            alpha=CONFIG["BOOTSTRAP_ALPHA"],
        )

    acc_lo, acc_hi = bootstrap_metric_ci(
        lambda yt, yp: accuracy_score(yt, yp),
        y_true, y_pred,
        n_boot=CONFIG["BOOTSTRAP_N"],
        seed=CONFIG["BOOTSTRAP_SEED"],
        alpha=CONFIG["BOOTSTRAP_ALPHA"],
    )

    sens_lo, sens_hi = bootstrap_metric_ci(
        lambda yt, yp: recall_score(yt, yp, pos_label=1, zero_division=0),
        y_true, y_pred,
        n_boot=CONFIG["BOOTSTRAP_N"],
        seed=CONFIG["BOOTSTRAP_SEED"],
        alpha=CONFIG["BOOTSTRAP_ALPHA"],
    )

    spec_lo, spec_hi = bootstrap_metric_ci(
        lambda yt, yp: (
            confusion_matrix(yt, yp, labels=[0, 1]).ravel()[0] /
            max(confusion_matrix(yt, yp, labels=[0, 1]).ravel()[0] + confusion_matrix(yt, yp, labels=[0, 1]).ravel()[1], 1)
        ),
        y_true, y_pred,
        n_boot=CONFIG["BOOTSTRAP_N"],
        seed=CONFIG["BOOTSTRAP_SEED"],
        alpha=CONFIG["BOOTSTRAP_ALPHA"],
    )

    f1_lo, f1_hi = bootstrap_metric_ci(
        lambda yt, yp: f1_score(yt, yp, zero_division=0),
        y_true, y_pred,
        n_boot=CONFIG["BOOTSTRAP_N"],
        seed=CONFIG["BOOTSTRAP_SEED"],
        alpha=CONFIG["BOOTSTRAP_ALPHA"],
    )

    return {
        "AUC": float(auc) if not np.isnan(auc) else np.nan,
        "AUC_CI": "" if np.isnan(auc_lo) else f"{auc_lo:.3f}-{auc_hi:.3f}",
        "Acc": float(acc),
        "Acc_CI": f"{acc_lo:.3f}-{acc_hi:.3f}" if not np.isnan(acc_lo) else "",
        "Sens": float(sens),
        "Sens_CI": f"{sens_lo:.3f}-{sens_hi:.3f}" if not np.isnan(sens_lo) else "",
        "Spec": float(spec),
        "Spec_CI": f"{spec_lo:.3f}-{spec_hi:.3f}" if not np.isnan(spec_lo) else "",
        "Balanced_Acc": float(bal_acc),
        "Prec": float(prec),
        "F1": float(f1),
        "F1_CI": f"{f1_lo:.3f}-{f1_hi:.3f}" if not np.isnan(f1_lo) else "",
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "N": int(len(y_true)),
    }


def plot_save_roc_curve(y_true, y_probs, save_path, title, auc_score):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc_score:.3f}")
        plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"[WARN] failed to save ROC curve: {e}")


def save_roc_raw_data(y_true, y_probs, save_path):
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        pd.DataFrame({
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds,
        }).to_csv(save_path, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"[WARN] failed to save ROC raw data: {e}")


def compute_group_metrics(y_true, y_pred):
    labels = [0, 1, 2]
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    macro_prec, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_prec, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)

    per_class_recall = {}
    for i, lab in enumerate(labels):
        denom = cm[i, :].sum()
        per_class_recall[f"Recall_{lab}"] = float(cm[i, i] / denom) if denom > 0 else np.nan

    return {
        "Acc": float(acc),
        "Macro_Prec": float(macro_prec),
        "Macro_Recall": float(macro_recall),
        "Macro_F1": float(macro_f1),
        "Weighted_Prec": float(weighted_prec),
        "Weighted_Recall": float(weighted_recall),
        "Weighted_F1": float(weighted_f1),
        "Cohen_Kappa": float(kappa),
        **per_class_recall,
        "N": int(len(y_true)),
    }


def merge_task_predictions(df_task1: pd.DataFrame, df_task2: pd.DataFrame) -> pd.DataFrame:
    df = df_task1[["pt_path", "file_name", "orig_label", "mean_prob", "hard_pred"]].copy()
    df = df.rename(columns={
        "mean_prob": "task1_prob_123",
        "hard_pred": "task1_pred_123",
    })

    df2 = df_task2[["pt_path", "mean_prob", "hard_pred"]].copy()
    df2 = df2.rename(columns={
        "mean_prob": "task2_prob_23",
        "hard_pred": "task2_pred_23",
    })

    df = df.merge(df2, on="pt_path", how="left")

    final_group_true = []
    final_group_pred = []

    for _, row in df.iterrows():
        orig = int(row["orig_label"])
        if orig == 0:
            true_group = 0
        elif orig == 1:
            true_group = 1
        else:
            true_group = 2

        if int(row["task1_pred_123"]) == 0:
            pred_group = 0
        else:
            pred_group = 1 if int(row["task2_pred_23"]) == 0 else 2

        final_group_true.append(true_group)
        final_group_pred.append(pred_group)

    df["final_group_true"] = final_group_true
    df["final_group_pred"] = final_group_pred
    df["final_correct"] = (df["final_group_true"] == df["final_group_pred"]).astype(int)
    return df


def save_group_confusion_matrix(y_true, y_pred, save_path):
    labels = [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("3-Class Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, ["normal", "early", "middle+late"], rotation=20)
    plt.yticks(tick_marks, ["normal", "early", "middle+late"])

    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def load_summary_df(summary_csv: Optional[str]):
    if summary_csv is None or (not os.path.exists(summary_csv)):
        return None
    df = pd.read_csv(summary_csv)
    if "Task" not in df.columns:
        return None
    return df


def get_thresholds_from_summary(summary_df: Optional[pd.DataFrame], experiment_name: str, task_names: list):
    thresholds = {task: 0.5 for task in task_names}
    source = {task: "default_0.5" for task in task_names}

    if summary_df is None:
        return thresholds, source

    df = summary_df.copy()
    if "Experiment" in df.columns:
        df = df[df["Experiment"].astype(str) == str(experiment_name)]

    if len(df) == 0:
        return thresholds, source

    for task in task_names:
        sub = df[df["Task"].astype(str) == task]
        if len(sub) == 0:
            continue

        row = sub.iloc[0]
        if "Threshold" in row and pd.notna(row["Threshold"]):
            thresholds[task] = float(row["Threshold"])
            source[task] = "summary:Threshold"

    return thresholds, source


def format_mean_std(values: List[float], digits: int = 3):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return ""
    mean_v = float(np.mean(values))
    std_v = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return f"{mean_v:.{digits}f} +/- {std_v:.{digits}f}"