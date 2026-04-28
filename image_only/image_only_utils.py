import os
import sys
import glob
import json
import copy
import random
import warnings
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
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

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message=".*scipy._lib.messagestream.MessageStream size changed.*",
    category=RuntimeWarning,
)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)



# =========================================================
# Global config
# =========================================================
ALL_TASKS = [
    "Task1_0_vs_123",
    "Task2_1_vs_23",
]

TASK_TO_POSITIVE_NAME = {
    "Task1_0_vs_123": "123",
    "Task2_1_vs_23": "23",
}

TASK_POLICIES = {
    "Task1_0_vs_123": {
        "best_model_mode": "auc",
        "threshold_search_mode": "auc",
        "class1_weight": 1.0,
        "use_weighted_sampler": False,
        "sampler_pos_multiplier": 1.0,
    },
    "Task2_1_vs_23": {
        "best_model_mode": "auc",
        "threshold_search_mode": "auc",
        "class1_weight": 1.0,
        "use_weighted_sampler": False,
        "sampler_pos_multiplier": 1.0,
    },
}

CONFIG = {
    "ROOT_DIR": ROOT_DIR,
    "TRAIN_FEAT_DIR": os.path.join(ROOT_DIR, "data", "features_train"),
    "TEST_FEAT_DIR": os.path.join(ROOT_DIR, "data", "features_test_noroi"),

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

    "THRESHOLD_STEPS": 1001,

    "BOOTSTRAP_N": 1000,
    "BOOTSTRAP_ALPHA": 0.95,
    "BOOTSTRAP_SEED": 42,
}


# =========================================================
# Basic helpers
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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


def get_task_label(original_label: int, task_name: str):
    if task_name == "Task1_0_vs_123":
        return 0 if original_label == 0 else 1
    if task_name == "Task2_1_vs_23":
        if original_label == 0:
            return None
        return 0 if original_label == 1 else 1
    raise ValueError(f"Unknown task: {task_name}")


def get_task_policy(task_name: str):
    if task_name not in TASK_POLICIES:
        raise ValueError(f"Unknown task: {task_name}")
    return TASK_POLICIES[task_name]


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
    return torch.cuda.amp.autocast(enabled=CONFIG["AMP"] and str(CONFIG["DEVICE"]).startswith("cuda"))


def maybe_compile(model):
    if CONFIG["COMPILE_MODEL"] and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Warning: torch.compile failed, fallback to eager mode. {e}")
    return model


def format_mean_std(values: List[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return ""
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return f"{mean:.3f} +/- {std:.3f}"


# =========================================================
# Data helpers
# =========================================================
def load_all_train_pt_files(train_feat_dir: Optional[str] = None) -> List[str]:
    pt_root = train_feat_dir or CONFIG["TRAIN_FEAT_DIR"]
    pt_files = sorted(glob.glob(os.path.join(pt_root, "**", "*.pt"), recursive=True))
    if len(pt_files) == 0:
        pt_files = sorted(glob.glob(os.path.join(pt_root, "*.pt")))
    if len(pt_files) == 0:
        raise FileNotFoundError(f"No .pt files found under: {pt_root}")
    return pt_files


def load_all_test_pts(test_feat_dir: Optional[str] = None) -> List[str]:
    pt_root = test_feat_dir or CONFIG["TEST_FEAT_DIR"]
    pt_files = sorted(glob.glob(os.path.join(pt_root, "**", "*.pt"), recursive=True))
    if len(pt_files) == 0:
        pt_files = sorted(glob.glob(os.path.join(pt_root, "*.pt")))
    if len(pt_files) == 0:
        raise FileNotFoundError(f"No .pt files found under: {pt_root}")
    return pt_files


def get_original_label_from_pt(pt_path: str) -> int:
    data = torch.load(pt_path, map_location="cpu")
    return int(data["label"])


def build_task_file_list(pt_files: List[str], task_name: str) -> Tuple[List[str], List[int]]:
    out_files, out_labels = [], []
    for p in pt_files:
        orig = get_original_label_from_pt(p)
        t = get_task_label(orig, task_name)
        if t is None:
            continue
        out_files.append(p)
        out_labels.append(int(t))
    return out_files, out_labels


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


# =========================================================
# Metrics / threshold
# =========================================================
def compute_binary_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= float(threshold)).astype(int)

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


def task_specific_score(metric_dict, mode: str):
    if mode == "auc":
        return None
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


def find_best_threshold(y_true, y_probs, mode="auc", n_steps=1001, task_name=None):
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)

    if mode == "auc":
        # for classification threshold used later, choose balanced_acc if auc mode is requested
        mode = "balanced_acc"

    best_th = 0.5
    best_score = -1.0
    best_metric_dict = None

    for th in np.linspace(0.0, 1.0, n_steps):
        m = compute_binary_metrics(y_true, y_probs, threshold=th)
        score = task_specific_score(m, mode)
        if score > best_score:
            best_score = score
            best_th = float(th)
            best_metric_dict = m

    return best_th, best_metric_dict


def bootstrap_metric_ci(metric_fn, y_true, y_other, n_boot=2000, seed=42, alpha=0.95):
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
            y_true,
            y_prob,
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
            max(
                confusion_matrix(yt, yp, labels=[0, 1]).ravel()[0] +
                confusion_matrix(yt, yp, labels=[0, 1]).ravel()[1], 1
            )
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


# =========================================================
# Plot / ROC helpers
# =========================================================
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
        print(f"Failed to save ROC curve: {e}")


def save_roc_raw_data(y_true, y_probs, save_path):
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds,
    }).to_csv(save_path, index=False, encoding="utf-8-sig")


def plot_loss_curve(history_df, save_path, title="Loss Curve"):
    plt.figure(figsize=(7, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_metric_curve(history_df, save_path, title="Validation Metrics Curve"):
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["val_auc"], label="Val AUC", linewidth=2)
    plt.plot(history_df["epoch"], history_df["val_acc"], label="Val Acc", linewidth=2)
    plt.plot(history_df["epoch"], history_df["val_sens"], label="Val Sens", linewidth=2)
    plt.plot(history_df["epoch"], history_df["val_spec"], label="Val Spec", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================================================
# Train / eval helpers
# =========================================================
def compute_clam_instance_loss(extra_dict: Dict[str, torch.Tensor], target: torch.Tensor):
    """
    Optional auxiliary CLAM instance loss.
    We keep it mild and only use if top/bot logits are available.
    """
    if "top_logits" not in extra_dict or "bot_logits" not in extra_dict:
        return None

    top_logits = extra_dict["top_logits"]  # [k, 2]
    bot_logits = extra_dict["bot_logits"]  # [k, 2]

    pos_label = int(target.item())
    neg_label = 1 - pos_label

    top_target = torch.full(
        (top_logits.size(0),),
        fill_value=pos_label,
        dtype=torch.long,
        device=top_logits.device,
    )
    bot_target = torch.full(
        (bot_logits.size(0),),
        fill_value=neg_label,
        dtype=torch.long,
        device=bot_logits.device,
    )

    loss_top = F.cross_entropy(top_logits, top_target)
    loss_bot = F.cross_entropy(bot_logits, bot_target)
    return 0.5 * (loss_top + loss_bot)


def train_one_epoch(model, loader, optimizer, scaler, criterion_bag, current_task, use_clam_inst_loss=False, clam_inst_weight=0.1):
    model.train()
    train_loss_total = 0.0
    train_count = 0

    for batch in loader:
        img_feats = to_device(batch["img_feats"])
        l = to_device(batch["label"])

        target = torch.tensor(
            [get_task_label(int(l.item()), current_task)],
            device=CONFIG["DEVICE"],
            dtype=torch.long,
        )

        if img_feats is not None and img_feats.size(1) > CONFIG["MAX_PATCHES"]:
            idx = torch.randperm(img_feats.size(1), device=img_feats.device)[:CONFIG["MAX_PATCHES"]]
            img_feats = img_feats[:, idx, :]

        optimizer.zero_grad(set_to_none=True)

        with maybe_autocast():
            bag_logits, A, extra = model(img_feats)
            loss_total = criterion_bag(bag_logits, target)

            if use_clam_inst_loss:
                loss_inst = compute_clam_instance_loss(extra, target)
                if loss_inst is not None:
                    loss_total = loss_total + clam_inst_weight * loss_inst

        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss_total += float(loss_total.item())
        train_count += 1

    return train_loss_total / max(train_count, 1)


@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion_bag, current_task, use_clam_inst_loss=False, clam_inst_weight=0.1):
    model.eval()
    val_probs, val_true_epoch = [], []
    val_loss_total = 0.0
    val_count = 0

    for batch in loader:
        img_feats = to_device(batch["img_feats"])
        l = to_device(batch["label"])

        target = torch.tensor(
            [get_task_label(int(l.item()), current_task)],
            device=CONFIG["DEVICE"],
            dtype=torch.long,
        )

        if img_feats is not None and img_feats.size(1) > CONFIG["MAX_PATCHES"]:
            img_feats = img_feats[:, :CONFIG["MAX_PATCHES"], :]

        with maybe_autocast():
            bag_logits, A, extra = model(img_feats)
            probs = torch.softmax(bag_logits, dim=1)

            loss_total = criterion_bag(bag_logits, target)
            if use_clam_inst_loss:
                loss_inst = compute_clam_instance_loss(extra, target)
                if loss_inst is not None:
                    loss_total = loss_total + clam_inst_weight * loss_inst

        val_probs.append(float(probs[0, 1].item()))
        val_true_epoch.append(get_task_label(int(l.item()), current_task))
        val_loss_total += float(loss_total.item())
        val_count += 1

    avg_val_loss = val_loss_total / max(val_count, 1)
    try:
        auc_val = roc_auc_score(val_true_epoch, val_probs)
    except Exception:
        auc_val = 0.5

    epoch_best_th, epoch_metric_dict = find_best_threshold(
        val_true_epoch,
        val_probs,
        mode=get_task_policy(current_task)["threshold_search_mode"],
        n_steps=CONFIG["THRESHOLD_STEPS"],
        task_name=current_task,
    )

    return {
        "val_loss": avg_val_loss,
        "val_auc": auc_val,
        "val_probs": val_probs,
        "val_true": val_true_epoch,
        "best_threshold": epoch_best_th,
        "metric_dict": epoch_metric_dict,
    }


# =========================================================
# Test-time checkpoint helpers
# =========================================================
def load_summary_df(summary_csv: str) -> pd.DataFrame:
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"Summary csv not found: {summary_csv}")
    return pd.read_csv(summary_csv)


def get_thresholds_from_summary(summary_df: pd.DataFrame, experiment_name: str, task_names: List[str]):
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
        elif "OOF_Best_Thres" in row and pd.notna(row["OOF_Best_Thres"]):
            thresholds[task] = float(row["OOF_Best_Thres"])
            source[task] = "summary:OOF_Best_Thres"
        elif "Best_Thres" in row and pd.notna(row["Best_Thres"]):
            thresholds[task] = float(row["Best_Thres"])
            source[task] = "summary:Best_Thres"

    return thresholds, source


def load_task_checkpoints(log_root: str, experiment_name: str, task_name: str):
    # support both
    # logs_image_only/<Task>/<Experiment>/fold*_best.pth
    # logs_image_only/<Experiment>/<Task>/fold*_best.pth
    p1 = os.path.join(log_root, task_name, experiment_name)
    p2 = os.path.join(log_root, experiment_name, task_name)

    ckpts = []
    for d in [p1, p2]:
        if os.path.isdir(d):
            ckpts = sorted(glob.glob(os.path.join(d, "fold*_best.pth")))
            if len(ckpts) > 0:
                return ckpts

    raise FileNotFoundError(f"No fold*_best.pth found for {experiment_name} | {task_name} under {log_root}")


# =========================================================
# Grouped 3-class merge / metrics
# =========================================================
def merge_task_predictions(df_task1: pd.DataFrame, df_task2: pd.DataFrame):
    """
    Clinical-only fixed logic style:
    - Task1 applies to all samples
    - Task2 probabilities may exist for all samples
    - final grouping:
        if task1_pred == 0 -> group 0
        else -> group 1 if task2_pred == 0 else group 2
    """
    df1 = df_task1[["pt_path", "orig_label", "mean_prob", "hard_pred"]].copy()
    df1 = df1.rename(columns={
        "mean_prob": "task1_prob_123",
        "hard_pred": "task1_pred_123",
    })

    df2 = df_task2[["pt_path", "mean_prob", "hard_pred"]].copy()
    df2 = df2.rename(columns={
        "mean_prob": "task2_prob_23",
        "hard_pred": "task2_pred_23",
    })

    df = df1.merge(df2, on="pt_path", how="left")

    final_group_pred = []
    final_group_true = []

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
            t2_pred = row["task2_pred_23"]
            if pd.isna(t2_pred):
                pred_group = 1  # fallback to mild positive
            else:
                pred_group = 1 if int(t2_pred) == 0 else 2

        final_group_true.append(true_group)
        final_group_pred.append(pred_group)

    df["final_group_true"] = final_group_true
    df["final_group_pred"] = final_group_pred
    df["final_correct"] = (df["final_group_true"] == df["final_group_pred"]).astype(int)
    return df


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
        "Recall_0": per_class_recall["Recall_0"],
        "Recall_1": per_class_recall["Recall_1"],
        "Recall_2": per_class_recall["Recall_2"],
        "Confusion_Matrix": json.dumps(cm.tolist(), ensure_ascii=False),
        "N": int(len(y_true)),
    }


def save_group_confusion_matrix(y_true, y_pred, save_path):
    labels = [0, 1, 2]
    names = ["normal", "early", "middle+late"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Grouped Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()