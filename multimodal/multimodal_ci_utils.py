import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    confusion_matrix,
    precision_score,
    f1_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
)

from multimodal.multimodal_utils import CONFIG


def bootstrap_metric_ci(metric_fn, y_true, y_other, n_boot=None, seed=None, alpha=None):
    if n_boot is None:
        n_boot = CONFIG["BOOTSTRAP_N"]
    if seed is None:
        seed = CONFIG["BOOTSTRAP_SEED"]
    if alpha is None:
        alpha = CONFIG["BOOTSTRAP_ALPHA"]

    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_other = np.asarray(y_other)
    n = len(y_true)

    vals = []
    for _ in range(int(n_boot)):
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


def format_ci(lo, hi, digits=3):
    if np.isnan(lo) or np.isnan(hi):
        return ""
    return f"{lo:.{digits}f}-{hi:.{digits}f}"


def format_metric_with_ci(value, lo, hi, digits=3):
    if value is None or np.isnan(value):
        return ""
    if np.isnan(lo) or np.isnan(hi):
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"


def _binary_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return np.nan
    tn, fp, fn, tp = cm.ravel()
    return float(tn / max(tn + fp, 1))


def compute_binary_metrics_with_ci_v2(y_true, y_prob, threshold):
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
        )

    acc_lo, acc_hi = bootstrap_metric_ci(
        lambda yt, yp: accuracy_score(yt, yp),
        y_true, y_pred,
    )

    sens_lo, sens_hi = bootstrap_metric_ci(
        lambda yt, yp: recall_score(yt, yp, pos_label=1, zero_division=0),
        y_true, y_pred,
    )

    spec_lo, spec_hi = bootstrap_metric_ci(
        _binary_specificity,
        y_true, y_pred,
    )

    f1_lo, f1_hi = bootstrap_metric_ci(
        lambda yt, yp: f1_score(yt, yp, zero_division=0),
        y_true, y_pred,
    )

    return {
        "AUC": float(auc) if not np.isnan(auc) else np.nan,
        "AUC_CI": format_ci(auc_lo, auc_hi),
        "AUC_with_95CI": format_metric_with_ci(float(auc) if not np.isnan(auc) else np.nan, auc_lo, auc_hi),

        "Acc": float(acc),
        "Acc_CI": format_ci(acc_lo, acc_hi),
        "Acc_with_95CI": format_metric_with_ci(float(acc), acc_lo, acc_hi),

        "Sens": float(sens),
        "Sens_CI": format_ci(sens_lo, sens_hi),
        "Sens_with_95CI": format_metric_with_ci(float(sens), sens_lo, sens_hi),

        "Spec": float(spec),
        "Spec_CI": format_ci(spec_lo, spec_hi),
        "Spec_with_95CI": format_metric_with_ci(float(spec), spec_lo, spec_hi),

        "Balanced_Acc": float(bal_acc),
        "Prec": float(prec),

        "F1": float(f1),
        "F1_CI": format_ci(f1_lo, f1_hi),
        "F1_with_95CI": format_metric_with_ci(float(f1), f1_lo, f1_hi),

        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "N": int(len(y_true)),
    }


def compute_group_metrics_plain(y_true, y_pred):
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


def _threeclass_recall_for_label(y_true, y_pred, cls):
    labels = [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    denom = cm[cls, :].sum()
    if denom <= 0:
        return np.nan
    return float(cm[cls, cls] / denom)


def compute_group_metrics_with_ci(y_true, y_pred):
    labels = [0, 1, 2]
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    base = compute_group_metrics_plain(y_true, y_pred)

    acc = base["Acc"]
    macro_f1 = base["Macro_F1"]
    weighted_f1 = base["Weighted_F1"]
    kappa = base["Cohen_Kappa"]

    acc_lo, acc_hi = bootstrap_metric_ci(
        lambda yt, yp: accuracy_score(yt, yp),
        y_true, y_pred,
    )

    macro_f1_lo, macro_f1_hi = bootstrap_metric_ci(
        lambda yt, yp: precision_recall_fscore_support(
            yt, yp, labels=labels, average="macro", zero_division=0
        )[2],
        y_true, y_pred,
    )

    weighted_f1_lo, weighted_f1_hi = bootstrap_metric_ci(
        lambda yt, yp: precision_recall_fscore_support(
            yt, yp, labels=labels, average="weighted", zero_division=0
        )[2],
        y_true, y_pred,
    )

    kappa_lo, kappa_hi = bootstrap_metric_ci(
        lambda yt, yp: cohen_kappa_score(yt, yp, labels=labels),
        y_true, y_pred,
    )

    recall_ci = {}
    recall_with_ci = {}
    for cls in labels:
        lo, hi = bootstrap_metric_ci(
            lambda yt, yp, _cls=cls: _threeclass_recall_for_label(yt, yp, _cls),
            y_true, y_pred,
        )
        key = f"Recall_{cls}"
        recall_ci[f"{key}_CI"] = format_ci(lo, hi)
        recall_with_ci[f"{key}_with_95CI"] = format_metric_with_ci(base[key], lo, hi)

    return {
        **base,

        "Acc_CI": format_ci(acc_lo, acc_hi),
        "Acc_with_95CI": format_metric_with_ci(acc, acc_lo, acc_hi),

        "Macro_F1_CI": format_ci(macro_f1_lo, macro_f1_hi),
        "Macro_F1_with_95CI": format_metric_with_ci(macro_f1, macro_f1_lo, macro_f1_hi),

        "Weighted_F1_CI": format_ci(weighted_f1_lo, weighted_f1_hi),
        "Weighted_F1_with_95CI": format_metric_with_ci(weighted_f1, weighted_f1_lo, weighted_f1_hi),

        "Cohen_Kappa_CI": format_ci(kappa_lo, kappa_hi),
        "Cohen_Kappa_with_95CI": format_metric_with_ci(kappa, kappa_lo, kappa_hi),

        **recall_ci,
        **recall_with_ci,
    }