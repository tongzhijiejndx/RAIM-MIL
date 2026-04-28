import os
import sys
import glob
import json
import warnings
from typing import Dict, List, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    confusion_matrix,
    precision_score,
    f1_score,
    roc_curve,
    precision_recall_fscore_support,
    cohen_kappa_score,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings(
    'ignore',
    message='.*scipy._lib.messagestream.MessageStream size changed.*',
    category=RuntimeWarning,
)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def find_project_root(start_dir: str) -> str:
    candidates = []
    cur = os.path.abspath(start_dir)
    while True:
        candidates.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    for cand in candidates:
        has_train = os.path.isdir(os.path.join(cand, 'data', 'features_train'))
        has_test = os.path.isdir(os.path.join(cand, 'data', 'features_test_noroi'))
        if has_train or has_test:
            return cand
    return os.path.abspath(os.path.join(start_dir, '..', '..'))


def find_log_root(cur_dir: str, root_dir: str) -> str:
    candidates = [
        os.path.join(cur_dir, 'logs_clinical_only'),
        os.path.join(root_dir, 'clinic_only', 'logs_clinical_only'),
        os.path.join(root_dir, 'data统计', 'clinical-only', 'logs_clinical_only'),
        os.path.join(root_dir, 'logs_clinical_only'),
    ]
    for cand in candidates:
        if os.path.isdir(cand) or os.path.exists(os.path.join(cand, 'FINAL_CLINICAL_ONLY_SUMMARY.csv')):
            return cand
    return candidates[0]


ROOT_DIR = find_project_root(CUR_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

print(f'[INFO] CUR_DIR  : {CUR_DIR}')
print(f'[INFO] ROOT_DIR : {ROOT_DIR}')

LOG_ROOT_DEFAULT = find_log_root(CUR_DIR, ROOT_DIR)
SAVE_DIR_DEFAULT = os.path.join(CUR_DIR, 'test_results_clinical_only')

CONFIG = {
    'TEST_FEAT_DIR': os.path.join(ROOT_DIR, 'data', 'features_test_noroi'),
    'LOG_ROOT': LOG_ROOT_DEFAULT,
    'SAVE_DIR': SAVE_DIR_DEFAULT,
    'TASKS': ['Task1_0_vs_123', 'Task2_1_vs_23'],
    'BOOTSTRAP_N': 2000,
    'BOOTSTRAP_ALPHA': 0.95,
    'BOOTSTRAP_SEED': 42,
    'THRESHOLD_MODE': 'from_summary_csv',
    'SUMMARY_CSV': os.path.join(LOG_ROOT_DEFAULT, 'FINAL_CLINICAL_ONLY_SUMMARY.csv'),
}

TASK_TO_POSITIVE_NAME = {
    'Task1_0_vs_123': '123',
    'Task2_1_vs_23': '23',
}


# ---------- helpers ----------
def get_task_label(original_label: int, task_name: str):
    if task_name == 'Task1_0_vs_123':
        return 0 if original_label == 0 else 1
    if task_name == 'Task2_1_vs_23':
        if original_label == 0:
            return None
        return 0 if original_label == 1 else 1
    raise ValueError(f'Unknown task: {task_name}')


def load_pt_records(pt_root: str) -> List[Dict[str, Any]]:
    pt_files = glob.glob(os.path.join(pt_root, '**', '*.pt'), recursive=True)
    if not pt_files:
        pt_files = glob.glob(os.path.join(pt_root, '*.pt'))
    if not pt_files:
        raise FileNotFoundError(f'No .pt files found under: {pt_root}')

    import torch

    records = []
    for path in sorted(pt_files):
        data = torch.load(path, map_location='cpu')
        clin_feats = data.get('clin_feats', None)
        label = data.get('label', None)
        if clin_feats is None or label is None:
            raise KeyError(f'{path} missing clin_feats or label')
        if not isinstance(clin_feats, torch.Tensor):
            clin_feats = torch.tensor(clin_feats, dtype=torch.float32)
        clin_arr = clin_feats.detach().cpu().numpy().astype(np.float32).reshape(-1)
        records.append({
            'path': path,
            'file_name': os.path.basename(path),
            'orig_label': int(label),
            'clin_feats': clin_arr,
        })
    return records


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
            y_true, y_prob,
            n_boot=CONFIG['BOOTSTRAP_N'], seed=CONFIG['BOOTSTRAP_SEED'], alpha=CONFIG['BOOTSTRAP_ALPHA']
        )

    acc_lo, acc_hi = bootstrap_metric_ci(
        lambda yt, yp: accuracy_score(yt, yp),
        y_true, y_pred,
        n_boot=CONFIG['BOOTSTRAP_N'], seed=CONFIG['BOOTSTRAP_SEED'], alpha=CONFIG['BOOTSTRAP_ALPHA']
    )
    sens_lo, sens_hi = bootstrap_metric_ci(
        lambda yt, yp: recall_score(yt, yp, pos_label=1, zero_division=0),
        y_true, y_pred,
        n_boot=CONFIG['BOOTSTRAP_N'], seed=CONFIG['BOOTSTRAP_SEED'], alpha=CONFIG['BOOTSTRAP_ALPHA']
    )
    spec_lo, spec_hi = bootstrap_metric_ci(
        lambda yt, yp: (
            confusion_matrix(yt, yp, labels=[0, 1]).ravel()[0] /
            max(confusion_matrix(yt, yp, labels=[0, 1]).ravel()[0] + confusion_matrix(yt, yp, labels=[0, 1]).ravel()[1], 1)
        ),
        y_true, y_pred,
        n_boot=CONFIG['BOOTSTRAP_N'], seed=CONFIG['BOOTSTRAP_SEED'], alpha=CONFIG['BOOTSTRAP_ALPHA']
    )
    f1_lo, f1_hi = bootstrap_metric_ci(
        lambda yt, yp: f1_score(yt, yp, zero_division=0),
        y_true, y_pred,
        n_boot=CONFIG['BOOTSTRAP_N'], seed=CONFIG['BOOTSTRAP_SEED'], alpha=CONFIG['BOOTSTRAP_ALPHA']
    )

    return {
        'AUC': float(auc) if not np.isnan(auc) else np.nan,
        'AUC_CI': '' if np.isnan(auc_lo) else f'{auc_lo:.3f}-{auc_hi:.3f}',
        'Acc': float(acc),
        'Acc_CI': f'{acc_lo:.3f}-{acc_hi:.3f}' if not np.isnan(acc_lo) else '',
        'Sens': float(sens),
        'Sens_CI': f'{sens_lo:.3f}-{sens_hi:.3f}' if not np.isnan(sens_lo) else '',
        'Spec': float(spec),
        'Spec_CI': f'{spec_lo:.3f}-{spec_hi:.3f}' if not np.isnan(spec_lo) else '',
        'Balanced_Acc': float(bal_acc),
        'Prec': float(prec),
        'F1': float(f1),
        'F1_CI': f'{f1_lo:.3f}-{f1_hi:.3f}' if not np.isnan(f1_lo) else '',
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'N': int(len(y_true)),
    }


def plot_save_roc_curve(y_true, y_probs, save_path, title, auc_score):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, lw=2, label=f'AUC = {auc_score:.3f}')
        plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f'Failed to save ROC curve: {e}')


def save_roc_raw_data(y_true, y_probs, save_path):
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': thresholds}).to_csv(save_path, index=False, encoding='utf-8-sig')


def load_threshold_map(summary_csv: str) -> Dict[tuple, float]:
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f'Summary csv not found: {summary_csv}')
    df = pd.read_csv(summary_csv)
    required_cols = {'Task', 'Experiment', 'Threshold'}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f'Summary csv missing columns: {sorted(missing)}')
    out = {}
    for _, row in df.iterrows():
        out[(str(row['Task']), str(row['Experiment']))] = float(row['Threshold'])
    return out


def load_fold_models(log_root: str, task_name: str, exp_name: str) -> List[Dict[str, Any]]:
    exp_dir = os.path.join(log_root, task_name, exp_name)
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f'Experiment dir not found: {exp_dir}')
    ckpts = sorted(glob.glob(os.path.join(exp_dir, 'fold*_best.pkl')))
    if len(ckpts) == 0:
        raise FileNotFoundError(f'No fold*_best.pkl found in: {exp_dir}')
    return [joblib.load(p) for p in ckpts]


def ensemble_predict_binary(fold_models: List[Dict[str, Any]], records: List[Dict[str, Any]]) -> np.ndarray:
    X = np.stack([r['clin_feats'] for r in records], axis=0).astype(np.float32)
    probs = []
    for bundle in fold_models:
        model = bundle['model']
        p = model.predict_proba(X)[:, 1]
        probs.append(p)
    return np.mean(np.vstack(probs), axis=0)


def infer_binary_task(log_root: str, exp_name: str, task_name: str, records: List[Dict[str, Any]], threshold: float, save_dir: str) -> Dict[str, Any]:
    os.makedirs(save_dir, exist_ok=True)
    fold_models = load_fold_models(log_root, task_name, exp_name)

    # inference on all records
    probs = ensemble_predict_binary(fold_models, records)

    y_true_mapped = []
    for r in records:
        mapped = get_task_label(r['orig_label'], task_name)
        if mapped is None:
            y_true_mapped.append(-1)
        else:
            y_true_mapped.append(int(mapped))
    y_true_mapped = np.asarray(y_true_mapped, dtype=int)
    y_pred = (probs >= float(threshold)).astype(int)

    valid_mask = y_true_mapped >= 0
    valid_true = y_true_mapped[valid_mask]
    valid_prob = probs[valid_mask]

    metrics = compute_binary_metrics_with_ci(valid_true, valid_prob, threshold)

    pred_df = pd.DataFrame({
        'file_name': [r['file_name'] for r in records],
        'orig_label': [int(r['orig_label']) for r in records],
        'task_label': y_true_mapped,
        'prob': probs,
        'pred': y_pred,
        'is_valid_for_metric': valid_mask.astype(int),
    })
    pred_df.to_csv(os.path.join(save_dir, 'test_predictions.csv'), index=False, encoding='utf-8-sig')

    if len(np.unique(valid_true)) > 1:
        save_roc_raw_data(valid_true, valid_prob, os.path.join(save_dir, 'test_roc_raw.csv'))
        plot_save_roc_curve(valid_true, valid_prob, os.path.join(save_dir, 'test_roc.png'), f'Test ROC | {task_name} | {exp_name}', metrics['AUC'])

    with open(os.path.join(save_dir, 'test_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pred_map = {}
    for _, row in pred_df.iterrows():
        pred_map[str(row['file_name'])] = {
            'true_label': int(row['orig_label']),
            'task_label': int(row['task_label']),
            'prob': float(row['prob']),
            'pred': int(row['pred']),
            'is_valid_for_metric': int(row['is_valid_for_metric']),
        }

    return {'metrics': metrics, 'pred_df': pred_df, 'pred_map': pred_map}


def merge_three_class_predictions(task1_map: Dict[str, Any], task2_map: Dict[str, Any], save_dir: str) -> Dict[str, Any]:
    rows = []
    for fname, t1_item in task1_map.items():
        true_label_raw = int(t1_item['true_label'])
        task1_prob = float(t1_item['prob'])
        task1_pred = int(t1_item['pred'])

        if task1_pred == 0:
            final_pred_3class = 0
            task2_prob = np.nan
            task2_pred = np.nan
        else:
            if fname not in task2_map:
                raise KeyError(f'{fname} predicted positive in Task1 but missing from Task2 predictions')
            t2_item = task2_map[fname]
            task2_prob = float(t2_item['prob'])
            task2_pred = int(t2_item['pred'])
            final_pred_3class = 1 if task2_pred == 0 else 2

        if true_label_raw == 0:
            true_label_3class = 0
        elif true_label_raw == 1:
            true_label_3class = 1
        else:
            true_label_3class = 2

        rows.append({
            'file_name': fname,
            'true_label_raw': true_label_raw,
            'true_label_3class': true_label_3class,
            'task1_prob': task1_prob,
            'task1_pred': task1_pred,
            'task2_prob': task2_prob,
            'task2_pred': task2_pred,
            'final_pred_3class': final_pred_3class,
        })

    df = pd.DataFrame(rows).sort_values('file_name').reset_index(drop=True)
    df.to_csv(os.path.join(save_dir, 'test_predictions_3class.csv'), index=False, encoding='utf-8-sig')

    y_true = df['true_label_3class'].to_numpy().astype(int)
    y_pred = df['final_pred_3class'].to_numpy().astype(int)

    acc = accuracy_score(y_true, y_pred)

    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    m_prec, m_rec, m_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    cls_prec, cls_rec, cls_f1, cls_sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )

    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    metrics_3class = pd.DataFrame([{
        'Acc': float(acc),
        'Macro_Prec': float(m_prec),
        'Macro_Rec': float(m_rec),
        'Macro_F1': float(m_f1),
        'Weighted_Prec': float(w_prec),
        'Weighted_Rec': float(w_rec),
        'Weighted_F1': float(w_f1),
        'Kappa': float(kappa),
        'Recall_F0': float(cls_rec[0]),
        'Recall_F1': float(cls_rec[1]),
        'Recall_F2_F3': float(cls_rec[2]),
        'Prec_F0': float(cls_prec[0]),
        'Prec_F1': float(cls_prec[1]),
        'Prec_F2_F3': float(cls_prec[2]),
        'F1_F0': float(cls_f1[0]),
        'F1_F1': float(cls_f1[1]),
        'F1_F2_F3': float(cls_f1[2]),
        'Support_F0': int(cls_sup[0]),
        'Support_F1': int(cls_sup[1]),
        'Support_F2_F3': int(cls_sup[2]),
        'N': int(len(df)),
        'CM_00': int(cm[0, 0]), 'CM_01': int(cm[0, 1]), 'CM_02': int(cm[0, 2]),
        'CM_10': int(cm[1, 0]), 'CM_11': int(cm[1, 1]), 'CM_12': int(cm[1, 2]),
        'CM_20': int(cm[2, 0]), 'CM_21': int(cm[2, 1]), 'CM_22': int(cm[2, 2]),
    }])
    metrics_3class.to_csv(os.path.join(save_dir, 'test_metrics_3class.csv'), index=False, encoding='utf-8-sig')

    return {
        'Acc_3class': float(acc),
        'Macro_F1_3class': float(m_f1),
        'Weighted_F1_3class': float(w_f1),
        'Kappa_3class': float(kappa),
        'Recall_F0': float(cls_rec[0]),
        'Recall_F1': float(cls_rec[1]),
        'Recall_F2_F3': float(cls_rec[2]),
        'N_3class': int(len(df)),
    }


def evaluate_one_experiment(exp_name: str, test_records: List[Dict[str, Any]], threshold_map: Dict[tuple, float]) -> Dict[str, Any]:
    print('\n' + '=' * 80)
    print(f'[EXPERIMENT] {exp_name}')
    print('=' * 80)

    exp_save_dir = os.path.join(CONFIG['SAVE_DIR'], exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)

    task_results = {}
    for task_name in CONFIG['TASKS']:
        key = (task_name, exp_name)
        threshold = float(threshold_map.get(key, 0.5)) if CONFIG['THRESHOLD_MODE'] == 'from_summary_csv' else 0.5
        task_save_dir = os.path.join(exp_save_dir, task_name)

        result = infer_binary_task(
            log_root=CONFIG['LOG_ROOT'],
            exp_name=exp_name,
            task_name=task_name,
            records=test_records,
            threshold=threshold,
            save_dir=task_save_dir,
        )
        task_results[task_name] = {'threshold': threshold, **result}
        m = result['metrics']
        print(f'[{exp_name}][{task_name}] AUC={m["AUC"]:.4f} | Acc={m["Acc"]:.4f} | Sens={m["Sens"]:.4f} | Spec={m["Spec"]:.4f}')

    merged = merge_three_class_predictions(
        task_results['Task1_0_vs_123']['pred_map'],
        task_results['Task2_1_vs_23']['pred_map'],
        exp_save_dir,
    )

    return {
        'Experiment': exp_name,
        'Task1_AUC': task_results['Task1_0_vs_123']['metrics']['AUC'],
        'Task1_AUC_CI': task_results['Task1_0_vs_123']['metrics']['AUC_CI'],
        'Task1_Acc': task_results['Task1_0_vs_123']['metrics']['Acc'],
        'Task1_Acc_CI': task_results['Task1_0_vs_123']['metrics']['Acc_CI'],
        'Task1_Sens': task_results['Task1_0_vs_123']['metrics']['Sens'],
        'Task1_Sens_CI': task_results['Task1_0_vs_123']['metrics']['Sens_CI'],
        'Task1_Spec': task_results['Task1_0_vs_123']['metrics']['Spec'],
        'Task1_Spec_CI': task_results['Task1_0_vs_123']['metrics']['Spec_CI'],
        'Task1_F1': task_results['Task1_0_vs_123']['metrics']['F1'],
        'Task1_F1_CI': task_results['Task1_0_vs_123']['metrics']['F1_CI'],
        'Task1_Threshold': task_results['Task1_0_vs_123']['threshold'],
        'Task2_AUC': task_results['Task2_1_vs_23']['metrics']['AUC'],
        'Task2_AUC_CI': task_results['Task2_1_vs_23']['metrics']['AUC_CI'],
        'Task2_Acc': task_results['Task2_1_vs_23']['metrics']['Acc'],
        'Task2_Acc_CI': task_results['Task2_1_vs_23']['metrics']['Acc_CI'],
        'Task2_Sens': task_results['Task2_1_vs_23']['metrics']['Sens'],
        'Task2_Sens_CI': task_results['Task2_1_vs_23']['metrics']['Sens_CI'],
        'Task2_Spec': task_results['Task2_1_vs_23']['metrics']['Spec'],
        'Task2_Spec_CI': task_results['Task2_1_vs_23']['metrics']['Spec_CI'],
        'Task2_F1': task_results['Task2_1_vs_23']['metrics']['F1'],
        'Task2_F1_CI': task_results['Task2_1_vs_23']['metrics']['F1_CI'],
        'Task2_Threshold': task_results['Task2_1_vs_23']['threshold'],
        **merged,
    }


def run_all_tests():
    print('========== Runtime Config (Clinical-Only Test) ==========')
    print(json.dumps(CONFIG, ensure_ascii=False, indent=2))
    print('=========================================================')

    os.makedirs(CONFIG['SAVE_DIR'], exist_ok=True)
    test_records = load_pt_records(CONFIG['TEST_FEAT_DIR'])

    if not os.path.exists(CONFIG['LOG_ROOT']):
        raise FileNotFoundError(f'LOG_ROOT not found: {CONFIG["LOG_ROOT"]}')

    threshold_map = load_threshold_map(CONFIG['SUMMARY_CSV']) if CONFIG['THRESHOLD_MODE'] == 'from_summary_csv' else {}

    experiments = []
    for task_name in CONFIG['TASKS']:
        task_dir = os.path.join(CONFIG['LOG_ROOT'], task_name)
        if not os.path.isdir(task_dir):
            raise FileNotFoundError(f'Task directory not found: {task_dir}')
        exp_names = [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]
        experiments.extend(exp_names)
    experiments = sorted(set(experiments))

    summary_rows = []
    for exp_name in experiments:
        summary_row = evaluate_one_experiment(exp_name, test_records, threshold_map)
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(CONFIG['SAVE_DIR'], 'FINAL_TEST_CLINICAL_ONLY_SUMMARY.csv')
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    print(f'\n✅ Final test summary saved to: {summary_csv}')
    print(summary_df)


if __name__ == '__main__':
    run_all_tests()