import os
import sys
import glob
import copy
import json
import random
import warnings
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    recall_score,
    confusion_matrix,
    precision_score,
    f1_score,
)
from xgboost import XGBClassifier

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

ROOT_DIR = find_project_root(CUR_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

print(f'[INFO] CUR_DIR  : {CUR_DIR}')
print(f'[INFO] ROOT_DIR : {ROOT_DIR}')

ALL_TASKS = [
    'Task1_0_vs_123',
    'Task2_1_vs_23',
]

TASK_POLICIES = {
    'Task1_0_vs_123': {
        'threshold_search_mode': 'auc',
    },
    'Task2_1_vs_23': {
        'threshold_search_mode': 'auc',
    },
}

CLINICAL_ONLY_EXPERIMENTS = {
    'LogisticRegression': {
        'builder': lambda seed: Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=1.0,
                max_iter=5000,
                class_weight='balanced',
                solver='liblinear',
                random_state=seed,
            )),
        ]),
    },
    'RandomForest': {
        'builder': lambda seed: Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced_subsample',
                n_jobs=-1,
                random_state=seed,
            )),
        ]),
    },
    'XGBoost': {
        'builder': lambda seed: Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.0,
                reg_lambda=1.0,
                min_child_weight=1,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=seed,
                n_jobs=4,
            )),
        ]),
    },
}

CONFIG = {
    'LOG_ROOT': os.path.join(CUR_DIR, 'logs_clinical_only'),
    'TRAIN_FEAT_DIR': os.path.join(ROOT_DIR, 'data', 'features_train'),
    'FOLDS': 5,
    'SEED': 42,
    'THRESHOLD_STEPS': 1001,
}


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


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
        try:
            data = torch.load(path, map_location='cpu')
            clin_feats = data.get('clin_feats', None)
            label = data.get('label', None)
            if clin_feats is None or label is None:
                raise KeyError('missing clin_feats or label')
            if not isinstance(clin_feats, torch.Tensor):
                clin_feats = torch.tensor(clin_feats, dtype=torch.float32)
            clin_arr = clin_feats.detach().cpu().numpy().astype(np.float32).reshape(-1)
            records.append({
                'path': path,
                'file_name': os.path.basename(path),
                'orig_label': int(label),
                'clin_feats': clin_arr,
            })
        except Exception as e:
            print(f'[WARN] skip bad pt: {path} | {e}')
    if not records:
        raise RuntimeError('No usable pt records loaded.')
    return records


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
        'acc': float(acc),
        'sens': float(sens),
        'spec': float(spec),
        'prec': float(prec),
        'f1': float(f1),
        'bal_acc': float(bal_acc),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }


def task_specific_score(metric_dict, mode):
    if mode == 'acc':
        return metric_dict['acc']
    if mode == 'sens':
        return metric_dict['sens']
    if mode == 'spec':
        return metric_dict['spec']
    if mode == 'f1':
        return metric_dict['f1']
    if mode == 'balanced_acc':
        return metric_dict['bal_acc']
    return metric_dict['acc']


def find_best_threshold(y_true, y_probs, mode='acc', n_steps=1001):
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)
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


def plot_save_roc_curve(y_true, y_probs, save_path, title, auc_score):
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


def save_roc_raw_data(y_true, y_probs, save_path):
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        'threshold': thresholds,
    }).to_csv(save_path, index=False, encoding='utf-8-sig')


def format_mean_std(values: List[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return ''
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return f'{mean:.3f}±{std:.3f}'


def run_all():
    seed_everything(CONFIG['SEED'])
    os.makedirs(CONFIG['LOG_ROOT'], exist_ok=True)

    all_records = load_pt_records(CONFIG['TRAIN_FEAT_DIR'])
    final_summary_rows = []

    print('========== Runtime Config (Clinical-Only) ==========')
    print(json.dumps(CONFIG, ensure_ascii=False, indent=2))
    print('===================================================')

    for current_task in ALL_TASKS:
        policy = TASK_POLICIES[current_task]
        task_records = []
        task_labels = []
        for rec in all_records:
            new_label = get_task_label(rec['orig_label'], current_task)
            if new_label is None:
                continue
            task_records.append(rec)
            task_labels.append(new_label)

        if not task_records:
            raise RuntimeError(f'No usable records for task: {current_task}')

        X = np.stack([r['clin_feats'] for r in task_records], axis=0).astype(np.float32)
        y = np.asarray(task_labels).astype(int)
        file_names = np.asarray([r['file_name'] for r in task_records])

        print(f'\n{"=" * 50}\nTASK: {current_task}\n{"=" * 50}')
        print(f'usable samples = {len(X)} | class-0 = {(y == 0).sum()} | class-1 = {(y == 1).sum()}')

        for exp_name, exp_cfg in CLINICAL_ONLY_EXPERIMENTS.items():
            print(f'\n--- Experiment: {exp_name} ---')
            exp_save_dir = os.path.join(CONFIG['LOG_ROOT'], current_task, exp_name)
            os.makedirs(exp_save_dir, exist_ok=True)

            skf = StratifiedKFold(n_splits=CONFIG['FOLDS'], shuffle=True, random_state=CONFIG['SEED'])
            fold_rows = []
            oof_true, oof_prob, oof_name = [], [], []
            fold_thresholds = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                file_val = file_names[val_idx]

                model = exp_cfg['builder'](CONFIG['SEED'] + fold)
                model.fit(X_train, y_train)
                val_prob = model.predict_proba(X_val)[:, 1]
                val_auc = roc_auc_score(y_val, val_prob) if len(np.unique(y_val)) > 1 else 0.5
                best_th, best_metric_dict = find_best_threshold(
                    y_val,
                    val_prob,
                    mode=policy['threshold_search_mode'],
                    n_steps=CONFIG['THRESHOLD_STEPS'],
                )

                model_path = os.path.join(exp_save_dir, f'fold{fold}_best.pkl')
                joblib.dump({
                    'model': model,
                    'task_name': current_task,
                    'experiment_name': exp_name,
                    'fold': fold,
                    'best_threshold': float(best_th),
                    'feature_dim': int(X.shape[1]),
                    'seed': int(CONFIG['SEED'] + fold),
                }, model_path)

                fold_pred_df = pd.DataFrame({
                    'file_name': file_val,
                    'y_true': y_val,
                    'y_prob': val_prob,
                })
                fold_pred_df.to_csv(
                    os.path.join(exp_save_dir, f'fold{fold}_validation_predictions.csv'),
                    index=False,
                    encoding='utf-8-sig',
                )
                save_roc_raw_data(
                    y_val,
                    val_prob,
                    os.path.join(exp_save_dir, f'fold{fold}_validation_roc_raw.csv'),
                )
                plot_save_roc_curve(
                    y_val,
                    val_prob,
                    os.path.join(exp_save_dir, f'fold{fold}_validation_roc.png'),
                    title=f'{current_task} | {exp_name} | Fold {fold}',
                    auc_score=val_auc,
                )

                fold_rows.append({
                    'fold': fold,
                    'auc': float(val_auc),
                    'acc': float(best_metric_dict['acc']),
                    'sens': float(best_metric_dict['sens']),
                    'spec': float(best_metric_dict['spec']),
                    'prec': float(best_metric_dict['prec']),
                    'f1': float(best_metric_dict['f1']),
                    'bal_acc': float(best_metric_dict['bal_acc']),
                    'threshold': float(best_th),
                    'n_val': int(len(y_val)),
                })

                oof_true.extend(y_val.tolist())
                oof_prob.extend(val_prob.tolist())
                oof_name.extend(file_val.tolist())
                fold_thresholds.append(float(best_th))

                print(
                    f'[{current_task}][{exp_name}][Fold {fold}] '
                    f'AUC={val_auc:.4f} | Acc={best_metric_dict["acc"]:.4f} | '
                    f'Sens={best_metric_dict["sens"]:.4f} | Spec={best_metric_dict["spec"]:.4f} | '
                    f'F1={best_metric_dict["f1"]:.4f} | Th={best_th:.3f}'
                )

            fold_df = pd.DataFrame(fold_rows)
            fold_df.to_csv(os.path.join(exp_save_dir, 'fold_metrics.csv'), index=False, encoding='utf-8-sig')

            oof_pred_df = pd.DataFrame({
                'file_name': oof_name,
                'y_true': oof_true,
                'y_prob': oof_prob,
            })
            oof_pred_df.to_csv(os.path.join(exp_save_dir, 'oof_predictions.csv'), index=False, encoding='utf-8-sig')

            oof_auc = roc_auc_score(oof_true, oof_prob) if len(set(oof_true)) > 1 else 0.5
            oof_best_th, oof_metric_dict = find_best_threshold(
                oof_true,
                oof_prob,
                mode=policy['threshold_search_mode'],
                n_steps=CONFIG['THRESHOLD_STEPS'],
            )
            save_roc_raw_data(oof_true, oof_prob, os.path.join(exp_save_dir, 'oof_roc_raw.csv'))
            plot_save_roc_curve(
                oof_true,
                oof_prob,
                os.path.join(exp_save_dir, 'oof_roc.png'),
                title=f'OOF ROC | {current_task} | {exp_name}',
                auc_score=oof_auc,
            )

            final_summary_rows.append({
                'Task': current_task,
                'Experiment': exp_name,
                'CV_AUC_Mean': float(fold_df['auc'].mean()),
                'CV_AUC_Std': float(fold_df['auc'].std(ddof=1)),
                'CV_AUC_MeanStd': format_mean_std(fold_df['auc'].tolist()),
                'CV_Acc_Mean': float(fold_df['acc'].mean()),
                'CV_Acc_Std': float(fold_df['acc'].std(ddof=1)),
                'CV_Acc_MeanStd': format_mean_std(fold_df['acc'].tolist()),
                'CV_Sens_Mean': float(fold_df['sens'].mean()),
                'CV_Sens_Std': float(fold_df['sens'].std(ddof=1)),
                'CV_Sens_MeanStd': format_mean_std(fold_df['sens'].tolist()),
                'CV_Spec_Mean': float(fold_df['spec'].mean()),
                'CV_Spec_Std': float(fold_df['spec'].std(ddof=1)),
                'CV_Spec_MeanStd': format_mean_std(fold_df['spec'].tolist()),
                'CV_F1_Mean': float(fold_df['f1'].mean()),
                'CV_F1_Std': float(fold_df['f1'].std(ddof=1)),
                'CV_F1_MeanStd': format_mean_std(fold_df['f1'].tolist()),
                'Threshold': float(oof_best_th),
                'FoldThresholdMean': float(np.mean(fold_thresholds)) if fold_thresholds else 0.5,
                'NumSamples': int(len(oof_true)),
                'OOF_AUC': float(oof_auc),
                'OOF_Acc': float(oof_metric_dict['acc']),
                'OOF_Sens': float(oof_metric_dict['sens']),
                'OOF_Spec': float(oof_metric_dict['spec']),
                'OOF_F1': float(oof_metric_dict['f1']),
            })

    summary_df = pd.DataFrame(final_summary_rows)
    summary_csv = os.path.join(CONFIG['LOG_ROOT'], 'FINAL_CLINICAL_ONLY_SUMMARY.csv')
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    print(f'\n✅ Final summary saved to: {summary_csv}')
    print(summary_df[['Task', 'Experiment', 'CV_AUC_MeanStd', 'CV_Acc_MeanStd', 'CV_Sens_MeanStd', 'CV_Spec_MeanStd', 'CV_F1_MeanStd', 'Threshold']])


if __name__ == '__main__':
    run_all()
