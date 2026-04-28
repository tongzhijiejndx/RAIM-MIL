import os
import sys
import glob
import json
import traceback
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from multimodal.multimodal_models import MultimodalMILModel
from image_only.image_only_models import build_image_only_model

CONFIG = {
    "TRAIN_PT_ROOT": os.path.join(ROOT_DIR, "data", "features_test_roi"),
    "LOG_ROOT": os.path.join(CUR_DIR, "logs_multimodal"),
    "SAVE_DIR": os.path.join(CUR_DIR, "attention_roi_consistency_multimodal"),
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "NON_BLOCKING": True,
    "MAX_PATCHES": 4000,
    "TASKS": ["Task1_0_vs_123", "Task2_1_vs_23"],
    "EXPERIMENTS": [
        {"name": "Ours", "enabled": True},
        {"name": "LateFusion", "enabled": True},
    ],
    "TOPK_RATIO": 0.10,
    "REQUIRE_POSITIVE_PATCH": True,
}


def to_device(x: torch.Tensor) -> torch.Tensor:
    return x.to(CONFIG["DEVICE"], non_blocking=CONFIG["NON_BLOCKING"])

def load_pt_files(pt_root: str) -> List[str]:
    pt_files = sorted(glob.glob(os.path.join(pt_root, "**", "*.pt"), recursive=True))
    if len(pt_files) == 0:
        pt_files = sorted(glob.glob(os.path.join(pt_root, "*.pt")))
    if len(pt_files) == 0:
        raise FileNotFoundError(f"No .pt files found under: {pt_root}")
    return pt_files

def load_ckpt_files(log_root: str, exp_name: str, task_name: str) -> List[str]:
    ckpt_dir = os.path.join(log_root, task_name, exp_name)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "fold*_best.pth")))
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No fold checkpoints found in: {ckpt_dir}")
    return ckpts

def build_ours_model_from_ckpt(ckpt: Dict[str, Any]):
    model = MultimodalMILModel(**ckpt["model_kwargs"]).to(CONFIG["DEVICE"])
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model

def build_late_image_model_from_ckpt(ckpt: Dict[str, Any]):
    model = build_image_only_model(
        model_name=ckpt["image_model_name"],
        **ckpt["image_model_kwargs"],
    ).to(CONFIG["DEVICE"])
    model.load_state_dict(ckpt["image_model_state_dict"], strict=True)
    model.eval()
    return model

def normalize_clin_array(x: np.ndarray, mean_arr: np.ndarray, std_arr: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    std_arr = np.where(np.asarray(std_arr, dtype=np.float32) == 0, 1.0, np.asarray(std_arr, dtype=np.float32))
    return (x - np.asarray(mean_arr, dtype=np.float32)) / std_arr

@torch.no_grad()
def forward_one_pt_ours(pt_path: str, ckpt_path: str):
    data = torch.load(pt_path, map_location="cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_ours_model_from_ckpt(ckpt)

    img_feats = data["img_feats"]
    if not isinstance(img_feats, torch.Tensor):
        img_feats = torch.tensor(img_feats, dtype=torch.float32)
    else:
        img_feats = img_feats.float()

    if img_feats.ndim == 2:
        img_feats = img_feats.unsqueeze(0)
    if img_feats.size(1) > CONFIG["MAX_PATCHES"]:
        img_feats = img_feats[:, :CONFIG["MAX_PATCHES"], :]

    clin_feats = data.get("clin_feats", None)
    if clin_feats is None:
        raise ValueError(f"{pt_path} missing clin_feats")
    if isinstance(clin_feats, torch.Tensor):
        clin_feats = clin_feats.detach().cpu().numpy()
    clin_feats = np.asarray(clin_feats, dtype=np.float32).reshape(-1)
    clin_feats = normalize_clin_array(clin_feats, ckpt["clin_mean"], ckpt["clin_std"])
    clin_feats = torch.tensor(clin_feats, dtype=torch.float32).unsqueeze(0)

    patch_labels = data.get("patch_labels", None)
    if patch_labels is None:
        raise KeyError(f"{pt_path} missing patch_labels")
    if not isinstance(patch_labels, torch.Tensor):
        patch_labels = torch.tensor(patch_labels, dtype=torch.float32)
    else:
        patch_labels = patch_labels.float().view(-1)
    if len(patch_labels) > CONFIG["MAX_PATCHES"]:
        patch_labels = patch_labels[:CONFIG["MAX_PATCHES"]]

    img_feats = to_device(img_feats)
    clin_feats = to_device(clin_feats)

    bag_logits, A, extras = model(img_feats, clin_feats)
    probs = torch.softmax(bag_logits, dim=1)[0].detach().cpu().numpy()

    attn = None
    if A is not None:
        attn = A.squeeze(0).detach().float().cpu().numpy()

    return {
        "probs": probs,
        "attn": attn,
        "patch_labels": patch_labels.detach().cpu().numpy().astype(np.float32),
        "patient_id": data.get("patient_id", os.path.splitext(os.path.basename(pt_path))[0]),
        "pt_path": pt_path,
    }

@torch.no_grad()
def forward_one_pt_late(pt_path: str, ckpt_path: str):
    data = torch.load(pt_path, map_location="cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_late_image_model_from_ckpt(ckpt)

    img_feats = data["img_feats"]
    if not isinstance(img_feats, torch.Tensor):
        img_feats = torch.tensor(img_feats, dtype=torch.float32)
    else:
        img_feats = img_feats.float()

    if img_feats.ndim == 2:
        img_feats = img_feats.unsqueeze(0)
    if img_feats.size(1) > CONFIG["MAX_PATCHES"]:
        img_feats = img_feats[:, :CONFIG["MAX_PATCHES"], :]

    patch_labels = data.get("patch_labels", None)
    if patch_labels is None:
        raise KeyError(f"{pt_path} missing patch_labels")
    if not isinstance(patch_labels, torch.Tensor):
        patch_labels = torch.tensor(patch_labels, dtype=torch.float32)
    else:
        patch_labels = patch_labels.float().view(-1)
    if len(patch_labels) > CONFIG["MAX_PATCHES"]:
        patch_labels = patch_labels[:CONFIG["MAX_PATCHES"]]

    img_feats = to_device(img_feats)

    bag_logits, A, extras = model(img_feats)
    probs = torch.softmax(bag_logits, dim=1)[0].detach().cpu().numpy()

    attn = None
    if A is not None:
        attn = A.squeeze(0).detach().float().cpu().numpy()

    return {
        "probs": probs,
        "attn": attn,
        "patch_labels": patch_labels.detach().cpu().numpy().astype(np.float32),
        "patient_id": data.get("patient_id", os.path.splitext(os.path.basename(pt_path))[0]),
        "pt_path": pt_path,
    }

@torch.no_grad()
def forward_one_pt_ensemble(pt_path: str, ckpt_paths: List[str], exp_name: str):
    if exp_name == "Ours":
        outs = [forward_one_pt_ours(pt_path, ck) for ck in ckpt_paths]
    elif exp_name == "LateFusion":
        outs = [forward_one_pt_late(pt_path, ck) for ck in ckpt_paths]
    else:
        raise ValueError(f"Unsupported experiment: {exp_name}")

    probs = np.mean(np.stack([o["probs"] for o in outs], axis=0), axis=0)

    attn_list = [o["attn"] for o in outs if o["attn"] is not None]
    attn = None
    if len(attn_list) > 0:
        lengths = [len(a) for a in attn_list]
        if len(set(lengths)) == 1:
            attn = np.mean(np.stack(attn_list, axis=0), axis=0)

    base = outs[0]
    return {
        "patient_id": base["patient_id"],
        "pt_path": base["pt_path"],
        "probs": probs,
        "attn": attn,
        "patch_labels": base["patch_labels"],
    }

def compute_attn_roi_metrics(attn: Optional[np.ndarray], patch_labels: np.ndarray, topk_ratio: float = 0.10):
    if attn is None:
        return None

    attn = np.asarray(attn, dtype=np.float32).reshape(-1)
    y = np.asarray(patch_labels, dtype=np.float32).reshape(-1)

    if len(attn) != len(y):
        raise ValueError(f"attention length mismatch: {len(attn)} vs {len(y)}")
    if len(attn) == 0:
        return None
    if float(attn.sum()) <= 0:
        return None

    attn = attn / max(float(attn.sum()), 1e-8)
    pos_mask = (y > 0.5)

    if CONFIG["REQUIRE_POSITIVE_PATCH"] and int(pos_mask.sum()) <= 0:
        return None

    attn_mass_in_roi = float(attn[pos_mask].sum()) if pos_mask.sum() > 0 else 0.0
    peak_idx = int(np.argmax(attn))
    peak_in_roi = int(y[peak_idx] > 0.5) if len(y) > 0 else 0

    k = max(1, int(round(len(attn) * float(topk_ratio))))
    k = min(k, len(attn))
    topk_idx = np.argsort(attn)[-k:]
    topk_hit = float(np.mean((y[topk_idx] > 0.5).astype(np.float32))) if len(topk_idx) > 0 else 0.0

    return {
        "AttnMassInROI": attn_mass_in_roi,
        "PeakInROI": peak_in_roi,
        "TopKHit": topk_hit,
        "NumPatches": int(len(attn)),
        "NumROIPatches": int(pos_mask.sum()),
    }

def summarize_case_metrics(case_rows: List[dict]) -> dict:
    if len(case_rows) == 0:
        return {
            "N_cases": 0,
            "AttnMassInROI_mean": np.nan,
            "AttnMassInROI_std": np.nan,
            "PeakInROI_rate": np.nan,
            "TopKHit_mean": np.nan,
            "Status": "No valid cases",
        }

    df = pd.DataFrame(case_rows)
    return {
        "N_cases": int(len(df)),
        "AttnMassInROI_mean": float(df["AttnMassInROI"].mean()),
        "AttnMassInROI_std": float(df["AttnMassInROI"].std(ddof=0)),
        "PeakInROI_rate": float(df["PeakInROI"].mean()),
        "TopKHit_mean": float(df["TopKHit"].mean()),
        "Status": "OK",
    }

def run_one_experiment(exp_name: str):
    print("=" * 100)
    print(f"[ROI-CONSISTENCY] {exp_name}")
    print("=" * 100)

    pt_files = load_pt_files(CONFIG["TRAIN_PT_ROOT"])
    save_dir = os.path.join(CONFIG["SAVE_DIR"], exp_name)
    os.makedirs(save_dir, exist_ok=True)

    all_summary_rows = []
    all_case_rows = []

    for task_name in CONFIG["TASKS"]:
        try:
            ckpt_paths = load_ckpt_files(CONFIG["LOG_ROOT"], exp_name, task_name)
        except Exception as e:
            print(f"[WARN] skip {exp_name} / {task_name}: {e}")
            all_summary_rows.append({
                "Experiment": exp_name,
                "Task": task_name,
                "N_cases": 0,
                "AttnMassInROI_mean": np.nan,
                "AttnMassInROI_std": np.nan,
                "PeakInROI_rate": np.nan,
                "TopKHit_mean": np.nan,
                "Status": f"Checkpoint missing: {e}",
            })
            continue

        case_rows = []
        no_attn_count = 0

        for idx, pt_path in enumerate(pt_files, start=1):
            if idx % 100 == 0:
                print(f"[{task_name}] {idx}/{len(pt_files)}")

            try:
                out = forward_one_pt_ensemble(pt_path, ckpt_paths, exp_name)
                metric = compute_attn_roi_metrics(
                    attn=out["attn"],
                    patch_labels=out["patch_labels"],
                    topk_ratio=CONFIG["TOPK_RATIO"],
                )

                if metric is None:
                    if out["attn"] is None:
                        no_attn_count += 1
                    continue

                row = {
                    "Experiment": exp_name,
                    "Task": task_name,
                    "patient_id": out["patient_id"],
                    "pt_path": out["pt_path"],
                    "prob_class0": float(out["probs"][0]),
                    "prob_class1": float(out["probs"][1]),
                    **metric,
                }
                case_rows.append(row)

            except Exception as e:
                print(f"[WARN] case failed: {pt_path} | {e}")

        case_df = pd.DataFrame(case_rows)
        case_df.to_csv(os.path.join(save_dir, f"{task_name}_roi_case_metrics.csv"), index=False, encoding="utf-8-sig")

        summary = summarize_case_metrics(case_rows)
        if no_attn_count > 0 and len(case_rows) == 0:
            summary["Status"] = "No attention output"

        all_summary_rows.append({"Experiment": exp_name, "Task": task_name, **summary})
        all_case_rows.extend(case_rows)

    pd.DataFrame(all_summary_rows).to_csv(
        os.path.join(save_dir, "roi_consistency_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(all_case_rows).to_csv(
        os.path.join(save_dir, "roi_consistency_all_cases.csv"),
        index=False,
        encoding="utf-8-sig",
    )

def merge_all_experiments():
    rows = []
    for item in CONFIG["EXPERIMENTS"]:
        if not item["enabled"]:
            continue
        csv_path = os.path.join(CONFIG["SAVE_DIR"], item["name"], "roi_consistency_summary.csv")
        if os.path.exists(csv_path):
            rows.append(pd.read_csv(csv_path))

    if len(rows) == 0:
        print("[WARN] no roi consistency summaries found.")
        return

    out_df = pd.concat(rows, axis=0, ignore_index=True)
    out_csv = os.path.join(CONFIG["SAVE_DIR"], "all_multimodal_roi_consistency_summary.csv")
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] merged ROI consistency summary -> {out_csv}")

def main():
    os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)
    print(json.dumps(CONFIG, ensure_ascii=False, indent=2))

    for item in CONFIG["EXPERIMENTS"]:
        if not item["enabled"]:
            continue
        try:
            run_one_experiment(item["name"])
        except Exception as e:
            print(f"[ERROR] {item['name']} failed: {e}")
            traceback.print_exc()

    merge_all_experiments()
    print("=" * 100)
    print("Multimodal attention-ROI consistency analysis completed.")
    print("=" * 100)

if __name__ == "__main__":
    main()
