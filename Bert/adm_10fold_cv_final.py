#!/usr/bin/env python3
"""
Stratified 10-fold cross-validation for the IoT Shepherd ADM (BERT).

All outputs are written under one run directory. The outer test fold is used
only for final fold evaluation. A validation split is carved from the outer
training portion for epoch-level checkpoint selection.

Example full run:
    python adm_10fold_cv.py \
        --data final_dataset.csv \
        --output-dir ADM_10fold_CV_results

One-fold smoke test:
    python adm_10fold_cv.py \
        --data final_dataset.csv \
        --output-dir ADM_10fold_CV_smoke \
        --max-folds 1
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from scipy.stats import t as student_t
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


# -----------------------------
# Defaults matching the 10 fold ADM training setup
# -----------------------------
DEFAULT_MODEL = "bert-base-uncased"
DEFAULT_SEED = 42
DEFAULT_N_SPLITS = 10
DEFAULT_VAL_FRACTION = 0.10
DEFAULT_MAX_LENGTH = 512
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_WARMUP_STEPS = 500
DEFAULT_WEIGHT_DECAY = 0.01

EXCLUDED_FEATURE_COLUMNS = [
    "ip.src_host",
    "ip.dst_host",
    "arp.src.proto_ipv4",
    "tcp.payload",
    "http.file_data",
]


@dataclass
class RunConfig:
    data: str
    output_dir: str
    model_name: str
    seed: int
    n_splits: int
    val_fraction: float
    max_length: int
    epochs: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    max_folds: int | None
    keep_checkpoints: bool


class EncodedSubset(Dataset):
    """View into pre-tokenized examples without duplicating encodings."""

    def __init__(
        self,
        encodings: Dict[str, List[List[int]]],
        labels: np.ndarray,
        indices: Sequence[int],
    ) -> None:
        self.encodings = encodings
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, object]:
        idx = int(self.indices[item])
        out = {key: values[idx] for key, values in self.encodings.items()}
        out["labels"] = int(self.labels[idx])
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stratified 10-fold CV for IoT Shepherd ADM")
    p.add_argument("--data", default="final_dataset.csv")
    p.add_argument("--output-dir", default="ADM_10fold_CV_results")
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--folds", type=int, default=DEFAULT_N_SPLITS)
    p.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    p.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Run only the first N outer folds (useful for a smoke test).",
    )
    p.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep best-model checkpoints for each fold. By default they are deleted after metrics/predictions are saved.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed fold outputs in an existing output directory.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output directory before starting.",
    )
    return p.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def gpu_name() -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "CUDA device"


def prepare_output_dir(out_dir: Path, overwrite: bool, resume: bool) -> None:
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    elif out_dir.exists() and any(out_dir.iterdir()) and not resume:
        raise RuntimeError(
            f"Output directory already exists and is not empty: {out_dir}\n"
            "Use --resume to continue it or --overwrite to replace it."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "folds").mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)


def load_and_prepare_data(data_path: Path) -> tuple[pd.DataFrame, List[str], np.ndarray, pd.DataFrame]:
    df = pd.read_csv(data_path)

    if "Attack_label" not in df.columns and "Label" not in df.columns:
        raise ValueError("Dataset must contain Attack_label or Label.")
    if "Attack_type" not in df.columns:
        raise ValueError("Dataset must contain Attack_type for class naming.")

    if "Attack_label" in df.columns and "Label" not in df.columns:
        df = df.rename(columns={"Attack_label": "Label"})

    # Match the deployed ADM feature exclusions.
    df = df.drop(columns=[c for c in EXCLUDED_FEATURE_COLUMNS if c in df.columns])
    df["Label"] = df["Label"].astype(int)

    # Validate label-to-attack-type mapping.
    mapping_raw = df[["Label", "Attack_type"]].drop_duplicates()
    conflicts = mapping_raw.groupby("Label")["Attack_type"].nunique()
    if (conflicts > 1).any():
        bad = conflicts[conflicts > 1].index.tolist()
        raise ValueError(f"Some labels map to multiple Attack_type values: {bad}")

    original_labels = sorted(df["Label"].unique().tolist())
    original_to_model = {orig: i for i, orig in enumerate(original_labels)}
    df["model_label"] = df["Label"].map(original_to_model).astype(int)

    label_name_raw = (
        mapping_raw.sort_values("Label").drop_duplicates("Label").set_index("Label")["Attack_type"]
    )
    class_mapping_rows = []
    class_names = []
    for orig in original_labels:
        model_idx = original_to_model[orig]
        name = str(label_name_raw.loc[orig])
        class_names.append(name)
        class_mapping_rows.append(
            {"model_label": model_idx, "original_label": int(orig), "class_name": name}
        )
    class_mapping = pd.DataFrame(class_mapping_rows)

    excluded_from_text = {"Label", "Attack_type", "model_label", "text"}
    feature_cols = [c for c in df.columns if c not in excluded_from_text]
    if not feature_cols:
        raise ValueError("No feature columns remain after preprocessing.")

    print(f"Building textual representations from {len(feature_cols)} features...")
    values = df[feature_cols].itertuples(index=False, name=None)
    texts = [
        " ".join(f"{col}: {value}" for col, value in zip(feature_cols, row))
        for row in values
    ]

    y = df["model_label"].to_numpy(dtype=np.int64)
    return df, texts, y, class_mapping


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_weighted": p_w,
        "recall_weighted": r_w,
        "f1_weighted": f_w,
        "precision_macro": p_m,
        "recall_macro": r_m,
        "f1_macro": f_m,
    }


def trainer_compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric_dict(labels, preds)


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    fold: int,
) -> pd.DataFrame:
    labels = np.arange(len(class_names))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return pd.DataFrame(
        {
            "fold": fold,
            "model_label": labels,
            "class_name": list(class_names),
            "precision": p,
            "recall": r,
            "f1": f,
            "support": s,
        }
    )


def per_class_auc(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: Sequence[str],
    fold: int,
) -> pd.DataFrame:
    rows = []
    for c, name in enumerate(class_names):
        y_bin = (y_true == c).astype(int)
        score = float("nan")
        if np.unique(y_bin).size == 2:
            score = roc_auc_score(y_bin, probs[:, c])
        rows.append(
            {
                "fold": fold,
                "model_label": c,
                "class_name": name,
                "auc_ovr": score,
            }
        )
    return pd.DataFrame(rows)


def summarize_columns(df: pd.DataFrame, metric_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for col in metric_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        sem = std / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        tcrit = student_t.ppf(0.975, df=len(vals) - 1) if len(vals) > 1 else 0.0
        margin = tcrit * sem if len(vals) > 1 else 0.0
        rows.append(
            {
                "metric": col,
                "n_folds": len(vals),
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci95_low": mean - margin,
                "ci95_high": mean + margin,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        )
    return pd.DataFrame(rows)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "class"


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    out_base: Path,
    normalized: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("10-fold out-of-fold confusion matrix" + (" (row-normalized)" if normalized else ""))

    if len(class_names) <= 20:
        threshold = np.nanmax(matrix) / 2.0 if matrix.size else 0.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                text = f"{val:.2f}" if normalized else f"{int(val)}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if val > threshold else "black",
                )
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fold_metrics(fold_df: pd.DataFrame, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = fold_df["fold"].to_numpy()
    for metric in ["accuracy", "f1_weighted", "f1_macro"]:
        if metric in fold_df:
            ax.plot(x, fold_df[metric], marker="o", label=metric.replace("_", " "))
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_ylim(0.0, 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(summary: pd.DataFrame, out_base: Path) -> None:
    pivot = summary[summary["metric"] == "f1"].copy()
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(pivot))
    ax.bar(x, pivot["mean"], yerr=pivot["std"], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot["class_name"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("F1-score")
    ax.set_ylim(0.0, 1.01)
    ax.set_title("Per-class F1 across 10 folds (mean ± SD)")
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_oof_roc(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: Sequence[str],
    out_base: Path,
    curve_csv: Path,
) -> Dict[str, float]:
    fig, ax = plt.subplots(figsize=(9, 7))
    rows = []
    fprs = {}
    tprs = {}
    aucs = {}

    for c, name in enumerate(class_names):
        y_bin = (y_true == c).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, probs[:, c])
        score = auc(fpr, tpr)
        fprs[c] = fpr
        tprs[c] = tpr
        aucs[name] = float(score)
        ax.plot(fpr, tpr, linewidth=1.2, label=f"{name} ({score:.3f})")
        rows.extend(
            {
                "curve": "class",
                "model_label": c,
                "class_name": name,
                "fpr": float(x),
                "tpr": float(y),
            }
            for x, y in zip(fpr, tpr)
        )

    all_fpr = np.unique(np.concatenate([fprs[c] for c in fprs]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in fprs:
        mean_tpr += np.interp(all_fpr, fprs[c], tprs[c])
    mean_tpr /= len(class_names)
    macro_auc_curve = auc(all_fpr, mean_tpr)
    ax.plot(
        all_fpr,
        mean_tpr,
        linestyle="--",
        linewidth=2.0,
        label=f"Macro-average ({macro_auc_curve:.3f})",
    )
    rows.extend(
        {
            "curve": "macro",
            "model_label": -1,
            "class_name": "Macro-average",
            "fpr": float(x),
            "tpr": float(y),
        }
        for x, y in zip(all_fpr, mean_tpr)
    )

    ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("10-fold out-of-fold ROC curves")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).to_csv(curve_csv, index=False)
    aucs["Macro-average-curve"] = float(macro_auc_curve)
    return aucs


def summarize_per_class(per_class_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (label, name), grp in per_class_all.groupby(["model_label", "class_name"], sort=True):
        for metric in ["precision", "recall", "f1"]:
            vals = grp[metric].to_numpy(dtype=float)
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            sem = std / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
            tcrit = student_t.ppf(0.975, df=len(vals) - 1) if len(vals) > 1 else 0.0
            margin = tcrit * sem
            rows.append(
                {
                    "model_label": int(label),
                    "class_name": name,
                    "metric": metric,
                    "n_folds": len(vals),
                    "mean": mean,
                    "std": std,
                    "ci95_low": mean - margin,
                    "ci95_high": mean + margin,
                }
            )
    return pd.DataFrame(rows)


def write_paper_summary(
    path: Path,
    summary_df: pd.DataFrame,
    oof_metrics: Dict[str, float],
    oof_auc_macro: float,
    oof_auc_weighted: float,
) -> None:
    wanted = [
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "f1_macro",
        "auc_macro_ovr",
        "auc_weighted_ovr",
    ]
    lookup = summary_df.set_index("metric").to_dict("index")
    lines = [
        "IoT Shepherd ADM: stratified 10-fold cross-validation summary",
        "============================================================",
        "",
        "Fold-level mean ± SD (95% CI across folds):",
    ]
    for metric in wanted:
        if metric in lookup:
            r = lookup[metric]
            lines.append(
                f"{metric}: {r['mean']:.6f} ± {r['std']:.6f} "
                f"(95% CI {r['ci95_low']:.6f} to {r['ci95_high']:.6f})"
            )
    lines += [
        "",
        "Pooled out-of-fold metrics (each sample predicted only in its outer test fold):",
        f"accuracy: {oof_metrics['accuracy']:.6f}",
        f"precision_weighted: {oof_metrics['precision_weighted']:.6f}",
        f"recall_weighted: {oof_metrics['recall_weighted']:.6f}",
        f"f1_weighted: {oof_metrics['f1_weighted']:.6f}",
        f"f1_macro: {oof_metrics['f1_macro']:.6f}",
        f"auc_macro_ovr: {oof_auc_macro:.6f}",
        f"auc_weighted_ovr: {oof_auc_weighted:.6f}",
        "",
        "Recommended manuscript reporting: use the 10-fold mean ± SD as the primary CV robustness result,",
        "and use the pooled out-of-fold confusion matrix/ROC figures for visualization.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if not (0.0 < args.val_fraction < 0.5):
        raise ValueError("--val-fraction must be between 0 and 0.5")
    if args.max_folds is not None and args.max_folds < 1:
        raise ValueError("--max-folds must be >= 1")

    data_path = Path(args.data).resolve()
    out_dir = Path(args.output_dir).resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    prepare_output_dir(out_dir, overwrite=args.overwrite, resume=args.resume)
    set_global_seed(args.seed)

    config = RunConfig(
        data=str(data_path),
        output_dir=str(out_dir),
        model_name=args.model_name,
        seed=args.seed,
        n_splits=args.folds,
        val_fraction=args.val_fraction,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_folds=args.max_folds,
        keep_checkpoints=args.keep_checkpoints,
    )

    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": gpu_name(),
    }
    (out_dir / "run_config.json").write_text(
        json.dumps({"config": asdict(config), "environment": env}, indent=2),
        encoding="utf-8",
    )

    print(f"Loading dataset: {data_path}")
    df, texts, y, class_mapping = load_and_prepare_data(data_path)
    class_mapping.to_csv(out_dir / "class_mapping.csv", index=False)
    class_names = class_mapping.sort_values("model_label")["class_name"].tolist()
    n_classes = len(class_names)
    n_samples = len(df)

    label_counts = (
        pd.Series(y)
        .value_counts()
        .sort_index()
        .rename_axis("model_label")
        .reset_index(name="count")
        .merge(class_mapping, on="model_label", how="left")
    )
    label_counts["fraction"] = label_counts["count"] / n_samples
    label_counts.to_csv(out_dir / "class_distribution.csv", index=False)

    smallest_class = int(label_counts["count"].min())
    if smallest_class < args.folds:
        raise ValueError(
            f"Stratified {args.folds}-fold CV requires at least {args.folds} samples "
            f"in every class, but the smallest class has {smallest_class}."
        )

    print(
        f"Samples: {n_samples:,} | Classes: {n_classes} | "
        f"Smallest class: {smallest_class:,}"
    )
    print(f"CUDA available: {torch.cuda.is_available()} | GPU: {gpu_name()}")
    print("Loading tokenizer and tokenizing dataset once...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=args.max_length,
        padding=False,
    )
    del texts
    gc.collect()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    skf = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )

    oof_pred = np.full(n_samples, -1, dtype=np.int64)
    oof_fold = np.full(n_samples, -1, dtype=np.int64)
    oof_probs = np.full((n_samples, n_classes), np.nan, dtype=np.float32)

    fold_metric_rows: List[Dict[str, float]] = []
    per_class_frames: List[pd.DataFrame] = []
    per_class_auc_frames: List[pd.DataFrame] = []

    folds_to_run = args.folds if args.max_folds is None else min(args.max_folds, args.folds)

    for fold, (outer_train_idx, test_idx) in enumerate(skf.split(np.zeros(n_samples), y), start=1):
        if fold > folds_to_run:
            break

        fold_dir = out_dir / "folds" / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        pred_path = fold_dir / "predictions.csv"
        metric_path = fold_dir / "fold_metrics.json"
        class_path = fold_dir / "per_class_metrics.csv"
        auc_path = fold_dir / "per_class_auc.csv"

        if args.resume and pred_path.exists() and metric_path.exists() and class_path.exists() and auc_path.exists():
            print(f"[Fold {fold:02d}] Reusing completed fold outputs.")
            pred_df = pd.read_csv(pred_path)
            idx = pred_df["source_row_index"].to_numpy(dtype=int)
            oof_pred[idx] = pred_df["pred_model_label"].to_numpy(dtype=int)
            oof_fold[idx] = fold
            for c in range(n_classes):
                oof_probs[idx, c] = pred_df[f"prob_{c:02d}"].to_numpy(dtype=float)
            fold_metric_rows.append(json.loads(metric_path.read_text(encoding="utf-8")))
            per_class_frames.append(pd.read_csv(class_path))
            per_class_auc_frames.append(pd.read_csv(auc_path))
            continue

        split_seed = args.seed + fold

        fit_idx, val_idx = train_test_split(
            outer_train_idx,
            test_size=args.val_fraction,
            random_state=split_seed,
            stratify=y[outer_train_idx],
        )

        # Keep training stochasticity fixed across folds for comparability.
        set_global_seed(args.seed)

        print(
            f"\n[Fold {fold:02d}/{args.folds}] fit={len(fit_idx):,}, "
            f"val={len(val_idx):,}, test={len(test_idx):,}"
        )

        split_assignments = pd.DataFrame(
            {
                "source_row_index": np.concatenate([fit_idx, val_idx, test_idx]),
                "split": (
                    ["fit"] * len(fit_idx)
                    + ["validation"] * len(val_idx)
                    + ["test"] * len(test_idx)
                ),
                "fold": fold,
            }
        ).sort_values("source_row_index")
        split_assignments.to_csv(fold_dir / "split_assignments.csv", index=False)

        train_ds = EncodedSubset(encodings, y, fit_idx)
        val_ds = EncodedSubset(encodings, y, val_idx)
        test_ds = EncodedSubset(encodings, y, test_idx)

        model = BertForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=n_classes,
        )

        checkpoint_dir = fold_dir / "checkpoints"
        logging_dir = fold_dir / "trainer_logs"

        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            optim="adamw_torch",
            lr_scheduler_type="linear",
            logging_dir=str(logging_dir),
            report_to=[],
            seed=args.seed,
            data_seed=args.seed,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=trainer_compute_metrics,
        )

        train_result = trainer.train()
        pd.DataFrame(trainer.state.log_history).to_csv(
            fold_dir / "trainer_log_history.csv", index=False
        )

        pred_output = trainer.predict(test_ds, metric_key_prefix="test")
        logits = np.asarray(pred_output.predictions)
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        y_test = y[test_idx]

        base_metrics = metric_dict(y_test, preds)
        try:
            auc_macro = roc_auc_score(
                y_test,
                probs,
                labels=np.arange(n_classes),
                multi_class="ovr",
                average="macro",
            )
            auc_weighted = roc_auc_score(
                y_test,
                probs,
                labels=np.arange(n_classes),
                multi_class="ovr",
                average="weighted",
            )
        except ValueError:
            auc_macro = float("nan")
            auc_weighted = float("nan")

        eval_logs = [x for x in trainer.state.log_history if "eval_loss" in x]
        best_epoch = None
        if eval_logs:
            best_epoch = min(eval_logs, key=lambda x: x["eval_loss"]).get("epoch")

        test_runtime = float(pred_output.metrics.get("test_runtime", float("nan")))
        samples_per_second = (
            len(test_idx) / test_runtime if test_runtime and np.isfinite(test_runtime) else float("nan")
        )

        fold_metrics = {
            "fold": fold,
            "training_seed": args.seed,
            "validation_split_seed": split_seed,
            "n_fit": len(fit_idx),
            "n_validation": len(val_idx),
            "n_test": len(test_idx),
            **{k: float(v) for k, v in base_metrics.items()},
            "auc_macro_ovr": float(auc_macro),
            "auc_weighted_ovr": float(auc_weighted),
            "train_runtime_s": float(train_result.metrics.get("train_runtime", float("nan"))),
            "test_runtime_s": test_runtime,
            "test_samples_per_second": float(samples_per_second),
            "best_validation_loss": float(trainer.state.best_metric)
            if trainer.state.best_metric is not None
            else float("nan"),
            "best_epoch": float(best_epoch) if best_epoch is not None else float("nan"),
        }

        pc = per_class_metrics(y_test, preds, class_names, fold)
        pca = per_class_auc(y_test, probs, class_names, fold)
        pc.to_csv(class_path, index=False)
        pca.to_csv(auc_path, index=False)

        pred_df = pd.DataFrame(
            {
                "source_row_index": test_idx,
                "fold": fold,
                "true_model_label": y_test,
                "pred_model_label": preds,
                "true_class": [class_names[i] for i in y_test],
                "pred_class": [class_names[i] for i in preds],
            }
        )
        for c in range(n_classes):
            pred_df[f"prob_{c:02d}"] = probs[:, c]
        pred_df.to_csv(pred_path, index=False)
        metric_path.write_text(json.dumps(fold_metrics, indent=2), encoding="utf-8")

        oof_pred[test_idx] = preds
        oof_fold[test_idx] = fold
        oof_probs[test_idx, :] = probs

        fold_metric_rows.append(fold_metrics)
        per_class_frames.append(pc)
        per_class_auc_frames.append(pca)

        if not args.keep_checkpoints and checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

        del trainer, model, train_ds, val_ds, test_ds, logits, probs, pred_output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Smoke tests intentionally stop before all OOF predictions exist.
    completed_mask = oof_fold >= 0
    completed_folds = int(len(np.unique(oof_fold[completed_mask])))

    fold_metrics_df = pd.DataFrame(fold_metric_rows).sort_values("fold")
    fold_metrics_df.to_csv(out_dir / "fold_metrics.csv", index=False)

    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "auc_macro_ovr",
        "auc_weighted_ovr",
        "train_runtime_s",
        "test_runtime_s",
        "test_samples_per_second",
    ]
    summary_df = summarize_columns(fold_metrics_df, metric_cols)
    summary_df.to_csv(out_dir / "fold_metrics_summary.csv", index=False)

    per_class_all = pd.concat(per_class_frames, ignore_index=True)
    per_class_all.to_csv(out_dir / "per_class_metrics_all_folds.csv", index=False)
    per_class_summary = summarize_per_class(per_class_all)
    per_class_summary.to_csv(out_dir / "per_class_metrics_summary.csv", index=False)

    per_class_auc_all = pd.concat(per_class_auc_frames, ignore_index=True)
    per_class_auc_all.to_csv(out_dir / "per_class_auc_all_folds.csv", index=False)
    auc_summary_rows = []
    for (label, name), grp in per_class_auc_all.groupby(["model_label", "class_name"], sort=True):
        vals = grp["auc_ovr"].dropna().to_numpy(dtype=float)
        mean = float(np.mean(vals)) if len(vals) else float("nan")
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        sem = std / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        tcrit = student_t.ppf(0.975, df=len(vals) - 1) if len(vals) > 1 else 0.0
        auc_summary_rows.append(
            {
                "model_label": int(label),
                "class_name": name,
                "n_folds": len(vals),
                "mean_auc_ovr": mean,
                "std_auc_ovr": std,
                "ci95_low": mean - tcrit * sem,
                "ci95_high": mean + tcrit * sem,
            }
        )
    pd.DataFrame(auc_summary_rows).to_csv(out_dir / "per_class_auc_summary.csv", index=False)

    plot_fold_metrics(fold_metrics_df, out_dir / "figures" / "fold_metrics")
    plot_per_class_f1(per_class_summary, out_dir / "figures" / "per_class_f1")

    if completed_folds == args.folds:
        if np.any(oof_pred < 0) or np.isnan(oof_probs).any():
            raise RuntimeError("OOF predictions are incomplete despite all folds being marked complete.")

        oof_df = pd.DataFrame(
            {
                "source_row_index": np.arange(n_samples),
                "fold": oof_fold,
                "true_model_label": y,
                "pred_model_label": oof_pred,
                "true_class": [class_names[i] for i in y],
                "pred_class": [class_names[i] for i in oof_pred],
            }
        )
        for c in range(n_classes):
            oof_df[f"prob_{c:02d}"] = oof_probs[:, c]
        oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)

        oof_metrics = metric_dict(y, oof_pred)
        oof_auc_macro = roc_auc_score(
            y,
            oof_probs,
            labels=np.arange(n_classes),
            multi_class="ovr",
            average="macro",
        )
        oof_auc_weighted = roc_auc_score(
            y,
            oof_probs,
            labels=np.arange(n_classes),
            multi_class="ovr",
            average="weighted",
        )
        oof_metrics.update(
            {
                "auc_macro_ovr": float(oof_auc_macro),
                "auc_weighted_ovr": float(oof_auc_weighted),
            }
        )
        (out_dir / "oof_metrics.json").write_text(
            json.dumps(oof_metrics, indent=2), encoding="utf-8"
        )

        cm = confusion_matrix(y, oof_pred, labels=np.arange(n_classes))
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
            out_dir / "confusion_matrix_counts.csv"
        )
        pd.DataFrame(cm_norm, index=class_names, columns=class_names).to_csv(
            out_dir / "confusion_matrix_normalized.csv"
        )
        plot_confusion_matrix(
            cm,
            class_names,
            out_dir / "figures" / "confusion_matrix_counts",
            normalized=False,
        )
        plot_confusion_matrix(
            cm_norm,
            class_names,
            out_dir / "figures" / "confusion_matrix_normalized",
            normalized=True,
        )

        oof_auc_by_class = plot_oof_roc(
            y,
            oof_probs,
            class_names,
            out_dir / "figures" / "roc_oof_all_classes",
            out_dir / "roc_curve_oof.csv",
        )
        pd.DataFrame(
            [{"class_name": k, "oof_auc": v} for k, v in oof_auc_by_class.items()]
        ).to_csv(out_dir / "oof_auc_by_class.csv", index=False)

        write_paper_summary(
            out_dir / "paper_summary.txt",
            summary_df,
            oof_metrics,
            float(oof_auc_macro),
            float(oof_auc_weighted),
        )

        print("\n10-fold CV complete.")
        print(f"Results directory: {out_dir}")
        print(f"Paper summary: {out_dir / 'paper_summary.txt'}")
    else:
        print(
            f"\nCompleted {completed_folds}/{args.folds} folds (smoke/partial run). "
            "OOF aggregate files will be created after all folds are completed."
        )
        print(f"Partial results directory: {out_dir}")


if __name__ == "__main__":
    main()

