#!/usr/bin/env python3
"""
Unified evaluation script for:
- QA quality metrics (BLEU, ROUGE-1/2/L, METEOR, BERTScore P/R/F1) for:
  * With-Context (RAG) vs No-Context (NC)
  * Multiple models
  * Multiple use cases
- Clean system metrics (latency around retrieval+LLM only, plus CPU/RAM/GPU deltas)

This script is designed to be robust:
- Supports DOCX QA formats:
    1) Question: Answer
    2) "Question" : "Answer",
- Keeps inference timing clean: metric computations happen AFTER timing stops.
- Handles environments with no GPU / no NVML / no nvidia-smi gracefully.
"""

import os
import re
import json
import time
import math
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import psutil
import pandas as pd
from docx import Document

# --- Quality metrics ---
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# BERTScore is expensive; we still compute it, but always outside the inference timer.
try:
    from bert_score import score as bert_score
    _HAS_BERTSCORE = True
except Exception:
    bert_score = None
    _HAS_BERTSCORE = False

# --- Token/size ---
try:
    from transformers import AutoTokenizer
    _HAS_HF = True
except Exception:
    AutoTokenizer = None
    _HAS_HF = False

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

# --- LangChain / Chroma / Ollama ---
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM as Ollama
from get_embedding_function import get_embedding_function

# --- GPU monitoring (optional) ---
_HAS_NVML = False
try:
    from pynvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetUtilizationRates,
    )
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False


# =========================
# User configuration
# =========================
CHROMA_PATH = "chroma"

MODELS = [
    "llama3.1:8b",
    "llava:7b",
    "gemma2:9b",
    "mistral:7b",
]

USE_CASE_FILES = {
    "Troubleshooting": "Evaluation/troubleshoot.docx",
    "Device Management": "Evaluation/device_management.docx",
    "Maintenance": "Evaluation/maintenance.docx",
    "Safety": "Evaluation/safety.docx",
    "Setup": "Evaluation/setup.docx",
}

TOP_K = 5  # retrieval k
SCORE_SCALE = 100.0  # scale quality metrics to 0–100 for summaries (kept raw in per-question logs too)


# =========================
# Utilities
# =========================
def _safe_model_tag(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _has_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None


# =========================
# GPU Monitoring
# =========================
def initialize_nvml() -> bool:
    if not _HAS_NVML:
        return False
    try:
        nvmlInit()
        return True
    except Exception:
        return False


def shutdown_nvml() -> None:
    if not _HAS_NVML:
        return
    try:
        nvmlShutdown()
    except Exception:
        pass


def get_gpu_usage_nvml(gpu_index: int = 0) -> Dict[str, float]:
    if not _HAS_NVML:
        return {"gpu_memory_used_mb": 0.0, "gpu_memory_total_mb": 0.0, "gpu_utilization_pct": 0.0}
    try:
        handle = nvmlDeviceGetHandleByIndex(gpu_index)
        memory_info = nvmlDeviceGetMemoryInfo(handle)
        utilization = nvmlDeviceGetUtilizationRates(handle)
        return {
            "gpu_memory_used_mb": memory_info.used / (1024 ** 2),
            "gpu_memory_total_mb": memory_info.total / (1024 ** 2),
            "gpu_utilization_pct": float(utilization.gpu),
        }
    except Exception:
        return {"gpu_memory_used_mb": 0.0, "gpu_memory_total_mb": 0.0, "gpu_utilization_pct": 0.0}


def get_gpu_usage_nvidia_smi() -> Dict[str, float]:
    if not _has_nvidia_smi():
        return {"smi_gpu_memory_used_mb": 0.0, "smi_gpu_memory_total_mb": 0.0, "smi_gpu_utilization_pct": 0.0}
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        # If multiple GPUs, take GPU0 for consistency
        line = result.strip().splitlines()[0]
        fields = [x.strip() for x in line.split(",")]
        if len(fields) != 3:
            raise ValueError("Unexpected nvidia-smi output")
        return {
            "smi_gpu_memory_used_mb": float(fields[0]),
            "smi_gpu_memory_total_mb": float(fields[1]),
            "smi_gpu_utilization_pct": float(fields[2]),
        }
    except Exception:
        return {"smi_gpu_memory_used_mb": 0.0, "smi_gpu_memory_total_mb": 0.0, "smi_gpu_utilization_pct": 0.0}


def get_gpu_usage_torch(device: int = 0) -> Dict[str, float]:
    if not _HAS_TORCH or torch is None or not torch.cuda.is_available():
        return {
            "torch_memory_allocated_mb": 0.0,
            "torch_memory_reserved_mb": 0.0,
            "torch_max_memory_allocated_mb": 0.0,
        }
    try:
        return {
            "torch_memory_allocated_mb": torch.cuda.memory_allocated(device) / (1024 ** 2),
            "torch_memory_reserved_mb": torch.cuda.memory_reserved(device) / (1024 ** 2),
            "torch_max_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        }
    except Exception:
        return {
            "torch_memory_allocated_mb": 0.0,
            "torch_memory_reserved_mb": 0.0,
            "torch_max_memory_allocated_mb": 0.0,
        }


# =========================
# DOCX QA loading
# =========================
def _normalize_quotes(s: str) -> str:
    # normalize curly quotes to straight quotes
    return (
        s.replace("“", '"')
         .replace("”", '"')
         .replace("‘", "'")
         .replace("’", "'")
    )


def _parse_safety_style(line: str) -> Optional[Tuple[str, str]]:
    """
    Parse:  "Question" : "Answer",\t
    Robustly:
    - Finds first quoted segment as question
    - Finds last quoted segment as answer (after the colon)
    """
    s = _normalize_quotes(line).strip()
    if not s.startswith('"'):
        return None
    # Find end of question (second quote)
    q2 = s.find('"', 1)
    if q2 == -1:
        return None
    question = s[1:q2].strip()

    # Find colon after question
    colon = s.find(":", q2 + 1)
    if colon == -1:
        return None

    # Find first quote for answer after colon
    a1 = s.find('"', colon + 1)
    if a1 == -1:
        return None

    # Find the closing quote for the answer: use rfind to tolerate commas/tabs after
    a2 = s.rfind('"')
    if a2 == -1 or a2 <= a1:
        return None

    answer = s[a1 + 1:a2].strip()
    if not question or not answer:
        return None
    return question, answer


def load_qa_pairs_docx(file_path: str) -> List[Tuple[str, str]]:
    """
    Returns list of (question, reference_answer) preserving file order.
    Supports:
      1) Question: Answer
      2) "Question" : "Answer",
    """
    document = Document(file_path)
    pairs: List[Tuple[str, str]] = []

    for p in document.paragraphs:
        line = p.text.strip()
        if not line:
            continue

        # Try safety.pdf-style first
        parsed = _parse_safety_style(line)
        if parsed:
            pairs.append(parsed)
            continue

        # Fallback: plain "Question: Answer"
        if ":" in line:
            q, a = line.split(":", 1)
            q = _normalize_quotes(q).strip().strip('"').strip()
            a = _normalize_quotes(a).strip().rstrip(",").strip().strip('"').strip()
            if q and a:
                pairs.append((q, a))

    return pairs


# =========================
# Token/size
# =========================
class TokenSizer:
    """
    Robust response size + token count calculator:
    - Attempts to load a HF tokenizer once (default: gpt2).
    - Falls back to whitespace tokenization if HF tokenizer isn't available.
    """
    def __init__(self, tokenizer_name: str = "gpt2"):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = None
        if _HAS_HF and AutoTokenizer is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
            except Exception:
                # Try without local_files_only (may still work if cached/configured)
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                except Exception:
                    self.tokenizer = None

    def size_and_tokens(self, text: str) -> Tuple[int, int]:
        b = len(text.encode("utf-8"))
        if self.tokenizer is None:
            # fallback tokenization
            tokens = text.split()
            return b, len(tokens)
        try:
            toks = self.tokenizer.tokenize(text)
            return b, len(toks)
        except Exception:
            tokens = text.split()
            return b, len(tokens)


# =========================
# Quality metrics
# =========================
_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
_SMOOTHIE = SmoothingFunction().method1


def evaluate_answer(generated_answer: str, reference_answer: str) -> Dict[str, float]:
    """
    Returns raw metric scores in [0,1] (when defined).
    If a metric fails (e.g., missing resources), it returns NaN for that metric.
    """
    out = {
        "bert_Precision": math.nan,
        "bert_Recall": math.nan,
        "bert_F1": math.nan,
        "rouge1": math.nan,
        "rouge2": math.nan,
        "rougeL": math.nan,
        "bleu": math.nan,
        "meteor": math.nan,
    }

    ref_tokens = reference_answer.split()
    gen_tokens = generated_answer.split()

    # ROUGE
    try:
        rs = _ROUGE.score(reference_answer, generated_answer)
        out["rouge1"] = rs["rouge1"].fmeasure
        out["rouge2"] = rs["rouge2"].fmeasure
        out["rougeL"] = rs["rougeL"].fmeasure
    except Exception:
        pass

    # BLEU
    try:
        out["bleu"] = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=_SMOOTHIE)
    except Exception:
        pass

    # METEOR
    try:
        out["meteor"] = meteor_score([ref_tokens], gen_tokens)
    except Exception:
        pass

    # BERTScore
    if _HAS_BERTSCORE and bert_score is not None:
        try:
            P, R, F1 = bert_score([generated_answer], [reference_answer], lang="en", verbose=False)
            out["bert_Precision"] = P.mean().item()
            out["bert_Recall"] = R.mean().item()
            out["bert_F1"] = F1.mean().item()
        except Exception:
            pass

    return out


# =========================
# Inference + system metrics
# =========================
@dataclass
class InferenceSystemMetrics:
    total_time_s: float
    retrieval_time_s: float
    llm_time_s: float

    rss_diff_mb: float
    est_cpu_usage_pct: float
    psutil_cpu_usage_pct: float

    response_bytes: int
    response_token_count: int

    nvml_gpu_memory_used_diff_mb: float
    nvml_gpu_utilization_diff_pct: float

    smi_gpu_memory_used_diff_mb: float
    smi_gpu_utilization_diff_pct: float

    torch_memory_allocated_diff_mb: float
    torch_memory_reserved_diff_mb: float


def _snapshot_process(process: psutil.Process) -> Dict[str, Any]:
    return {
        "rss_mb": process.memory_info().rss / (1024 ** 2),
        "cpu_times": process.cpu_times(),  # user/system
    }


def _snapshot_system(process: psutil.Process) -> Dict[str, Any]:
    snap = _snapshot_process(process)
    snap["nvml"] = get_gpu_usage_nvml()
    snap["smi"] = get_gpu_usage_nvidia_smi()
    snap["torch"] = get_gpu_usage_torch()
    return snap


def run_inference_with_metrics(
    *,
    model: Ollama,
    db: Optional[Chroma],
    question: str,
    mode: str,  # "rag" or "no_context"
    token_sizer: TokenSizer,
    k: int,
) -> Tuple[str, InferenceSystemMetrics]:
    """
    Returns (response_text, system_metrics) for one question.
    Timing is clean: starts right before retrieval/prompt/LLM and stops right after LLM returns.
    """
    process = psutil.Process()

    # Prime cpu_percent so later call is meaningful
    _ = process.cpu_percent(interval=None)
    _ = psutil.cpu_percent(interval=None)

    before = _snapshot_system(process)
    psutil_cpu_before = psutil.cpu_percent(interval=None)

    t0 = time.perf_counter()

    retrieval_time = 0.0
    if mode == "rag":
        if db is None:
            raise RuntimeError("Chroma DB is None, cannot run RAG mode.")
        t_retr_start = time.perf_counter()
        results_db = db.similarity_search_with_score(question, k=k)
        t_retr_end = time.perf_counter()
        retrieval_time = t_retr_end - t_retr_start

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results_db])
        prompt = (
            "Answer the question based only on the following context:\n\n"
            f"{context_text}\n\n---\n\n"
            f"Answer the question based on the above context: {question}"
        )
    else:
        prompt = f"Please provide an answer to the following question: {question}"

    t_llm_start = time.perf_counter()
    response_text = model.invoke(prompt)
    t_llm_end = time.perf_counter()

    t1 = time.perf_counter()

    after = _snapshot_system(process)
    psutil_cpu_after = psutil.cpu_percent(interval=None)

    # Clean timing outputs
    total_time = t1 - t0
    llm_time = t_llm_end - t_llm_start

    # CPU estimate
    cpu_times_before = before["cpu_times"]
    cpu_times_after = after["cpu_times"]
    total_cpu_time = (cpu_times_after.user - cpu_times_before.user) + (cpu_times_after.system - cpu_times_before.system)
    est_cpu_usage = (total_cpu_time / total_time) * 100 if total_time > 0 else 0.0
    psutil_cpu_avg = (psutil_cpu_before + psutil_cpu_after) / 2.0

    # Memory diff
    rss_diff = after["rss_mb"] - before["rss_mb"]

    # Response size/tokens (AFTER timing)
    response_bytes, response_tokens = token_sizer.size_and_tokens(response_text)

    # GPU diffs
    nvml_before, nvml_after = before["nvml"], after["nvml"]
    smi_before, smi_after = before["smi"], after["smi"]
    torch_before, torch_after = before["torch"], after["torch"]

    metrics = InferenceSystemMetrics(
        total_time_s=total_time,
        retrieval_time_s=retrieval_time,
        llm_time_s=llm_time,
        rss_diff_mb=rss_diff,
        est_cpu_usage_pct=est_cpu_usage,
        psutil_cpu_usage_pct=psutil_cpu_avg,
        response_bytes=int(response_bytes),
        response_token_count=int(response_tokens),
        nvml_gpu_memory_used_diff_mb=float(nvml_after["gpu_memory_used_mb"] - nvml_before["gpu_memory_used_mb"]),
        nvml_gpu_utilization_diff_pct=float(nvml_after["gpu_utilization_pct"] - nvml_before["gpu_utilization_pct"]),
        smi_gpu_memory_used_diff_mb=float(smi_after["smi_gpu_memory_used_mb"] - smi_before["smi_gpu_memory_used_mb"]),
        smi_gpu_utilization_diff_pct=float(smi_after["smi_gpu_utilization_pct"] - smi_before["smi_gpu_utilization_pct"]),
        torch_memory_allocated_diff_mb=float(torch_after["torch_memory_allocated_mb"] - torch_before["torch_memory_allocated_mb"]),
        torch_memory_reserved_diff_mb=float(torch_after["torch_memory_reserved_mb"] - torch_before["torch_memory_reserved_mb"]),
    )

    return response_text, metrics


# =========================
# Aggregation helpers
# =========================
QUALITY_KEYS = [
    "bert_Precision",
    "bert_Recall",
    "bert_F1",
    "rouge1",
    "rouge2",
    "rougeL",
    "bleu",
    "meteor",
]

SYSTEM_KEYS = [
    "total_time_s",
    "retrieval_time_s",
    "llm_time_s",
    "rss_diff_mb",
    "est_cpu_usage_pct",
    "psutil_cpu_usage_pct",
    "response_bytes",
    "response_token_count",
    "nvml_gpu_memory_used_diff_mb",
    "nvml_gpu_utilization_diff_pct",
    "smi_gpu_memory_used_diff_mb",
    "smi_gpu_utilization_diff_pct",
    "torch_memory_allocated_diff_mb",
    "torch_memory_reserved_diff_mb",
]


def _scaled_mean(series: pd.Series) -> float:
    # mean ignoring NaNs
    m = series.dropna().mean()
    if pd.isna(m):
        return float("nan")
    return float(m) * SCORE_SCALE


def _scaled_std(series: pd.Series) -> float:
    s = series.dropna().std()
    if pd.isna(s):
        return float("nan")
    return float(s) * SCORE_SCALE


def build_improvement_table(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Builds a screenshot-style table:
      RAG Mean | No Context Mean | Absolute Difference | Percentage Improvement
    for each quality metric, grouped by group_cols (e.g., ["model", "use_case"] or ["model"]).
    Assumes df has columns:
      rag_<metric>, no_context_<metric>
    """
    rows = []
    for _, g in df.groupby(group_cols):
        row_base = {c: g.iloc[0][c] for c in group_cols}
        for m in QUALITY_KEYS:
            rag_col = f"rag_{m}"
            nc_col = f"no_context_{m}"
            rag_mean = _scaled_mean(g[rag_col])
            nc_mean = _scaled_mean(g[nc_col])
            diff = rag_mean - nc_mean
            pct = (diff / nc_mean * 100.0) if (not math.isnan(nc_mean) and nc_mean != 0) else float("nan")
            row_base[f"{m}_RAG_Mean"] = rag_mean
            row_base[f"{m}_NC_Mean"] = nc_mean
            row_base[f"{m}_Abs_Diff"] = diff
            row_base[f"{m}_Pct_Improvement"] = pct
        rows.append(row_base)
    return pd.DataFrame(rows)


def summarize_metrics(df: pd.DataFrame, group_cols: List[str], prefix: str) -> pd.DataFrame:
    """
    Summarize system metrics with mean and std per group.
    prefix is 'rag' or 'no_context' and expects columns like:
      rag_total_time_s, rag_rss_diff_mb, ...
    """
    rows = []
    for _, g in df.groupby(group_cols):
        row = {c: g.iloc[0][c] for c in group_cols}
        for k in SYSTEM_KEYS:
            col = f"{prefix}_{k}"
            series = g[col]
            row[f"{prefix}_{k}_mean"] = float(series.dropna().mean()) if series.dropna().shape[0] else float("nan")
            row[f"{prefix}_{k}_std"] = float(series.dropna().std()) if series.dropna().shape[0] else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# =========================
# Main
# =========================
def main() -> None:
    # Validate input files
    for uc, path in USE_CASE_FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing use-case evaluation file: {uc} -> {path}")

    if not os.path.isdir(CHROMA_PATH):
        raise FileNotFoundError(f"Chroma directory not found at '{CHROMA_PATH}'. Please ensure the DB is populated.")

    # Initialize NVML if available
    nvml_ok = initialize_nvml()

    # Load Chroma once
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Token sizing helper
    token_sizer = TokenSizer(tokenizer_name="gpt2")

    # Output directory
    run_tag = _now_tag()
    run_dir = os.path.join("Evaluation", "runs", run_tag)
    _ensure_dir(run_dir)

    config_path = os.path.join(run_dir, "config.json")
    _write_json(config_path, {
        "models": MODELS,
        "use_case_files": USE_CASE_FILES,
        "chroma_path": CHROMA_PATH,
        "top_k": TOP_K,
        "score_scale_for_summaries": SCORE_SCALE,
        "nvml_available": nvml_ok,
        "nvidia_smi_available": _has_nvidia_smi(),
        "torch_cuda_available": bool(_HAS_TORCH and torch is not None and torch.cuda.is_available()),
        "bert_score_available": _HAS_BERTSCORE,
        "started_at": run_tag,
    })

    all_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    # Main evaluation loops
    for model_name in MODELS:
        model_tag = _safe_model_tag(model_name)
        model_out_dir = os.path.join(run_dir, "per_question", model_tag)
        _ensure_dir(model_out_dir)

        model = Ollama(model=model_name)

        for use_case, qa_path in USE_CASE_FILES.items():
            pairs = load_qa_pairs_docx(qa_path)
            if not pairs:
                error_rows.append({
                    "model": model_name,
                    "use_case": use_case,
                    "qa_file": qa_path,
                    "error": "No QA pairs parsed from DOCX",
                })
                continue

            # Per use-case output list
            for idx, (question, reference_answer) in enumerate(pairs, start=1):
                row: Dict[str, Any] = {
                    "model": model_name,
                    "use_case": use_case,
                    "qa_file": qa_path,
                    "qa_index": idx,
                    "question": question,
                    "reference_answer": reference_answer,
                }
                try:
                    # RAG (with context) – timed cleanly
                    rag_answer, rag_sys = run_inference_with_metrics(
                        model=model,
                        db=db,
                        question=question,
                        mode="rag",
                        token_sizer=token_sizer,
                        k=TOP_K,
                    )

                    # No context – timed cleanly
                    nc_answer, nc_sys = run_inference_with_metrics(
                        model=model,
                        db=None,
                        question=question,
                        mode="no_context",
                        token_sizer=token_sizer,
                        k=0,
                    )

                    # Quality metrics (outside timing)
                    rag_scores = evaluate_answer(rag_answer, reference_answer)
                    nc_scores = evaluate_answer(nc_answer, reference_answer)

                    row["rag_answer"] = rag_answer
                    row["no_context_answer"] = nc_answer

                    for k, v in rag_scores.items():
                        row[f"rag_{k}"] = v
                    for k, v in nc_scores.items():
                        row[f"no_context_{k}"] = v

                    # System metrics
                    for k in SYSTEM_KEYS:
                        row[f"rag_{k}"] = getattr(rag_sys, k)
                        row[f"no_context_{k}"] = getattr(nc_sys, k)

                    all_rows.append(row)

                except Exception as e:
                    error_rows.append({
                        "model": model_name,
                        "use_case": use_case,
                        "qa_file": qa_path,
                        "qa_index": idx,
                        "question": question,
                        "error": repr(e),
                    })

            # Save per-question CSV for this model/use-case (filtered from all_rows)
            df_uc = pd.DataFrame([r for r in all_rows if r["model"] == model_name and r["use_case"] == use_case])
            out_csv = os.path.join(model_out_dir, f"{_safe_model_tag(use_case)}.csv")
            df_uc.to_csv(out_csv, index=False)

    # Save all-rows master
    df_all = pd.DataFrame(all_rows)
    df_all_path = os.path.join(run_dir, "all_per_question.csv")
    df_all.to_csv(df_all_path, index=False)

    # Save errors
    if error_rows:
        df_err = pd.DataFrame(error_rows)
        df_err.to_csv(os.path.join(run_dir, "errors.csv"), index=False)

    # Build summary tables (quality)
    if not df_all.empty:
        # Improvement table per model/use_case and per model overall
        quality_by_model_usecase = build_improvement_table(df_all, ["model", "use_case"])
        quality_by_model = build_improvement_table(df_all, ["model"])

        quality_by_model_usecase.to_csv(os.path.join(run_dir, "summary_quality_by_model_usecase.csv"), index=False)
        quality_by_model.to_csv(os.path.join(run_dir, "summary_quality_by_model.csv"), index=False)

        # Also provide Table-I style means (RAG vs NC) per model/use_case
        # (scaled and raw)
        table_rows = []
        for (model, use_case), g in df_all.groupby(["model", "use_case"]):
            r = {"model": model, "use_case": use_case}
            for m in QUALITY_KEYS:
                r[f"rag_{m}_mean_raw"] = float(g[f"rag_{m}"].dropna().mean()) if g[f"rag_{m}"].dropna().shape[0] else float("nan")
                r[f"no_context_{m}_mean_raw"] = float(g[f"no_context_{m}"].dropna().mean()) if g[f"no_context_{m}"].dropna().shape[0] else float("nan")
                r[f"rag_{m}_mean"] = r[f"rag_{m}_mean_raw"] * SCORE_SCALE if not math.isnan(r[f"rag_{m}_mean_raw"]) else float("nan")
                r[f"no_context_{m}_mean"] = r[f"no_context_{m}_mean_raw"] * SCORE_SCALE if not math.isnan(r[f"no_context_{m}_mean_raw"]) else float("nan")
            table_rows.append(r)
        pd.DataFrame(table_rows).to_csv(os.path.join(run_dir, "table1_quality_means.csv"), index=False)

        # System metrics summaries (mean+std) per model/use_case for RAG and NC
        sys_rag = summarize_metrics(df_all, ["model", "use_case"], prefix="rag")
        sys_nc = summarize_metrics(df_all, ["model", "use_case"], prefix="no_context")
        sys_rag.to_csv(os.path.join(run_dir, "summary_system_rag_by_model_usecase.csv"), index=False)
        sys_nc.to_csv(os.path.join(run_dir, "summary_system_no_context_by_model_usecase.csv"), index=False)

        # Overall per model
        sys_rag_model = summarize_metrics(df_all, ["model"], prefix="rag")
        sys_nc_model = summarize_metrics(df_all, ["model"], prefix="no_context")
        sys_rag_model.to_csv(os.path.join(run_dir, "summary_system_rag_by_model.csv"), index=False)
        sys_nc_model.to_csv(os.path.join(run_dir, "summary_system_no_context_by_model.csv"), index=False)

    # Shutdown NVML if it was initialized
    shutdown_nvml()

    print(f"Evaluation complete. Outputs saved under: {run_dir}")


if __name__ == "__main__":
    main()

