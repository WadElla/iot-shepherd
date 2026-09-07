#!/usr/bin/env python3
"""
IoT Shepherd end-to-end incident-to-mitigation evaluation.

This script evaluates a bounded retrieval-repair agent for the IoT Shepherd
end-to-end experiment. It is a standalone batch evaluator, not a Streamlit UI.

Two modes are evaluated for every model, attack case, and QA pair:

1) with_context
   - Python first retrieves mitigation-guide chunks using the original
     administrator question exactly as written in the QA file.
   - The LLM reviews whether those chunks are sufficient to answer the question
     for the dominant attack described in the compact incident summary.
   - If the chunks are insufficient, the LLM gets exactly one chance to rewrite
     the retrieval query while preserving the original question intent and
     focusing on the dominant attack.
   - Python executes the retry query exactly as returned by the LLM.
   - The LLM generates the final answer from the compact incident summary and
     the selected mitigation-guide chunks.

2) incident_only
   - No retrieval, no planning, no chunk review, and no retry.
   - The LLM answers directly from the compact incident summary and question.

Design choices:
- Python controls the outer experiment loop and tool execution.
- Python does not filter Chroma retrieval by attack type or PDF source.
- Python does not rewrite the initial query. The first query is the exact QA
  question. If retry is needed, Python uses the LLM's retry query as-is.
- The workflow is bounded: original-question retrieval plus at most one retry.
- Only timing and answer-quality metrics are measured. CPU/RAM/GPU metrics are
  not sampled.
- Quality metrics are computed outside the timed inference window.

Expected folder layout from the project root:

  chroma/
  data/
    Backdoor_Mitigation_Guide.pdf
    DDoS_HTTP_Mitigation_Guide.pdf
    Ransomware_Mitigation_Guide.pdf
  Shepherd_Eval/
    incident_summaries/
      Backdoor_incident_summary.txt
      DDoS_HTTP_incident_summary.txt
      Ransomware_incident_summary.txt
    qa/
      Backdoor.docx
      DDoS_HTTP.docx
      Ransomware.docx
    runs/

Run:
  python performance_shepherd_end_to_end_harness_retry.py

Quick test:
  python performance_shepherd_end_to_end_harness_retry.py --skip-bertscore --models gemma2:9b --cases Backdoor

Recommended final run:
  nohup python -u performance_shepherd_end_to_end_harness_retry.py > shepherd_e2e_harness_retry.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from docx import Document

# -------------------------
# Quality metrics
# -------------------------
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

try:
    from bert_score import score as bert_score
    _HAS_BERTSCORE = True
except Exception:
    bert_score = None
    _HAS_BERTSCORE = False

# -------------------------
# Token helper
# -------------------------
try:
    from transformers import AutoTokenizer
    _HAS_HF = True
except Exception:
    AutoTokenizer = None
    _HAS_HF = False

# -------------------------
# Chroma / Ollama compatibility fallbacks
# -------------------------
try:
    from langchain_chroma import Chroma
except Exception:  # pragma: no cover
    try:
        from langchain.vectorstores.chroma import Chroma
    except Exception:  # pragma: no cover
        from langchain.vectorstores import Chroma

try:
    from langchain_ollama import OllamaLLM as Ollama
except Exception:  # pragma: no cover
    try:
        from langchain_community.llms import Ollama
    except Exception:  # pragma: no cover
        from langchain.llms import Ollama

from get_embedding_function import get_embedding_function


# =========================
# Defaults
# =========================
DEFAULT_MODELS = [
    "gemma2:9b",
    "llama3.1:8b",
    "mistral:7b",
    "llava:7b",
]

DEFAULT_CASES = ["Backdoor", "DDoS_HTTP", "Ransomware"]

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

TIME_KEYS = [
    "total_time_s",
    "initial_retrieval_time_s",
    "chunk_review_time_s",
    "retry_retrieval_time_s",
    "generation_time_s",
]

SYSTEM_KEYS = TIME_KEYS + [
    "retrieval_attempt_count",
    "retry_used",
    "review_sufficient",
    "review_parse_ok",
    "review_fallback_used",
    "retrieved_chunk_count",
    "initial_retrieved_chunk_count",
    "retry_retrieved_chunk_count",
    "context_char_count",
    "context_token_count",
    "response_bytes",
    "response_token_count",
]

SCORE_SCALE = 100.0
_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
_SMOOTHIE = SmoothingFunction().method1


# =========================
# Utilities
# =========================
def safe_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def basename(path: str) -> str:
    return os.path.basename(path.replace("\\", "/"))


def normalize_quotes(text: str) -> str:
    return (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def run_unmeasured_warmup(model: Any, model_name: str) -> Dict[str, Any]:
    """Run one short unmeasured call before timed evaluation for a model."""
    prompt = "Reply with OK only."
    t0 = time.perf_counter()
    try:
        response = str(model.invoke(prompt)).strip()
        return {
            "model": model_name,
            "status": "ok",
            "elapsed_s": time.perf_counter() - t0,
            "response_preview": response[:80],
        }
    except Exception as exc:
        return {
            "model": model_name,
            "status": "error",
            "elapsed_s": time.perf_counter() - t0,
            "error": repr(exc),
        }


# =========================
# DOCX QA loading
# =========================
def parse_quoted_qa(line: str) -> Optional[Tuple[str, str]]:
    """Parse lines of the form: "Question" : "Answer", """
    s = normalize_quotes(line).strip()
    if not s.startswith('"'):
        return None

    q2 = s.find('"', 1)
    if q2 == -1:
        return None
    question = s[1:q2].strip()

    colon = s.find(":", q2 + 1)
    if colon == -1:
        return None

    a1 = s.find('"', colon + 1)
    a2 = s.rfind('"')
    if a1 == -1 or a2 == -1 or a2 <= a1:
        return None

    answer = s[a1 + 1 : a2].strip()
    if not question or not answer:
        return None
    return question, answer


def load_qa_pairs_docx(file_path: str) -> List[Tuple[str, str]]:
    """
    Load QA pairs from a DOCX file with no headings.

    Supported formats:
      1) Question: Answer
      2) "Question" : "Answer",
    """
    document = Document(file_path)
    pairs: List[Tuple[str, str]] = []

    for paragraph in document.paragraphs:
        line = paragraph.text.strip()
        if not line:
            continue

        parsed = parse_quoted_qa(line)
        if parsed is not None:
            pairs.append(parsed)
            continue

        if ":" in line:
            question, answer = line.split(":", 1)
            question = normalize_quotes(question).strip().strip('"').strip()
            answer = normalize_quotes(answer).strip().rstrip(",").strip().strip('"').strip()
            if question and answer:
                pairs.append((question, answer))

    return pairs


# =========================
# Token sizing
# =========================
class TokenSizer:
    """Response/context size helper. Default is whitespace token counting."""

    def __init__(self, tokenizer_name: str = "whitespace") -> None:
        self.tokenizer = None
        self.tokenizer_name = tokenizer_name

        if tokenizer_name.lower() in {"none", "simple", "whitespace"}:
            return

        if _HAS_HF and AutoTokenizer is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
            except Exception:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                except Exception:
                    self.tokenizer = None

    def size_and_tokens(self, text: str) -> Tuple[int, int]:
        text = text or ""
        response_bytes = len(text.encode("utf-8"))
        if self.tokenizer is not None:
            try:
                return response_bytes, len(self.tokenizer.tokenize(text))
            except Exception:
                pass
        return response_bytes, len(text.split())


# =========================
# Quality metrics
# =========================
def evaluate_answer(generated_answer: str, reference_answer: str, *, compute_bertscore: bool) -> Dict[str, float]:
    """Compute answer-quality metrics outside the timed inference window."""
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

    try:
        rouge_scores = _ROUGE.score(reference_answer, generated_answer)
        out["rouge1"] = rouge_scores["rouge1"].fmeasure
        out["rouge2"] = rouge_scores["rouge2"].fmeasure
        out["rougeL"] = rouge_scores["rougeL"].fmeasure
    except Exception:
        pass

    try:
        out["bleu"] = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=_SMOOTHIE)
    except Exception:
        pass

    try:
        out["meteor"] = meteor_score([ref_tokens], gen_tokens)
    except Exception:
        pass

    if compute_bertscore and _HAS_BERTSCORE and bert_score is not None:
        try:
            p, r, f1 = bert_score([generated_answer], [reference_answer], lang="en", verbose=False)
            out["bert_Precision"] = p.mean().item()
            out["bert_Recall"] = r.mean().item()
            out["bert_F1"] = f1.mean().item()
        except Exception:
            pass

    return out


# =========================
# Chroma retrieval
# =========================
def list_chroma_sources(db: Chroma) -> List[str]:
    try:
        payload = db.get(include=["metadatas"])
        metadatas = payload.get("metadatas", []) or []
        return sorted({m.get("source") for m in metadatas if isinstance(m, dict) and m.get("source")})
    except Exception:
        return []


def retrieve_context_unfiltered(*, db: Chroma, query: str, k: int) -> Tuple[str, List[Dict[str, Any]], float]:
    """
    Retrieve top-k chunks from the full shared Chroma database.

    No source/PDF/attack filtering is applied. Retrieval quality is part of the
    agent's performance.
    """
    t0 = time.perf_counter()
    try:
        docs_scores = db.similarity_search_with_score(query, k=k)
    except Exception:
        docs_scores = []
    retrieval_time = time.perf_counter() - t0

    retrieved: List[Dict[str, Any]] = []
    context_parts: List[str] = []

    for idx, (doc, score) in enumerate(docs_scores, start=1):
        metadata = dict(doc.metadata or {})
        source_name = metadata.get("source", "unknown")
        page = metadata.get("page", "unknown")
        chunk_id = metadata.get("id", "")
        content = (doc.page_content or "").strip()

        retrieved.append(
            {
                "rank": idx,
                "score": float(score) if isinstance(score, (int, float)) else score,
                "source": source_name,
                "source_basename": basename(str(source_name)),
                "page": page,
                "chunk_id": chunk_id,
                "chars": len(content),
            }
        )
        context_parts.append(
            f"[Retrieved Chunk {idx} | Source: {basename(str(source_name))} | Page: {page} | Score: {score}]\n{content}"
        )

    return "\n\n---\n\n".join(context_parts), retrieved, retrieval_time


# =========================
# JSON parsing and fallbacks
# =========================
def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from raw LLM output, tolerating code fences or extra text."""
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "y"}
    return default


def derive_dominant_attack_from_text(incident_summary: str) -> str:
    patterns = [
        r"Dominant Attack\s*:\s*([^\n]+)",
        r"dominant attack\s*[:=]\s*([^\n,]+)",
        r"Dominant attack\s+is\s+([^\n,.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, incident_summary, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return "Unknown"


# =========================
# Agent prompts
# =========================
def build_chunk_review_prompt(
    *,
    question: str,
    incident_summary: str,
    initial_query: str,
    retrieved_context: str,
) -> str:
    return f"""You are the retrieval-review component of IoT Shepherd.

The initial retrieval query was the administrator question from the QA file. Your task is to decide whether the retrieved mitigation-guide chunks are sufficient to answer that exact question for the dominant attack in the incident summary.

Rules:
1. Identify the dominant attack from the incident summary and focus on that attack. Ignore secondary attacks unless the question explicitly asks about them.
2. Treat the administrator question as the main intent. Do not change the question unless the retrieved chunks do not contain enough concrete guidance to answer it.
3. Mark sufficient=true only if the chunks provide concrete mitigation, containment, investigation, recovery, or hardening guidance that directly answers the question.
4. Do not mark chunks insufficient merely because they omit endpoint IPs, anomaly counts, percentages, or other incident-specific values; those can come from the incident summary.
5. If sufficient=false, write one revised retrieval query. Preserve the original question's intent, focus on the dominant attack, and add only the missing concept needed to find better guide chunks.
6. Do not include endpoint pairs, anomaly counts, percentages, secondary attacks, unrelated attack types, or broad generic phase lists in the revised query unless the question asks for them.

Return strict JSON only with exactly these keys:
{{
  "dominant_attack": "dominant attack name from the incident summary",
  "sufficient": true,
  "revised_query": "",
  "reason": "brief reason"
}}

Compact Incident Summary:
{incident_summary}

Administrator Question:
{question}

Initial Retrieval Query:
{initial_query}

Retrieved Mitigation-Guide Chunks:
{retrieved_context}

JSON:"""


def build_with_context_prompt(
    *,
    question: str,
    incident_summary: str,
    dominant_attack: str,
    retrieved_context: str,
) -> str:
    return f"""You are the IoT Shepherd end-to-end mitigation agent.

Task: answer the administrator's question using the compact incident summary and the selected mitigation-guide chunks.

Rules:
1. Answer for the dominant attack shown below. Ignore secondary attacks unless the question explicitly asks about them.
2. Treat the selected mitigation-guide chunks as the authoritative source for mitigation actions. Ignore chunks that are unrelated to the dominant attack or the question.
3. Use the incident summary only as operational evidence, such as the detected attack type, severity, anomaly counts, dominant attack percentage, and affected endpoint pairs.
4. When the retrieved guide chunks contain mitigation steps that directly answer the question, use that guide wording as closely as possible while inserting only the relevant incident evidence from the summary.
5. Do not invent CVEs, ports, vendors, tools, commands, identities, or device details that are not present in the incident summary or the retrieved chunks.
6. Keep the answer complete and focused, usually as a well-structured paragraph, and answer directly without extra commentary.

Dominant Attack for this Question:
{dominant_attack}

Compact Incident Summary:
{incident_summary}

Selected Mitigation-Guide Chunks:
{retrieved_context}

Administrator Question:
{question}

Answer:"""


def build_incident_only_prompt(question: str, incident_summary: str) -> str:
    return f"""You are the IoT Shepherd end-to-end mitigation agent.

Task: answer the administrator's question using the compact incident summary. No vector-store mitigation guide is available in this mode.

Rules:
1. Use the incident summary as operational evidence, including the detected attack type, severity, anomaly counts, dominant attack percentage, and affected endpoint pairs.
2. Provide practical mitigation guidance using general IoT incident-response knowledge.
3. Do not invent CVEs, vendors, commands, identities, or device details that are not present in the incident summary.
4. Keep the answer complete and focused, usually as a well-structured paragraph, and answer directly without extra commentary.

Compact Incident Summary:
{incident_summary}

Administrator Question:
{question}

Answer:"""


# =========================
# Agent step execution
# =========================
@dataclass
class ChunkReview:
    dominant_attack: str
    sufficient: bool
    revised_query: str
    reason: str
    raw_output: str
    parse_ok: bool
    fallback_used: bool
    time_s: float


@dataclass
class AgentRunResult:
    answer: str
    total_time_s: float
    initial_retrieval_time_s: float
    chunk_review_time_s: float
    retry_retrieval_time_s: float
    generation_time_s: float
    response_bytes: int
    response_token_count: int
    retrieval_attempt_count: int
    retry_used: bool
    review_sufficient: bool
    review_parse_ok: bool
    review_fallback_used: bool
    review_raw_output: str
    review_dominant_attack: str
    review_reason: str
    initial_retrieval_query: str
    retry_retrieval_query: str
    final_retrieval_query: str
    retrieved_chunk_count: int
    initial_retrieved_chunk_count: int
    retry_retrieved_chunk_count: int
    context_char_count: int
    context_token_count: int
    initial_retrieved_chunks: List[Dict[str, Any]]
    retry_retrieved_chunks: List[Dict[str, Any]]
    final_retrieved_chunks: List[Dict[str, Any]]


def run_chunk_review(
    *,
    model: Any,
    question: str,
    incident_summary: str,
    initial_query: str,
    retrieved_context: str,
) -> ChunkReview:
    prompt = build_chunk_review_prompt(
        question=question,
        incident_summary=incident_summary,
        initial_query=initial_query,
        retrieved_context=retrieved_context,
    )
    t0 = time.perf_counter()
    raw = model.invoke(prompt).strip()
    elapsed = time.perf_counter() - t0

    parsed = extract_json_object(raw)
    parse_ok = parsed is not None
    fallback_used = False

    dominant_fallback = derive_dominant_attack_from_text(incident_summary)
    if parsed is None:
        # If the review is not parseable, allow one bounded retry using the
        # original question plus the dominant attack. This protects answer
        # quality while preserving the one-retry limit.
        fallback_used = True
        parsed = {
            "dominant_attack": dominant_fallback,
            "sufficient": False,
            "revised_query": f"{dominant_fallback} mitigation: {question}".strip(),
            "reason": "Fallback requested one retry because review JSON could not be parsed.",
        }

    dominant_attack = str(parsed.get("dominant_attack", "")).strip() or dominant_fallback
    sufficient = parse_bool(parsed.get("sufficient", True), default=True)
    revised_query = str(parsed.get("revised_query", "")).strip()
    reason = str(parsed.get("reason", "")).strip()

    if not sufficient and not revised_query:
        fallback_used = True
        revised_query = f"{dominant_attack} mitigation: {question}".strip()
        if not reason:
            reason = "Fallback retry query preserves the original question and dominant attack."

    return ChunkReview(
        dominant_attack=dominant_attack,
        sufficient=sufficient,
        revised_query=revised_query,
        reason=reason,
        raw_output=raw,
        parse_ok=parse_ok,
        fallback_used=fallback_used,
        time_s=elapsed,
    )


def run_with_context_agent(
    *,
    model: Any,
    db: Chroma,
    question: str,
    incident_summary: str,
    top_k: int,
    max_retries: int,
    token_sizer: TokenSizer,
) -> AgentRunResult:
    """
    Run the bounded retrieval-repair agent.

    Initial retrieval uses the original QA question exactly. The LLM only gets
    one retry query if it judges the retrieved chunks insufficient.
    """
    t_total_start = time.perf_counter()

    initial_query = question.strip()
    initial_context, initial_chunks, initial_retrieval_time = retrieve_context_unfiltered(
        db=db,
        query=initial_query,
        k=top_k,
    )

    review = run_chunk_review(
        model=model,
        question=question,
        incident_summary=incident_summary,
        initial_query=initial_query,
        retrieved_context=initial_context,
    )

    retry_context = ""
    retry_chunks: List[Dict[str, Any]] = []
    retry_retrieval_time = 0.0
    retry_query = ""
    retry_used = False
    retrieval_attempt_count = 1

    final_context = initial_context
    final_chunks = initial_chunks
    final_query = initial_query

    if (not review.sufficient) and max_retries > 0:
        retry_used = True
        retry_query = review.revised_query
        retry_context, retry_chunks, retry_retrieval_time = retrieve_context_unfiltered(
            db=db,
            query=retry_query,
            k=top_k,
        )
        retrieval_attempt_count += 1
        # Use retry chunks as the selected context. If Chroma unexpectedly
        # returns no retry context, fall back to the initial chunks.
        if retry_context.strip():
            final_context = retry_context
            final_chunks = retry_chunks
            final_query = retry_query
        else:
            final_context = initial_context
            final_chunks = initial_chunks
            final_query = initial_query

    prompt = build_with_context_prompt(
        question=question,
        incident_summary=incident_summary,
        dominant_attack=review.dominant_attack,
        retrieved_context=final_context,
    )
    t_gen_start = time.perf_counter()
    answer = model.invoke(prompt).strip()
    generation_time = time.perf_counter() - t_gen_start

    total_time = time.perf_counter() - t_total_start

    # Token/byte counting is intentionally outside the timed inference window.
    response_bytes, response_tokens = token_sizer.size_and_tokens(answer)
    _, context_tokens = token_sizer.size_and_tokens(final_context)

    return AgentRunResult(
        answer=answer,
        total_time_s=total_time,
        initial_retrieval_time_s=initial_retrieval_time,
        chunk_review_time_s=review.time_s,
        retry_retrieval_time_s=retry_retrieval_time,
        generation_time_s=generation_time,
        response_bytes=response_bytes,
        response_token_count=response_tokens,
        retrieval_attempt_count=retrieval_attempt_count,
        retry_used=retry_used,
        review_sufficient=review.sufficient,
        review_parse_ok=review.parse_ok,
        review_fallback_used=review.fallback_used,
        review_raw_output=review.raw_output,
        review_dominant_attack=review.dominant_attack,
        review_reason=review.reason,
        initial_retrieval_query=initial_query,
        retry_retrieval_query=retry_query,
        final_retrieval_query=final_query,
        retrieved_chunk_count=len(final_chunks),
        initial_retrieved_chunk_count=len(initial_chunks),
        retry_retrieved_chunk_count=len(retry_chunks),
        context_char_count=len(final_context),
        context_token_count=context_tokens,
        initial_retrieved_chunks=initial_chunks,
        retry_retrieved_chunks=retry_chunks,
        final_retrieved_chunks=final_chunks,
    )


def run_incident_only(
    *,
    model: Any,
    question: str,
    incident_summary: str,
    token_sizer: TokenSizer,
) -> AgentRunResult:
    """Run the incident-only baseline. There is no planning, retrieval, review, or retry."""
    t_total_start = time.perf_counter()

    prompt = build_incident_only_prompt(question, incident_summary)
    t_gen_start = time.perf_counter()
    answer = model.invoke(prompt).strip()
    generation_time = time.perf_counter() - t_gen_start

    total_time = time.perf_counter() - t_total_start

    # Token/byte counting is intentionally outside the timed inference window.
    response_bytes, response_tokens = token_sizer.size_and_tokens(answer)

    return AgentRunResult(
        answer=answer,
        total_time_s=total_time,
        initial_retrieval_time_s=0.0,
        chunk_review_time_s=0.0,
        retry_retrieval_time_s=0.0,
        generation_time_s=generation_time,
        response_bytes=response_bytes,
        response_token_count=response_tokens,
        retrieval_attempt_count=0,
        retry_used=False,
        review_sufficient=True,
        review_parse_ok=True,
        review_fallback_used=False,
        review_raw_output="",
        review_dominant_attack="",
        review_reason="Incident-only mode does not perform retrieval review.",
        initial_retrieval_query="",
        retry_retrieval_query="",
        final_retrieval_query="",
        retrieved_chunk_count=0,
        initial_retrieved_chunk_count=0,
        retry_retrieved_chunk_count=0,
        context_char_count=0,
        context_token_count=0,
        initial_retrieved_chunks=[],
        retry_retrieved_chunks=[],
        final_retrieved_chunks=[],
    )


# =========================
# Summary helpers
# =========================
def scaled_mean(series: pd.Series) -> float:
    cleaned = series.dropna()
    if cleaned.empty:
        return float("nan")
    return float(cleaned.mean()) * SCORE_SCALE


def scaled_std(series: pd.Series) -> float:
    cleaned = series.dropna()
    if cleaned.empty:
        return float("nan")
    return float(cleaned.std()) * SCORE_SCALE


def raw_mean(series: pd.Series) -> float:
    cleaned = series.dropna()
    if cleaned.empty:
        return float("nan")
    return float(cleaned.mean())


def raw_std(series: pd.Series) -> float:
    cleaned = series.dropna()
    if cleaned.empty:
        return float("nan")
    return float(cleaned.std())


def build_quality_improvement_table(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for _, group in df.groupby(group_cols):
        row = {col: group.iloc[0][col] for col in group_cols}
        for metric in QUALITY_KEYS:
            wc_col = f"with_context_{metric}"
            io_col = f"incident_only_{metric}"
            wc_mean = scaled_mean(group[wc_col])
            io_mean = scaled_mean(group[io_col])
            diff = wc_mean - io_mean
            pct = (diff / io_mean * 100.0) if not math.isnan(io_mean) and io_mean != 0 else float("nan")
            row[f"{metric}_with_context_mean"] = wc_mean
            row[f"{metric}_incident_only_mean"] = io_mean
            row[f"{metric}_abs_diff"] = diff
            row[f"{metric}_pct_improvement"] = pct
        rows.append(row)
    return pd.DataFrame(rows)


def build_quality_means_table(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for _, group in df.groupby(group_cols):
        row = {col: group.iloc[0][col] for col in group_cols}
        for metric in QUALITY_KEYS:
            for prefix in ["with_context", "incident_only"]:
                col = f"{prefix}_{metric}"
                row[f"{prefix}_{metric}_mean_raw"] = raw_mean(group[col])
                row[f"{prefix}_{metric}_std_raw"] = raw_std(group[col])
                row[f"{prefix}_{metric}_mean"] = scaled_mean(group[col])
                row[f"{prefix}_{metric}_std"] = scaled_std(group[col])
        rows.append(row)
    return pd.DataFrame(rows)


def build_system_summary(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for _, group in df.groupby(group_cols):
        row = {col: group.iloc[0][col] for col in group_cols}
        for metric in SYSTEM_KEYS:
            for prefix in ["with_context", "incident_only"]:
                col = f"{prefix}_{metric}"
                if col in group.columns:
                    row[f"{prefix}_{metric}_mean"] = raw_mean(group[col])
                    row[f"{prefix}_{metric}_std"] = raw_std(group[col])

        wc_time = row.get("with_context_total_time_s_mean", float("nan"))
        io_time = row.get("incident_only_total_time_s_mean", float("nan"))
        row["with_context_reports_per_minute"] = 60.0 / wc_time if wc_time and not math.isnan(wc_time) else float("nan")
        row["incident_only_reports_per_minute"] = 60.0 / io_time if io_time and not math.isnan(io_time) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def save_generated_answers(run_dir: str, df_all: pd.DataFrame) -> None:
    base_dir = os.path.join(run_dir, "generated_answers")
    ensure_dir(base_dir)

    for (model, case_id), group in df_all.groupby(["model", "case_id"]):
        model_dir = os.path.join(base_dir, safe_tag(model))
        ensure_dir(model_dir)
        path = os.path.join(model_dir, f"{safe_tag(case_id)}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for _, row in group.sort_values("qa_index").iterrows():
                f.write("=" * 100 + "\n")
                f.write(f"Case: {row['case_id']}\n")
                f.write(f"Model: {row['model']}\n")
                f.write(f"QA Index: {row['qa_index']}\n")
                f.write(f"Review Dominant Attack: {row.get('with_context_review_dominant_attack', '')}\n")
                f.write(f"Retry Used: {row.get('with_context_retry_used', '')}\n")
                f.write(f"Initial Retrieval Query: {row.get('with_context_initial_retrieval_query', '')}\n")
                f.write(f"Retry Retrieval Query: {row.get('with_context_retry_retrieval_query', '')}\n")
                f.write(f"Final Retrieved Sources: {row.get('with_context_final_retrieved_sources', '')}\n\n")
                f.write(f"Question:\n{row['question']}\n\n")
                f.write(f"Reference Answer:\n{row['reference_answer']}\n\n")
                f.write(f"With Context Answer:\n{row['with_context_answer']}\n\n")
                f.write(f"Incident Only Answer:\n{row['incident_only_answer']}\n\n")


# =========================
# CLI and main execution
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IoT Shepherd end-to-end bounded retrieval-repair evaluation.")
    parser.add_argument("--chroma-path", default="chroma", help="Path to populated Chroma vector database.")
    parser.add_argument("--eval-root", default="Shepherd_Eval", help="Root folder for QA, incidents, and runs.")
    parser.add_argument("--qa-dir", default=None, help="Directory containing case QA DOCX files. Default: <eval-root>/qa")
    parser.add_argument(
        "--incident-dir",
        default=None,
        help="Directory containing compact incident summary TXT files. Default: <eval-root>/incident_summaries",
    )
    parser.add_argument("--run-dir", default=None, help="Optional explicit output run directory.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Ollama model names to evaluate.")
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES, help="Case IDs used only for file loading.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks per retrieval attempt.")
    parser.add_argument("--max-retries", type=int, default=1, help="Maximum retry attempts after initial retrieval. Default: 1.")
    parser.add_argument("--tokenizer", default="whitespace", help="Tokenizer for token-count estimates. Default: whitespace.")
    parser.add_argument("--skip-bertscore", action="store_true", help="Skip expensive BERTScore computation.")
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable the single unmeasured warm-up call performed once per model before timed evaluation.",
    )
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace, qa_dir: str, incident_dir: str) -> None:
    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if not os.path.isdir(args.chroma_path):
        raise FileNotFoundError(
            f"Chroma directory not found at '{args.chroma_path}'. Run populate_database.py before evaluation."
        )
    if not os.path.isdir(qa_dir):
        raise FileNotFoundError(f"QA directory not found: {qa_dir}")
    if not os.path.isdir(incident_dir):
        raise FileNotFoundError(f"Incident summary directory not found: {incident_dir}")

    missing = []
    for case_id in args.cases:
        qa_file = os.path.join(qa_dir, f"{case_id}.docx")
        incident_file = os.path.join(incident_dir, f"{case_id}_incident_summary.txt")
        if not os.path.exists(qa_file):
            missing.append(qa_file)
        if not os.path.exists(incident_file):
            missing.append(incident_file)
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))


def main() -> None:
    args = parse_args()
    qa_dir = args.qa_dir or os.path.join(args.eval_root, "qa")
    incident_dir = args.incident_dir or os.path.join(args.eval_root, "incident_summaries")
    validate_inputs(args, qa_dir, incident_dir)

    compute_bertscore = (not args.skip_bertscore) and _HAS_BERTSCORE

    run_dir = args.run_dir or os.path.join(args.eval_root, "runs", now_tag())
    ensure_dir(run_dir)

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=args.chroma_path, embedding_function=embedding_function)
    chroma_sources = list_chroma_sources(db)
    token_sizer = TokenSizer(tokenizer_name=args.tokenizer)

    config = {
        "models": args.models,
        "cases": args.cases,
        "case_id_definition": "File organization label only; not supplied to the LLM as the attack type.",
        "chroma_path": args.chroma_path,
        "eval_root": args.eval_root,
        "qa_dir": qa_dir,
        "incident_dir": incident_dir,
        "top_k": args.top_k,
        "max_retries": args.max_retries,
        "retrieval_policy": "Initial retrieval uses the exact QA question. If retrieved chunks are insufficient, the LLM may rewrite the query once. Retrieval is unfiltered over the full shared Chroma database.",
        "incident_only_policy": "No planning, no retrieval, no review, and no retry; direct LLM answer from incident summary and question.",
        "resource_metrics": "Not measured. This script records timing only, plus response size/token counts outside timing.",
        "score_scale_for_summaries": SCORE_SCALE,
        "bertscore_requested": not args.skip_bertscore,
        "bertscore_available": _HAS_BERTSCORE,
        "bertscore_computed": compute_bertscore,
        "tokenizer": args.tokenizer,
        "warmup_enabled": not args.no_warmup,
        "warmup_policy": "One short unmeasured Ollama call per model before timed evaluation; not included in quality or latency metrics.",
        "chroma_sources": chroma_sources,
        "timing_policy": {
            "total_time_s": "Full per-answer latency excluding quality scoring, token counting, CSV writing, and summary aggregation.",
            "initial_retrieval_time_s": "First Chroma retrieval using the exact administrator question; zero for incident-only.",
            "chunk_review_time_s": "LLM call that judges chunk sufficiency and optionally writes one retry query; zero for incident-only.",
            "retry_retrieval_time_s": "Second Chroma retrieval time only when the reviewer requests a retry.",
            "generation_time_s": "Final LLM answer-generation call.",
        },
        "started_at": now_tag(),
    }
    write_json(os.path.join(run_dir, "config.json"), config)

    all_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    warmup_rows: List[Dict[str, Any]] = []

    for model_name in args.models:
        print(f"\n=== Model: {model_name} ===", flush=True)
        model = Ollama(model=model_name)
        if not args.no_warmup:
            print(f"  Warm-up: {model_name}", flush=True)
            warmup_rows.append(run_unmeasured_warmup(model, model_name))

        model_dir = os.path.join(run_dir, "per_question", safe_tag(model_name))
        ensure_dir(model_dir)

        for case_id in args.cases:
            print(f"--- Case: {case_id} ---", flush=True)
            qa_file = os.path.join(qa_dir, f"{case_id}.docx")
            incident_file = os.path.join(incident_dir, f"{case_id}_incident_summary.txt")
            incident_summary = read_text(incident_file)
            qa_pairs = load_qa_pairs_docx(qa_file)

            if not qa_pairs:
                error_rows.append(
                    {
                        "model": model_name,
                        "case_id": case_id,
                        "qa_file": qa_file,
                        "error": "No QA pairs parsed from DOCX.",
                    }
                )
                continue

            for qa_index, (question, reference_answer) in enumerate(qa_pairs, start=1):
                print(f"  QA {qa_index}/{len(qa_pairs)}", flush=True)
                row: Dict[str, Any] = {
                    "model": model_name,
                    "case_id": case_id,
                    "qa_file": qa_file,
                    "incident_file": incident_file,
                    "qa_index": qa_index,
                    "question": question,
                    "reference_answer": reference_answer,
                }

                try:
                    with_context = run_with_context_agent(
                        model=model,
                        db=db,
                        question=question,
                        incident_summary=incident_summary,
                        top_k=args.top_k,
                        max_retries=args.max_retries,
                        token_sizer=token_sizer,
                    )

                    incident_only = run_incident_only(
                        model=model,
                        question=question,
                        incident_summary=incident_summary,
                        token_sizer=token_sizer,
                    )

                    wc_scores = evaluate_answer(
                        with_context.answer,
                        reference_answer,
                        compute_bertscore=compute_bertscore,
                    )
                    io_scores = evaluate_answer(
                        incident_only.answer,
                        reference_answer,
                        compute_bertscore=compute_bertscore,
                    )

                    row["with_context_answer"] = with_context.answer
                    row["incident_only_answer"] = incident_only.answer

                    row["with_context_initial_retrieval_query"] = with_context.initial_retrieval_query
                    row["with_context_retry_retrieval_query"] = with_context.retry_retrieval_query
                    row["with_context_final_retrieval_query"] = with_context.final_retrieval_query
                    row["with_context_review_raw_output"] = with_context.review_raw_output
                    row["with_context_review_dominant_attack"] = with_context.review_dominant_attack
                    row["with_context_review_reason"] = with_context.review_reason
                    row["with_context_initial_retrieved_chunks_json"] = json.dumps(with_context.initial_retrieved_chunks)
                    row["with_context_retry_retrieved_chunks_json"] = json.dumps(with_context.retry_retrieved_chunks)
                    row["with_context_final_retrieved_chunks_json"] = json.dumps(with_context.final_retrieved_chunks)
                    row["with_context_final_retrieved_sources"] = "; ".join(
                        [str(c.get("source_basename", "")) for c in with_context.final_retrieved_chunks]
                    )

                    # Incident-only has no retrieval/review data, but keep empty columns for consistency.
                    row["incident_only_initial_retrieval_query"] = ""
                    row["incident_only_retry_retrieval_query"] = ""
                    row["incident_only_final_retrieval_query"] = ""
                    row["incident_only_final_retrieved_sources"] = ""

                    for key, value in wc_scores.items():
                        row[f"with_context_{key}"] = value
                    for key, value in io_scores.items():
                        row[f"incident_only_{key}"] = value

                    for key, value in asdict(with_context).items():
                        if key in {
                            "answer",
                            "review_raw_output",
                            "initial_retrieved_chunks",
                            "retry_retrieved_chunks",
                            "final_retrieved_chunks",
                        }:
                            continue
                        row[f"with_context_{key}"] = value

                    for key, value in asdict(incident_only).items():
                        if key in {
                            "answer",
                            "review_raw_output",
                            "initial_retrieved_chunks",
                            "retry_retrieved_chunks",
                            "final_retrieved_chunks",
                        }:
                            continue
                        row[f"incident_only_{key}"] = value

                    all_rows.append(row)

                except Exception as exc:
                    error_rows.append(
                        {
                            "model": model_name,
                            "case_id": case_id,
                            "qa_file": qa_file,
                            "qa_index": qa_index,
                            "question": question,
                            "error": repr(exc),
                        }
                    )

            # Save per-model/per-case partial CSV for resilience.
            partial = pd.DataFrame([r for r in all_rows if r["model"] == model_name and r["case_id"] == case_id])
            partial.to_csv(os.path.join(model_dir, f"{safe_tag(case_id)}.csv"), index=False)

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(os.path.join(run_dir, "all_per_question.csv"), index=False)

    if error_rows:
        pd.DataFrame(error_rows).to_csv(os.path.join(run_dir, "errors.csv"), index=False)

    if warmup_rows:
        pd.DataFrame(warmup_rows).to_csv(os.path.join(run_dir, "warmup_log.csv"), index=False)

    if not df_all.empty:
        build_quality_improvement_table(df_all, ["model", "case_id"]).to_csv(
            os.path.join(run_dir, "summary_quality_improvement_by_model_case.csv"), index=False
        )
        build_quality_improvement_table(df_all, ["model"]).to_csv(
            os.path.join(run_dir, "summary_quality_improvement_by_model.csv"), index=False
        )
        build_quality_improvement_table(df_all, ["case_id"]).to_csv(
            os.path.join(run_dir, "summary_quality_improvement_by_case.csv"), index=False
        )
        build_quality_means_table(df_all, ["model", "case_id"]).to_csv(
            os.path.join(run_dir, "summary_quality_means_by_model_case.csv"), index=False
        )
        build_system_summary(df_all, ["model", "case_id"]).to_csv(
            os.path.join(run_dir, "summary_system_by_model_case.csv"), index=False
        )
        build_system_summary(df_all, ["model"]).to_csv(
            os.path.join(run_dir, "summary_system_by_model.csv"), index=False
        )
        build_system_summary(df_all, ["case_id"]).to_csv(
            os.path.join(run_dir, "summary_system_by_case.csv"), index=False
        )
        save_generated_answers(run_dir, df_all)

    print(f"\nEvaluation complete. Outputs saved under: {run_dir}", flush=True)


if __name__ == "__main__":
    main()

    
    """
nohup python performance_shepherd_end_to_end_harness.py &
"""