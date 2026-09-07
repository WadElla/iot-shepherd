#!/usr/bin/env python3
"""
IoT Shepherd end-to-end incident-to-mitigation evaluation.

This script evaluates a bounded agentic retrieval workflow for three IoT attack
cases. It is designed to be run as a standalone batch experiment, not through
Streamlit.

Two modes are evaluated for every model, case, and QA pair:

1) with_context
   - LLM reads the compact incident summary and administrator question.
   - LLM identifies the dominant attack and builds the first retrieval query.
   - The first query should preserve the original question unless it is ambiguous;
     the dominant attack can be added as grounding context.
   - Python executes unfiltered Chroma retrieval over the full shared vector DB.
   - LLM reviews whether the retrieved chunks are sufficient.
   - If insufficient, LLM gets one retry to produce a minimally revised query.
   - Python retrieves once more using the revised query.
   - LLM generates the final answer from the incident summary and selected chunks.

2) incident_only
   - No planning and no retrieval.
   - LLM answers directly from the compact incident summary and question.

Important design choices:
- Python only handles file loading, Chroma tool execution, LLM calls, timing,
  scoring, and logging.
- Python does not filter retrieval by attack type or PDF source.
- Python does not rewrite the LLM retrieval query.
- The agent is bounded: initial retrieval plus at most one retry.
- Only timing metrics are measured. CPU/RAM/GPU resource metrics are not sampled.
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
  python performance_shepherd_end_to_end_agentic_v2.py

Quick test:
  python performance_shepherd_end_to_end_agentic_v2.py --skip-bertscore --models gemma2:9b --cases Backdoor
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
# Token/size helper
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
    "llama3.1:8b",
    "llava:7b",
    "gemma2:9b",
    "mistral:7b",
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
    "query_planning_time_s",
    "initial_retrieval_time_s",
    "chunk_review_time_s",
    "retry_retrieval_time_s",
    "generation_time_s",
]

SYSTEM_KEYS = TIME_KEYS + [
    "retrieval_attempt_count",
    "retry_used",
    "review_parse_ok",
    "planner_parse_ok",
    "planner_fallback_used",
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

def run_unmeasured_warmup(model: Any, model_name: str) -> Dict[str, Any]:
    """Run one short unmeasured Ollama call before timed evaluation for a model.

    The warm-up is not included in timing, quality metrics, or answer outputs.
    It only reduces first-call/model-loading bias in the measured questions.
    """
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
    return re.sub(r"\s+", " ", text).strip()


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
    """
    Response/context size helper.

    Default is whitespace counting to avoid HF model-length warnings. Use
    --tokenizer gpt2 only if exact HF-style token estimates are needed.
    """

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
    """Compute quality metrics outside the timed inference window."""
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
# Chroma diagnostics and retrieval
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
    agent's performance because the retrieval query comes from the LLM.
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
def build_query_planner_prompt(question: str, incident_summary: str) -> str:
    return f"""You are the retrieval-planning component of IoT Shepherd.

Read the compact incident summary and the administrator question. Identify the dominant attack from the incident summary. Focus only on the dominant attack unless the question explicitly asks about another listed attack.

Your job is to create the first retrieval query for the mitigation-guide database. Preserve the administrator question as the main retrieval intent. If the question is clear, do not rephrase it. In that case, make the retrieval query by combining the dominant attack with the original administrator question. If the question is ambiguous, make the smallest possible clarification using the dominant attack and the question's intent. Do not include endpoint pairs, anomaly counts, percentages, secondary attacks, or broad generic phase lists unless the question asks for them.

Examples:
- If the dominant attack is "Backdoor" and the question is "What immediate containment steps should the administrator take?", the retrieval query should be: "Backdoor mitigation: What immediate containment steps should the administrator take?"
- If the dominant attack is "Ransomware" and the question is "How should recovery be handled?", the retrieval query should be: "Ransomware mitigation: How should recovery be handled?"
- If the dominant attack is "DDoS_HTTP" and the question is ambiguous, such as "What should be done first?", the retrieval query may clarify minimally: "DDoS_HTTP mitigation: first response actions for service availability protection"

Return strict JSON only with exactly these keys:
{{
  "dominant_attack": "dominant attack name from the incident summary",
  "question_ambiguous": false,
  "retrieval_query": "dominant attack mitigation: original question, unless the question is ambiguous",
  "reason": "brief reason for the query"
}}

Compact Incident Summary:
{incident_summary}

Administrator Question:
{question}

JSON:"""


def build_chunk_review_prompt(
    *,
    question: str,
    incident_summary: str,
    dominant_attack: str,
    retrieval_query: str,
    retrieved_context: str,
) -> str:
    return f"""You are the retrieval-review component of IoT Shepherd.

Decide whether the retrieved mitigation-guide chunks contain enough concrete guidance to answer the administrator question for the dominant attack. Focus only on the dominant attack unless the question explicitly asks about another listed attack.

Mark sufficient as true only when the chunks include mitigation, investigation, containment, recovery, or hardening actions that directly answer this specific question. Do not mark the chunks sufficient if they mainly describe the attack, discuss another attack, or contain only generic background. If the chunks are insufficient, provide one revised retrieval query that preserves the original question, includes the dominant attack, and adds only the missing concept needed for this question. Do not add secondary attacks, endpoint pairs, anomaly counts, or unrelated response phases.

Return strict JSON only with exactly these keys:
{{
  "sufficient": true,
  "revised_query": "",
  "reason": "brief reason"
}}

Dominant Attack:
{dominant_attack}

Compact Incident Summary:
{incident_summary}

Administrator Question:
{question}

Initial Retrieval Query:
{retrieval_query}

Retrieved Mitigation-Guide Chunks:
{retrieved_context}

JSON:"""


def build_with_context_prompt(
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
class QueryPlan:
    dominant_attack: str
    question_ambiguous: bool
    retrieval_query: str
    reason: str
    raw_output: str
    parse_ok: bool
    fallback_used: bool
    time_s: float


@dataclass
class ChunkReview:
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
    query_planning_time_s: float
    initial_retrieval_time_s: float
    chunk_review_time_s: float
    retry_retrieval_time_s: float
    generation_time_s: float
    response_bytes: int
    response_token_count: int
    retrieval_attempt_count: int
    retry_used: bool
    review_sufficient: bool
    planner_parse_ok: bool
    planner_fallback_used: bool
    review_parse_ok: bool
    review_fallback_used: bool
    planner_raw_output: str
    review_raw_output: str
    planner_dominant_attack: str
    question_ambiguous: bool
    planner_reason: str
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


def run_query_planner(*, model: Any, question: str, incident_summary: str) -> QueryPlan:
    prompt = build_query_planner_prompt(question, incident_summary)
    t0 = time.perf_counter()
    raw = model.invoke(prompt).strip()
    elapsed = time.perf_counter() - t0

    parsed = extract_json_object(raw)
    parse_ok = parsed is not None
    fallback_used = False

    if parsed is None:
        fallback_used = True
        dominant = derive_dominant_attack_from_text(incident_summary)
        parsed = {
            "dominant_attack": dominant,
            "question_ambiguous": False,
            "retrieval_query": f"{dominant} {question}".strip(),
            "reason": "Fallback query uses the original question with dominant-attack grounding.",
        }

    dominant_attack = str(parsed.get("dominant_attack", "")).strip() or derive_dominant_attack_from_text(incident_summary)
    question_ambiguous = parse_bool(parsed.get("question_ambiguous", False), default=False)
    raw_retrieval_query = str(parsed.get("retrieval_query", "")).strip()
    reason = str(parsed.get("reason", "")).strip()

    # Enforce the agreed retrieval policy: if the question is clear, the first
    # retrieval uses the original question grounded by the dominant attack. The
    # LLM can only rephrase the first query when it marks the question ambiguous.
    if question_ambiguous and raw_retrieval_query:
        retrieval_query = raw_retrieval_query
    else:
        retrieval_query = f"{dominant_attack} {question}".strip()
        if raw_retrieval_query and raw_retrieval_query != retrieval_query:
            reason = (reason + " ").strip() + "Question was treated as clear, so the original wording was preserved with dominant-attack grounding."

    if not retrieval_query:
        fallback_used = True
        retrieval_query = question.strip()
        if not reason:
            reason = "Fallback query uses the original question."

    return QueryPlan(
        dominant_attack=dominant_attack,
        question_ambiguous=question_ambiguous,
        retrieval_query=retrieval_query,
        reason=reason,
        raw_output=raw,
        parse_ok=parse_ok,
        fallback_used=fallback_used,
        time_s=elapsed,
    )


def run_chunk_review(
    *,
    model: Any,
    question: str,
    incident_summary: str,
    dominant_attack: str,
    retrieval_query: str,
    retrieved_context: str,
) -> ChunkReview:
    prompt = build_chunk_review_prompt(
        question=question,
        incident_summary=incident_summary,
        dominant_attack=dominant_attack,
        retrieval_query=retrieval_query,
        retrieved_context=retrieved_context,
    )
    t0 = time.perf_counter()
    raw = model.invoke(prompt).strip()
    elapsed = time.perf_counter() - t0

    parsed = extract_json_object(raw)
    parse_ok = parsed is not None
    fallback_used = False

    if parsed is None:
        # If review parsing fails, prefer one bounded retry rather than silently
        # accepting weak chunks. This protects answer quality while preserving the
        # one-retry limit.
        fallback_used = True
        parsed = {
            "sufficient": False,
            "revised_query": f"{dominant_attack} {question} mitigation guide",
            "reason": "Fallback requested one retry because the review JSON could not be parsed.",
        }

    sufficient = parse_bool(parsed.get("sufficient", True), default=True)
    revised_query = str(parsed.get("revised_query", "")).strip()
    reason = str(parsed.get("reason", "")).strip()

    if not sufficient and not revised_query:
        # Keep the workflow functional while preserving the original intent.
        fallback_used = True
        revised_query = f"{dominant_attack} {question} mitigation guide".strip()
        if not reason:
            reason = "Fallback retry query preserves the original question and dominant attack."

    return ChunkReview(
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
    Run the bounded with-context agent.

    The agent performs one LLM-planned retrieval, reviews the chunks, and retries
    at most max_retries times. The intended experiment uses max_retries=1.
    """
    t_total_start = time.perf_counter()

    plan = run_query_planner(model=model, question=question, incident_summary=incident_summary)

    initial_context, initial_chunks, initial_retrieval_time = retrieve_context_unfiltered(
        db=db,
        query=plan.retrieval_query,
        k=top_k,
    )

    review = run_chunk_review(
        model=model,
        question=question,
        incident_summary=incident_summary,
        dominant_attack=plan.dominant_attack,
        retrieval_query=plan.retrieval_query,
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
    final_query = plan.retrieval_query

    if (not review.sufficient) and max_retries > 0:
        retry_used = True
        retry_query = review.revised_query
        retry_context, retry_chunks, retry_retrieval_time = retrieve_context_unfiltered(
            db=db,
            query=retry_query,
            k=top_k,
        )
        retrieval_attempt_count += 1
        # Keep the retry chunks first, but retain the initial chunks as secondary
        # evidence. This avoids throwing away useful initial evidence when the
        # retry only partially improves retrieval.
        final_context = (
            "Retry Retrieval Chunks:\n" + retry_context +
            "\n\n---\n\nInitial Retrieval Chunks for Additional Context:\n" + initial_context
        ).strip()
        seen = set()
        combined_chunks = []
        for ch in retry_chunks + initial_chunks:
            key = (ch.get("source"), ch.get("page"), ch.get("chunk_id"))
            if key not in seen:
                seen.add(key)
                combined_chunks.append(ch)
        final_chunks = combined_chunks
        final_query = retry_query

    prompt = build_with_context_prompt(
        question=question,
        incident_summary=incident_summary,
        dominant_attack=plan.dominant_attack,
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
        query_planning_time_s=plan.time_s,
        initial_retrieval_time_s=initial_retrieval_time,
        chunk_review_time_s=review.time_s,
        retry_retrieval_time_s=retry_retrieval_time,
        generation_time_s=generation_time,
        response_bytes=response_bytes,
        response_token_count=response_tokens,
        retrieval_attempt_count=retrieval_attempt_count,
        retry_used=retry_used,
        review_sufficient=review.sufficient,
        planner_parse_ok=plan.parse_ok,
        planner_fallback_used=plan.fallback_used,
        review_parse_ok=review.parse_ok,
        review_fallback_used=review.fallback_used,
        planner_raw_output=plan.raw_output,
        review_raw_output=review.raw_output,
        planner_dominant_attack=plan.dominant_attack,
        question_ambiguous=plan.question_ambiguous,
        planner_reason=plan.reason,
        review_reason=review.reason,
        initial_retrieval_query=plan.retrieval_query,
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
    """Run the incident-only baseline. There is no planning and no retrieval."""
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
        query_planning_time_s=0.0,
        initial_retrieval_time_s=0.0,
        chunk_review_time_s=0.0,
        retry_retrieval_time_s=0.0,
        generation_time_s=generation_time,
        response_bytes=response_bytes,
        response_token_count=response_tokens,
        retrieval_attempt_count=0,
        retry_used=False,
        review_sufficient=True,
        planner_parse_ok=True,
        planner_fallback_used=False,
        review_parse_ok=True,
        review_fallback_used=False,
        planner_raw_output="",
        review_raw_output="",
        planner_dominant_attack="",
        question_ambiguous=False,
        planner_reason="",
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
                f.write(f"Planner Dominant Attack: {row.get('with_context_planner_dominant_attack', '')}\n")
                f.write(f"Retry Used: {row.get('with_context_retry_used', '')}\n")
                f.write(f"Initial Retrieval Query: {row.get('with_context_initial_retrieval_query', '')}\n")
                f.write(f"Retry Retrieval Query: {row.get('with_context_retry_retrieval_query', '')}\n\n")
                f.write(f"Question:\n{row['question']}\n\n")
                f.write(f"Reference Answer:\n{row['reference_answer']}\n\n")
                f.write(f"With Context Answer:\n{row['with_context_answer']}\n\n")
                f.write(f"Incident Only Answer:\n{row['incident_only_answer']}\n\n")


# =========================
# CLI and main execution
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IoT Shepherd end-to-end agentic incident-to-mitigation evaluation.")
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
    parser.add_argument("--max-retries", type=int, default=1, help="Maximum query-rewrite retries after the initial retrieval. Default: 1.")
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
        "retrieval_policy": "Unfiltered Chroma retrieval over the full shared database. The LLM identifies the dominant attack and ambiguity. Clear questions use the original question grounded by the dominant attack; rephrasing is allowed only when the planner marks the question ambiguous. One bounded retry is allowed if retrieved chunks are insufficient.",
        "incident_only_policy": "No planning and no retrieval; direct LLM answer from incident summary and question.",
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
            "total_time_s": "Full per-answer agent latency excluding quality scoring, token counting, CSV writing, and summary aggregation.",
            "query_planning_time_s": "LLM call that creates the first retrieval query for with-context mode; zero for incident-only.",
            "initial_retrieval_time_s": "First Chroma retrieval time for with-context mode; zero for incident-only.",
            "chunk_review_time_s": "LLM call that judges retrieved chunk sufficiency and optionally writes one revised query; zero for incident-only.",
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
                    row["with_context_planner_raw_output"] = with_context.planner_raw_output
                    row["with_context_review_raw_output"] = with_context.review_raw_output
                    row["with_context_planner_dominant_attack"] = with_context.planner_dominant_attack
                    row["with_context_question_ambiguous"] = with_context.question_ambiguous
                    row["with_context_planner_reason"] = with_context.planner_reason
                    row["with_context_review_reason"] = with_context.review_reason
                    row["with_context_initial_retrieved_chunks_json"] = json.dumps(with_context.initial_retrieved_chunks)
                    row["with_context_retry_retrieved_chunks_json"] = json.dumps(with_context.retry_retrieved_chunks)
                    row["with_context_final_retrieved_chunks_json"] = json.dumps(with_context.final_retrieved_chunks)
                    row["with_context_final_retrieved_sources"] = "; ".join(
                        [str(c.get("source_basename", "")) for c in with_context.final_retrieved_chunks]
                    )

                    # Incident-only has no retrieval/planner data, but keep empty columns for consistency.
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
                            "planner_raw_output",
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
                            "planner_raw_output",
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
nohup python performance_shepherd_end_to_end_agentic_v2.py &

"""