from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from .retrieval import retrieve_context


CGM_SYSTEM = """You are the IoT Shepherd CGM (Context-Aware Generation Module).
You answer the administrator's question using ONLY the provided IoT manual excerpts.

Rules:
- Do not invent device-specific steps that are not supported by the excerpts.
- If the manuals do not contain the answer, say so.
- When you rely on an excerpt, cite it as [KB:<id>] where <id> is the chunk id provided.
- Keep the answer operator-facing and practical (steps/checks/config pointers).
"""


def answer_from_manuals(
    question: str,
    chroma_dir: Path,
    embed_model: str,
    llm_model: str,
    ollama_host: str,
    k: int = 5,
) -> Dict[str, Any]:
    payload = retrieve_context(
        query=question,
        chroma_dir=chroma_dir,
        embed_model=embed_model,
        k=k,
    )

    ok = bool(payload.get("ok", False))
    err = payload.get("error")
    total = int(payload.get("total_chunks", 0) or 0)

    # If KB is empty or retrieval failed, do NOT call the LLM.
    if not ok:
        if err == "KB_EMPTY":
            return {
                "answer": (
                    "⚠️ **No manuals are indexed right now (0 chunks).**\n\n"
                    "Upload PDFs in **Knowledge Base (CGM)** and run indexing, then ask your question again."
                ),
                "evidence": [],
                "kb_total_chunks": 0,
                "error": "KB_EMPTY",
            }
        if err == "NO_MATCHES":
            return {
                "answer": (
                    "I could not find relevant manual excerpts for that question.\n\n"
                    f"**KB status:** {total} chunk(s) are indexed, but none matched your query.\n"
                    "Try rephrasing with device/model keywords, menu names, error codes, or protocol terms."
                ),
                "evidence": [],
                "kb_total_chunks": total,
                "error": "NO_MATCHES",
            }
        if err == "EMPTY_QUERY":
            return {
                "answer": "⚠️ Empty question. Please type a question so I can retrieve relevant manual evidence.",
                "evidence": [],
                "kb_total_chunks": total,
                "error": "EMPTY_QUERY",
            }
        if err == "CHROMA_SCHEMA_MISMATCH":
            return {
                "answer": "⚠️ Manuals index is in an incompatible ChromaDB format. Please reset the Chroma index (Knowledge Base → Reset Chroma index) and re-index manuals.",
                "evidence": [],
                "kb_total_chunks": 0,
                "error": "CHROMA_SCHEMA_MISMATCH",
            }
        return {
            "answer": f"⚠️ Knowledge Base retrieval error: {err}",
            "evidence": [],
            "kb_total_chunks": total,
            "error": err or "UNKNOWN",
        }

    results: List[dict] = payload.get("results", []) or []
    evidence = []
    context_blocks = []
    for r in results:
        cid = r.get("id") or "unknown"
        excerpt = (r.get("excerpt") or "").strip()
        evidence.append({
            "id": cid,
            "source": r.get("source"),
            "page": r.get("page"),
            "score": r.get("score"),
            "excerpt": excerpt,
        })
        context_blocks.append(f"[KB:{cid}]\n{excerpt}")

    context = "\n\n".join(context_blocks).strip()
    if not context:
        return {
            "answer": "⚠️ No manual excerpts were returned. Please try re-indexing the manuals.",
            "evidence": [],
            "kb_total_chunks": total,
            "error": "NO_CONTEXT",
        }

    llm = ChatOllama(model=llm_model, base_url=ollama_host, temperature=0.2)

    user_prompt = f"""Question:
{question}

Manual excerpts:
{context}

Write the best possible answer grounded in the excerpts."""

    resp = llm.invoke([SystemMessage(content=CGM_SYSTEM), HumanMessage(content=user_prompt)])
    answer = getattr(resp, "content", None) or str(resp)

    # Guarantee citations if evidence exists
    if evidence and "[KB:" not in answer:
        ids = [e.get("id") for e in evidence if e.get("id")]
        if ids:
            answer = answer.strip() + "\n\nSources: " + ", ".join([f"[KB:{cid}]" for cid in ids[:5]])

    return {"answer": answer, "evidence": evidence, "kb_total_chunks": total, "error": None}
