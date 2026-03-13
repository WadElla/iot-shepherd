from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

from .kb import COLLECTION_NAME, _open_chroma
from ..config import get_active_chroma_dir, USE_CHROMA_SESSIONS


def _collection_count(db: Any) -> Optional[int]:
    """Best-effort collection count across Chroma/LangChain wrappers."""
    try:
        col = getattr(db, "_collection", None)
        if col is not None and hasattr(col, "count"):
            return int(col.count())
    except Exception:
        pass
    try:
        data = db.get(include=[])
        if isinstance(data, dict) and "ids" in data:
            return int(len(data.get("ids") or []))
    except Exception:
        pass
    return None


def retrieve_context(
    query: str,
    chroma_dir: Path,
    embed_model: str = "embeddinggemma:latest",
    k: int = 5,
) -> Dict[str, Any]:
    """Retrieve top-k manual chunks from Chroma, robustly.

    Errors:
      - EMPTY_QUERY: query is blank
      - KB_EMPTY: KB has 0 chunks indexed
      - NO_MATCHES: KB has chunks but none matched
    """
    q = (query or "").strip()
    kk = int(k) if k else 5
    if kk < 1:
        kk = 1

    if not q:
        return {"type": "kb_context", "question": query, "k": kk, "collection": COLLECTION_NAME,
                "results": [], "ok": False, "error": "EMPTY_QUERY", "total_chunks": 0}

    try:
        db = _open_chroma(get_active_chroma_dir() if USE_CHROMA_SESSIONS else chroma_dir, embed_model)
    except Exception as e:
        msg = str(e)
        if "CHROMA_SCHEMA_MISMATCH" in msg or "default_tenant" in msg:
            return {
                "type": "kb_context",
                "question": q,
                "k": kk,
                "collection": COLLECTION_NAME,
                "results": [],
                "ok": False,
                "error": "CHROMA_SCHEMA_MISMATCH",
                "total_chunks": 0,
            }
        return {
            "type": "kb_context",
            "question": q,
            "k": kk,
            "collection": COLLECTION_NAME,
            "results": [],
            "ok": False,
            "error": "CHROMA_OPEN_FAILED",
            "total_chunks": 0,
        }
    total = _collection_count(db)

    if total is not None and total == 0:
        return {"type": "kb_context", "question": q, "k": kk, "collection": COLLECTION_NAME,
                "results": [], "ok": False, "error": "KB_EMPTY", "total_chunks": 0}

    results = db.similarity_search_with_score(q, k=kk)

    items: List[Dict[str, Any]] = []
    for doc, score in results:
        meta = dict(getattr(doc, "metadata", None) or {})
        text = getattr(doc, "page_content", "") or ""
        items.append({
            "id": meta.get("id"),
            "source": (Path(str(meta.get("source") or "")) .name or meta.get("source")),
            "page": meta.get("page"),
            "score": float(score),
            "text": text,
            "excerpt": (text[:280] + "…") if len(text) > 280 else text,
        })

    if total is None:
        total = 0 if not items else max(len(items), kk)

    if not items:
        return {"type": "kb_context", "question": q, "k": kk, "collection": COLLECTION_NAME,
                "results": [], "ok": False, "error": "NO_MATCHES", "total_chunks": int(total)}

    return {"type": "kb_context", "question": q, "k": kk, "collection": COLLECTION_NAME,
            "results": items, "ok": True, "error": None, "total_chunks": int(total)}
