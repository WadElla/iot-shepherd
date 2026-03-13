from __future__ import annotations

import json

from pathlib import Path
from typing import Dict, Any, List
import shutil
import os
import warnings
import gc
import time

from ..config import activate_new_chroma_dir, get_active_chroma_dir, USE_CHROMA_SESSIONS

# Silence noisy deprecation warnings; we still keep runtime errors visible.
warnings.filterwarnings('ignore', category=DeprecationWarning)
try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning  # type: ignore
    warnings.filterwarnings('ignore', category=LangChainDeprecationWarning)
except Exception:
    pass

from .embeddings import get_embedding_function

# PDF loader
try:
    from langchain_community.document_loaders import PyPDFDirectoryLoader
except Exception:
    from langchain.document_loaders.pdf import PyPDFDirectoryLoader  # type: ignore

# Splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

# Chroma
try:
    from langchain_chroma import Chroma  # type: ignore
except Exception:  # pragma: no cover
    from langchain_community.vectorstores import Chroma  # type: ignore

COLLECTION_NAME = "iot_shepherd_manuals"
KB_META_FILENAME = "kb_meta.json"



def _ensure_writable_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)
    test = p / ".write_test"
    try:
        test.write_text("ok", encoding="utf-8")
        try:
            test.unlink()
        except Exception:
            pass
    except Exception as e:
        raise PermissionError(
            f"Chroma directory is not writable: {p}. "
            "This commonly happens if the app is running from a read-only location or the path is wrong. "
            f"Original error: {e}"
        )


def _rmtree_force(p: Path) -> None:
    """Remove a directory tree robustly (handles read-only bits and transient file locks).

    On macOS, sqlite files can briefly appear "busy" after a process used them.
    We retry a few times with backoff and aggressively chmod entries before deletion.
    """
    p = Path(p)
    if not p.exists():
        return

    def _onerror(func, path, excinfo):
        try:
            os.chmod(path, 0o777)
        except Exception:
            pass
        try:
            func(path)
        except Exception:
            pass

    last_err: Exception | None = None
    for attempt in range(6):
        try:
            shutil.rmtree(p, onerror=_onerror)
            if not p.exists():
                return
        except Exception as e:
            last_err = e
        # Try to clear Python references/handles and backoff
        try:
            gc.collect()
        except Exception:
            pass
        time.sleep(0.15 * (2 ** attempt))

    # Final attempt: remove remaining children individually
    try:
        if p.exists():
            for child in p.rglob("*"):
                try:
                    if child.is_file() or child.is_symlink():
                        try:
                            child.chmod(0o777)
                        except Exception:
                            pass
                        child.unlink(missing_ok=True)
                    elif child.is_dir():
                        try:
                            child.chmod(0o777)
                        except Exception:
                            pass
                except Exception:
                    pass
            # remove dirs bottom-up
            for child in sorted([d for d in p.rglob("*") if d.is_dir()], reverse=True):
                try:
                    child.rmdir()
                except Exception:
                    pass
            try:
                p.rmdir()
            except Exception:
                pass
    except Exception as e:
        last_err = e

    if p.exists():
        # Keep error informative but don't crash higher-level flows unless absolutely necessary.
        raise RuntimeError(f"Failed to delete Chroma directory: {p}. Last error: {last_err}")


def _ensure_writable_tree(p: Path) -> None:
    """Ensure directory exists and is writable; also tries to fix permissions on common Chroma files."""
    _ensure_writable_dir(p)
    # Best-effort: fix permissions on any existing DB files
    for name in ("chroma.sqlite3",):
        f = p / name
        if f.exists():
            try:
                f.chmod(0o600)
            except Exception:
                pass
    try:
        p.chmod(0o755)
    except Exception:
        pass
def _kb_meta_path(chroma_dir: Path) -> Path:
    return Path(chroma_dir) / KB_META_FILENAME


def _write_kb_meta(chroma_dir: Path, payload: Dict[str, Any]) -> None:
    """Write small status metadata for fast UI stats without opening Chroma."""
    try:
        p = _kb_meta_path(chroma_dir)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _read_kb_meta(chroma_dir: Path) -> Dict[str, Any]:
    try:
        p = _kb_meta_path(chroma_dir)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}



def clear_chroma(chroma_dir: Path) -> Path:
    """Reset Chroma persistence.

    IMPORTANT (macOS/SQLite): deleting or renaming a Chroma directory that contains a sqlite DB
    can fail if any process still holds a file handle. This leads to recurring schema errors
    (e.g., default_tenant / CHROMA_SCHEMA_MISMATCH).

    We therefore default to a *session-based* persistence strategy:
      - Reset = activate a fresh session directory (no destructive delete required).
      - Old sessions can be cleaned up best-effort (optional).

    Returns the (new) active chroma directory.
    """
    chroma_dir = Path(chroma_dir)

    if USE_CHROMA_SESSIONS:
        # Safe reset: switch to a brand new session directory.
        return activate_new_chroma_dir()

    # Legacy mode: destructive delete/rename (best effort)
    if chroma_dir.exists():
        ts = int(time.time())
        renamed = chroma_dir.parent / f"{chroma_dir.name}.bak_{ts}"
        try:
            chroma_dir.rename(renamed)
        except Exception:
            _rmtree_force(chroma_dir)

    _ensure_writable_tree(chroma_dir)
    return chroma_dir


def _calculate_chunk_ids(chunks):
    """Assign stable, readable chunk IDs.

    IMPORTANT: Do not embed absolute file paths in IDs (keeps UI clean and portable).
    """
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        src = str(chunk.metadata.get("source", "") or "")
        source = Path(src).name or src
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        # Normalize displayed source too
        chunk.metadata["source"] = source
        last_page_id = current_page_id
    return chunks


def _looks_like_chroma_tenant_schema_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        ("no such table" in msg and "tenants" in msg)
        or ("tenant" in msg and "no such table" in msg)
        or ("could not connect to tenant" in msg)
        or ("default_tenant" in msg and "tenant" in msg)
        or ("are you sure it exists" in msg and "tenant" in msg)
    )


def _looks_like_readonly_db_error(err: Exception) -> bool:
    msg = str(err).lower()
    return ("readonly database" in msg) or ("attempt to write a readonly database" in msg) or ("code: 1032" in msg)

def _open_chroma(chroma_dir: Path, embed_model: str) -> Chroma:
    embeddings = get_embedding_function(model=embed_model)
    try:
        # NOTE: collection_name avoids accidental mismatches (some stacks default to 'langchain')
        return Chroma(
            persist_directory=str(chroma_dir),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
    except Exception as e:
        # Common when an old chroma.sqlite3 exists from an older schema.
        if _looks_like_chroma_tenant_schema_error(e):
            # CHROMA_SCHEMA_MISMATCH / default_tenant errors happen when an old persisted schema is opened
            # with a different ChromaDB version. On macOS, deleting/renaming sqlite folders can fail due
            # to file locks. We therefore prefer session-based recovery when enabled.
            try:
                if USE_CHROMA_SESSIONS:
                    new_dir = activate_new_chroma_dir()
                    _write_kb_meta(new_dir, {
                        "chunks_indexed": 0,
                        "note": "Auto-recovered from CHROMA_SCHEMA_MISMATCH by activating a fresh session. Please re-index manuals.",
                    })
                    return Chroma(
                        persist_directory=str(new_dir),
                        embedding_function=embeddings,
                        collection_name=COLLECTION_NAME,
                    )

                # Legacy recovery (best-effort): rename/delete the incompatible directory in-place
                chroma_dir = Path(chroma_dir)
                if chroma_dir.exists():
                    ts = int(time.time())
                    bad_dir = chroma_dir.parent / f"{chroma_dir.name}.schema_mismatch_{ts}"
                    i = 0
                    while bad_dir.exists():
                        i += 1
                        bad_dir = chroma_dir.parent / f"{chroma_dir.name}.schema_mismatch_{ts}_{i}"
                    try:
                        chroma_dir.rename(bad_dir)
                    except Exception:
                        _rmtree_force(chroma_dir)
                _ensure_writable_tree(chroma_dir)
                _write_kb_meta(chroma_dir, {
                    "chunks_indexed": 0,
                    "note": "Auto-recovered from CHROMA_SCHEMA_MISMATCH; please re-index manuals.",
                })
                return Chroma(
                    persist_directory=str(chroma_dir),
                    embedding_function=embeddings,
                    collection_name=COLLECTION_NAME,
                )
            except Exception:
                # Fall through to a clear error if recovery fails.
                raise RuntimeError(

                    "CHROMA_SCHEMA_MISMATCH: Your Chroma DB schema is incompatible with the installed ChromaDB version. "
                    "Fix: Reset the Chroma index (destructive) or force-delete the Chroma folder and re-index manuals. "
                    "Details: Could not connect to tenant default_tenant. Are you sure it exists?"
                ) from e
        if _looks_like_readonly_db_error(e):
            raise RuntimeError(
                "CHROMA_READONLY: Chroma persistence directory appears read-only or sqlite is locked. "
                "Fix: Close other processes using the DB, then reset the Chroma index."
            ) from e
        raise
        raise


def ingest_manuals(
    manuals_dir: Path,
    chroma_dir: Path,
    embed_model: str = "embeddinggemma:latest",
    reset: bool = False,
) -> Dict[str, Any]:
    manuals_dir.mkdir(parents=True, exist_ok=True)
    _ensure_writable_tree(chroma_dir)

    if reset:
        chroma_dir = clear_chroma(chroma_dir)

    loader = PyPDFDirectoryLoader(str(manuals_dir))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    chunks = _calculate_chunk_ids(chunks)

    # Open DB (auto-recover on schema mismatch by rebuilding the index directory)
    try:
        db = _open_chroma(chroma_dir, embed_model)
    except RuntimeError as e:
        msg = str(e)
        if ("CHROMA_SCHEMA_MISMATCH" in msg) or reset:
            chroma_dir = clear_chroma(chroma_dir)
            db = _open_chroma(chroma_dir, embed_model)
        else:
            raise

    existing = db.get(include=[])
    existing_ids = set(existing.get("ids", [])) if isinstance(existing, dict) else set()

    new_chunks = [c for c in chunks if c.metadata.get("id") not in existing_ids]
    if new_chunks:
        new_ids = [c.metadata["id"] for c in new_chunks]
        try:
            db.add_documents(new_chunks, ids=new_ids)
        except Exception as e:
            if _looks_like_readonly_db_error(e):
                raise PermissionError(
                    f"Chroma failed to write to its SQLite database (read-only).\n\n"
                    f"Chroma directory: {chroma_dir}\n\n"
                    "Fix: Ensure the project folder is writable and no other process is holding chroma.sqlite3 open. "
                    "Try: (1) stop other Streamlit instances, (2) click Reset Chroma index, (3) re-run indexing. "
                    "On macOS/Linux you can also run: chmod -R u+rwX storage && rm -rf storage/chroma"
                ) from e
            raise
    
    # Persist lightweight KB stats for the UI without opening Chroma later
    _write_kb_meta(
        chroma_dir,
        {
            "chunks_indexed": len(existing_ids) + len(new_chunks),
            "docs_loaded": len(docs),
            "chunks_total": len(chunks),
            "chunks_added": len(new_chunks),
            "collection": COLLECTION_NAME,
        },
    )

    return {
        "docs_loaded": len(docs),
        "chunks_total": len(chunks),
        "chunks_added": len(new_chunks),
        "db_size_before": len(existing_ids),
        "db_size_after": len(existing_ids) + len(new_chunks),
        "collection": COLLECTION_NAME,
        "manuals_dir": str(manuals_dir),
        "chroma_dir": str(chroma_dir),
    }


def kb_stats(chroma_dir: Path, embed_model: str = "embeddinggemma:latest") -> Dict[str, Any]:
    """Return KB status without opening Chroma.

    We avoid opening Chroma here to prevent schema-mismatch errors from showing up as noisy warnings.
    Accurate counts are taken from a small meta file written after indexing.
    """
    chroma_dir = Path(chroma_dir)
    try:
        entries = sorted([p.name for p in chroma_dir.iterdir()]) if chroma_dir.exists() else []
    except Exception:
        entries = []

    meta = _read_kb_meta(chroma_dir)
    n = int(meta.get("chunks_indexed", 0) or 0)

    return {
        "chroma_dir": str(chroma_dir),
        "collection": COLLECTION_NAME,
        "chunks_indexed": n,
        "dir_entries": entries,
        "meta": meta,
    }
