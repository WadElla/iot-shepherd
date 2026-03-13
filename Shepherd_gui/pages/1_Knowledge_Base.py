from __future__ import annotations

import gc
import shutil
from pathlib import Path
import streamlit as st

from src.iot_shepherd.config import AppSettings, get_active_chroma_dir, activate_new_chroma_dir, list_chroma_sessions, USE_CHROMA_SESSIONS
from src.iot_shepherd.utils.io import safe_filename, write_upload_to_path
from src.iot_shepherd.cgm.kb import ingest_manuals, kb_stats, clear_chroma


st.set_page_config(page_title="Knowledge Base | IoT Shepherd", page_icon="📚", layout="wide")
settings = AppSettings()

# Active Chroma persistence directory (session-based when enabled)
active_chroma_dir = get_active_chroma_dir()

st.title("Knowledge Base (CGM)")
st.caption(
    "Upload IoT manuals (PDFs) and build/update the local Chroma index. "
    "This page is **non-agent** by design."
)

def _clear_caches() -> None:
    # Release any cached resources that might hold a file handle to chroma.sqlite3
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    # Encourage GC to drop any lingering db objects
    gc.collect()

def _safe_rmtree(p: Path) -> None:
    if not p.exists():
        return
    def _onerror(func, path, excinfo):
        try:
            Path(path).chmod(0o777)
            func(path)
        except Exception:
            pass
    shutil.rmtree(p, onerror=_onerror)

def _dir_nonempty(p: Path) -> bool:
    try:
        return p.exists() and any(p.iterdir())
    except Exception:
        return False
def _report_chroma_state(p: Path) -> None:
    try:
        entries = sorted([x.name for x in p.iterdir()]) if p.exists() else []
        st.info(f"Chroma dir state: exists={p.exists()} | entries={entries if entries else '[]'}")
    except Exception as e:
        st.warning(f"Could not inspect Chroma dir: {type(e).__name__}: {e}")



# -------------------------
# Upload manuals
# -------------------------
uploads = st.file_uploader("Upload PDF manuals", type=["pdf"], accept_multiple_files=True)
if uploads:
    saved = 0
    for uf in uploads:
        dest = settings.manuals_dir / safe_filename(uf.name)
        write_upload_to_path(uf, dest)
        saved += 1
    st.success(f"Saved {saved} file(s) into {settings.manuals_dir}")

# -------------------------
# Current manuals + stats
# -------------------------
st.markdown("#### Manuals in repository")
pdfs = sorted(settings.manuals_dir.glob("*.pdf"))
if pdfs:
    st.write(f"{len(pdfs)} PDF(s) found.")
    with st.expander("Show manuals list"):
        for p in pdfs:
            st.code(str(p), language="text")
else:
    st.info("No manuals uploaded yet.")

stats = kb_stats(active_chroma_dir, embed_model=settings.embed_model)

# Surface automatic schema recovery notes (if any)
_note = (stats.get('meta') or {}).get('note')
if isinstance(_note, str) and _note:
    if 'Auto-recovered from CHROMA_SCHEMA_MISMATCH' in _note:
        st.warning(_note)

st.markdown("#### Index status")
c1, c2, c3 = st.columns(3)
c1.metric("Chunks indexed", stats.get("chunks_indexed", 0))
c2.write("Collection:")
c2.code(stats.get("collection", ""), language="text")
c3.write("Chroma path:")
c3.code(str(active_chroma_dir), language="text")

with st.expander("Chroma directory state", expanded=False):
    st.json({"dir_entries": stats.get("dir_entries", [])})


# -------------------------
# Build / Update index
# -------------------------
st.markdown("#### Build / Update index")
reset = st.checkbox(
    "Reset index (destructive)",
    value=False,
    help="Deletes the existing Chroma directory before indexing.",
)

if st.button("Run indexing", type="primary", use_container_width=True):
    with st.status("Indexing manuals...", expanded=True) as status:
        try:
            if reset:
                status.write("Resetting Chroma directory...")
                _clear_caches()
                active_chroma_dir = clear_chroma(active_chroma_dir)
                _clear_caches()
                if _dir_nonempty(active_chroma_dir):
                    st.warning("Chroma directory is not empty after repair. A process may still be holding files open.")

            status.write("Loading PDFs and splitting into chunks...")
            out = ingest_manuals(
                manuals_dir=settings.manuals_dir,
                chroma_dir=active_chroma_dir,
                embed_model=settings.embed_model,
                reset=False,  # reset handled above for clearer UX
            )
            status.update(label="Index update complete", state="complete")
            st.success("Index update complete.")
            st.json(out)
        except Exception as e:
            status.update(label="Indexing failed", state="error")
            st.error(f"{type(e).__name__}: {e}")

# -------------------------
# Maintenance controls
# -------------------------
st.markdown("#### Maintenance controls")
m1, m2, m3 = st.columns(3)

with m1:
    if st.button("Clear manuals folder", use_container_width=True):
        with st.spinner("Deleting manuals..."):
            for f in settings.manuals_dir.glob("*.pdf"):
                try:
                    f.unlink()
                except Exception:
                    pass
        st.success("Manuals cleared.")
        st.rerun()

with m2:
    if st.button("Reset Chroma index", use_container_width=True):
        # Soft reset via clear_chroma (rename + cleanup + recreate)

        with st.spinner("Resetting Chroma index..."):
            _clear_caches()
            try:
                active_chroma_dir = clear_chroma(active_chroma_dir)
            except Exception as e:
                st.error("Reset failed. Close other Streamlit instances and try again.")
                st.code(f"{type(e).__name__}: {e}")
                st.stop()
            _clear_caches()
            if _dir_nonempty(active_chroma_dir):
                st.warning("Chroma directory is not empty after reset. A process may still be holding files open.")
        st.success("Chroma index reset.")
        _report_chroma_state(active_chroma_dir)
        st.caption("Refreshing index status…")
        st.json(kb_stats(active_chroma_dir, embed_model=settings.embed_model))
        st.rerun()

    st.markdown("**Advanced:**")
    if st.button("Force delete Chroma folder (hard)", use_container_width=True):
        with st.spinner("Force-deleting Chroma folder..."):
            try:
                active_chroma_dir = clear_chroma(active_chroma_dir)
                st.success("Chroma folder force-deleted and recreated.")
            except Exception as e:
                st.error("Force delete failed. Close any process using the folder and try again, or delete manually.")
                st.code(f"{type(e).__name__}: {e}")
        st.rerun()

with m3:

    if st.button("Clear cached resources", use_container_width=True):
        with st.spinner("Clearing Streamlit caches..."):
            _clear_caches()
        st.success("Caches cleared.")
