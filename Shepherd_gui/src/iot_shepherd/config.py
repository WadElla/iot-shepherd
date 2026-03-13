from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import time
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # <repo>/src/iot_shepherd/config.py -> <repo>

def _ensure_writable_dir(p: Path) -> None:
    """Create directory and confirm it is writable."""
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
            f"Directory is not writable: {p}\n\n"
            "Fix options:\n"
            "  1) Move the project to a writable location (e.g., your home directory), or\n"
            "  2) Set an explicit writable storage location:\n"
            "     export IOT_SHEPHERD_STORAGE_DIR=~/.iot_shepherd/storage\n\n"
            f"Original error: {type(e).__name__}: {e}"
        )

# -----------------------
# Data / storage locations
# -----------------------

DATA_DIR = PROJECT_ROOT / "data"
MANUALS_DIR = DATA_DIR / "manuals"
PCAPS_DIR = DATA_DIR / "pcaps"

# Storage (default is project-local unless overridden)
STORAGE_DIR = Path(os.getenv("IOT_SHEPHERD_STORAGE_DIR", str(PROJECT_ROOT / "storage")))

# Chroma persistence strategy:
# - To avoid macOS sqlite file-lock / schema mismatch issues, we default to session-based directories
#   (similar to the Revelation approach): storage/chroma_sessions/<session_id>/
# - "Reset index" simply activates a new session folder (no destructive delete required).
USE_CHROMA_SESSIONS = os.getenv("IOT_SHEPHERD_USE_CHROMA_SESSIONS", "1").strip().lower() in ("1","true","yes","y","on")

CHROMA_SESSIONS_DIR = Path(os.getenv("IOT_SHEPHERD_CHROMA_SESSIONS_DIR", str(STORAGE_DIR / "chroma_sessions")))
LATEST_SESSION_FILE = CHROMA_SESSIONS_DIR / "latest_session.txt"

# Legacy single-folder chroma (only used if sessions disabled)
CHROMA_DIR = Path(os.getenv("IOT_SHEPHERD_CHROMA_DIR", str(STORAGE_DIR / "chroma")))

# Models
MODELS_DIR = PROJECT_ROOT / "models"
ADM_MODEL_DIR = Path(os.getenv("IOT_SHEPHERD_ADM_MODEL_DIR", str(MODELS_DIR / "adm_bert")))

# Runs
RUNS_DIR = PROJECT_ROOT / "runs"

# Ensure required base directories exist
for p in (MANUALS_DIR, PCAPS_DIR, STORAGE_DIR, MODELS_DIR, RUNS_DIR, CHROMA_SESSIONS_DIR):
    _ensure_writable_dir(Path(p))
if not USE_CHROMA_SESSIONS:
    _ensure_writable_dir(Path(CHROMA_DIR))


def _new_session_id() -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"sess_{ts}_{uuid.uuid4().hex[:8]}"


def _read_latest_session_id() -> str:
    if LATEST_SESSION_FILE.exists():
        sid = (LATEST_SESSION_FILE.read_text(encoding="utf-8", errors="ignore") or "").strip()
        if sid:
            return sid
    sid = _new_session_id()
    LATEST_SESSION_FILE.write_text(sid, encoding="utf-8")
    return sid


def get_active_chroma_dir() -> Path:
    """Return the active Chroma persistence directory."""
    if not USE_CHROMA_SESSIONS:
        _ensure_writable_dir(CHROMA_DIR)
        return CHROMA_DIR
    CHROMA_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sid = _read_latest_session_id()
    p = CHROMA_SESSIONS_DIR / sid
    _ensure_writable_dir(p)
    return p


def activate_new_chroma_dir() -> Path:
    """Activate a fresh Chroma persistence directory (safe reset)."""
    if not USE_CHROMA_SESSIONS:
        # legacy mode: keep using CHROMA_DIR (reset handled elsewhere)
        _ensure_writable_dir(CHROMA_DIR)
        return CHROMA_DIR
    sid = _new_session_id()
    CHROMA_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_SESSION_FILE.write_text(sid, encoding="utf-8")
    p = CHROMA_SESSIONS_DIR / sid
    _ensure_writable_dir(p)
    return p


def list_chroma_sessions() -> list[Path]:
    if not CHROMA_SESSIONS_DIR.exists():
        return []
    return sorted([p for p in CHROMA_SESSIONS_DIR.iterdir() if p.is_dir()], key=lambda x: x.name, reverse=True)


def validate_adm_model_dir(model_dir: Path) -> tuple[bool, str]:
    """Validate that the saved ADM BERT model artifacts exist.

    Accepts typical Transformers exports:
      - config.json AND (pytorch_model.bin OR model.safetensors)
      - tokenizer.json OR tokenizer_config.json (recommended)
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return False, f"ADM model directory not found: {model_dir}"
    if not model_dir.is_dir():
        return False, f"ADM model path is not a directory: {model_dir}"

    cfg_file = model_dir / "config.json"
    w_bin = model_dir / "pytorch_model.bin"
    w_safe = model_dir / "model.safetensors"
    tok_json = model_dir / "tokenizer.json"
    tok_cfg = model_dir / "tokenizer_config.json"

    if not cfg_file.exists():
        return False, f"Missing config.json in ADM model directory: {model_dir}"
    if not (w_bin.exists() or w_safe.exists()):
        return False, f"Missing model weights (pytorch_model.bin or model.safetensors) in: {model_dir}"
    if not (tok_json.exists() or tok_cfg.exists()):
        return True, f"Model found, but tokenizer files are missing (tokenizer.json/tokenizer_config.json). If inference fails, copy tokenizer files into {model_dir}."
    return True, "ADM model looks present."


@dataclass(frozen=True)
class AppSettings:
    # LLM backend (Ollama by default)
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llm_model: str = os.getenv("IOT_SHEPHERD_LLM_MODEL", "llama3.2:latest")

    # Embeddings (Ollama embeddings)
    embed_model: str = os.getenv("IOT_SHEPHERD_EMBED_MODEL", "embeddinggemma:latest")

    # External search (optional)
    enable_web_search: bool = os.getenv("IOT_SHEPHERD_ENABLE_WEB_SEARCH", "0").strip().lower() in ("1","true","yes","y","on")

    # Knowledge base
    chroma_dir: Path = field(default_factory=get_active_chroma_dir)
    manuals_dir: Path = MANUALS_DIR

    # ADM
    adm_model_dir: Path = ADM_MODEL_DIR

    # Runs
    runs_dir: Path = RUNS_DIR
