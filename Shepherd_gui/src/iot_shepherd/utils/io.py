from __future__ import annotations
from pathlib import Path
from typing import BinaryIO
import hashlib
import os
import re
from datetime import datetime, timezone

SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def safe_filename(name: str) -> str:
    name = os.path.basename(name)
    name = SAFE_NAME_RE.sub("_", name).strip("_")
    return name or "upload.bin"

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def write_upload_to_path(uploaded_file, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Streamlit UploadedFile provides .getbuffer()
    with dest.open("wb") as out:
        out.write(uploaded_file.getbuffer())
    return dest
