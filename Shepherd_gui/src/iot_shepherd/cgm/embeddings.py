from __future__ import annotations

import os
from langchain_ollama import OllamaEmbeddings


def get_embedding_function(model: str = "embeddinggemma:latest", host: str | None = None):
    # OllamaEmbeddings uses base_url; default to OLLAMA_HOST if provided.
    base_url = host or os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    return OllamaEmbeddings(model=model, base_url=base_url)
