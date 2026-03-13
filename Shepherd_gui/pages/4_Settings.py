from __future__ import annotations
import streamlit as st
from src.iot_shepherd.config import AppSettings

st.set_page_config(page_title="Settings | IoT Shepherd", page_icon="⚙️", layout="wide")
settings = AppSettings()

st.title("Settings")
st.caption("Settings are controlled by environment variables for reproducible runs.")

st.markdown("#### LLM (Ollama)")
st.code(f"OLLAMA_HOST={settings.ollama_host}\nIOT_SHEPHERD_LLM_MODEL={settings.llm_model}")

st.markdown("#### Embeddings")
st.code(f"IOT_SHEPHERD_EMBED_MODEL={settings.embed_model}")

st.markdown("#### Paths")
st.write("Manuals:", settings.manuals_dir)
st.write("Chroma:", settings.chroma_dir)
st.write("ADM model:", settings.adm_model_dir)
st.write("Runs:", settings.runs_dir)

st.markdown("#### External search")
st.write("Enabled:", settings.enable_web_search)
