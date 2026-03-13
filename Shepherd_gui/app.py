from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

from src.iot_shepherd.config import AppSettings
from src.iot_shepherd.cgm.kb import kb_stats

st.set_page_config(page_title="IoT Shepherd", page_icon="🛡️", layout="wide")

css_path = Path(__file__).parent / "assets" / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

settings = AppSettings()

st.title("IoT Shepherd")
st.caption("GUI for the IoT Shepherd paper: separable CGM (manuals), ADM (traffic analysis), and agentic coordination.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="card"><h3>CGM</h3><div class="small-muted">Manuals ingestion + retrieval grounded answers.</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="card"><h3>ADM</h3><div class="small-muted">PCAP → features → BERT → Incident Card artifacts.</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="card"><h3>Shepherd Agent</h3><div class="small-muted">Uses Incident Card to generate retrieval queries + mitigation guidance.</div></div>', unsafe_allow_html=True)

st.markdown("### End-to-end workflows")

# Single, complete diagram (no split diagrams)
diagram_html = r"""
<div id="mermaid" style="width: 100%;"></div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
  mermaid.initialize({ startOnLoad: true, theme: "dark" });
  const chart = `
flowchart LR

  %% One diagram, three explicit workflows (paper-faithful)

  subgraph W1["Manuals-only mode (CGM)"]
    direction TB
    U1["Upload manuals PDFs"] --> I1["Index manuals → chunks"] --> DB1["Chroma knowledge base"]
    Q1["Ask question"] --> R1["Retrieve evidence chunks"] --> A1["Manuals-grounded answer"]
    DB1 --> R1
  end

  subgraph W2["Traffic-analysis-only mode (ADM)"]
    direction TB
    P2["Upload PCAP"] --> FE2["Feature extraction"] --> B2["BERT inference"]
    B2 --> IC2["Incident Card"] --> OUT2["Artifacts<br/>• adm_predictions.csv<br/>• bert_traffic_report.txt<br/>• incident_card.json"]
  end

  subgraph W3["Agentic mode (Shepherd Agent)"]
    direction TB
    IC3["Active Incident Card"]:::focus
    P3["Upload PCAP (optional)"] --> ADM3["Run ADM (optional)"] --> IC3
    U3["Upload incident_card.json"] --> IC3

    IC3 --> S31["Extract incident signals"]
    S31 --> S32["Generate 4–8 retrieval queries"]
    S32 --> S33["Retrieve manuals evidence (CGM)"]
    S33 --> S34["Compose mitigation guidance<br/>(cite chunk IDs)"]
    WS3["Optional web search (if enabled)"] --> S34
    S34 --> RESP3["Evidence-grounded response"]
  end

  classDef focus fill:#1f2a44,stroke:#7aa2f7,stroke-width:2px;

`;
  document.getElementById("mermaid").innerHTML = `<pre class="mermaid">${chart}</pre>`;
  mermaid.init(undefined, document.querySelectorAll(".mermaid"));
</script>
"""
components.html(diagram_html, height=520, scrolling=True)

st.markdown("### Quick status")
stats = kb_stats(settings.chroma_dir, embed_model=settings.embed_model)
c1, c2, c3 = st.columns(3)
c1.metric("Manual chunks indexed", stats.get("chunks_indexed", 0))
c2.write("Chroma dir:")
c2.code(str(settings.chroma_dir), language="text")
c3.write("Models dir:")
c3.code(str(settings.adm_model_dir), language="text")

if stats.get("error"):
    st.warning("KB status warning: " + stats["error"])
