from __future__ import annotations

import json
from pathlib import Path
import streamlit as st
import plotly.express as px
import pandas as pd

from src.iot_shepherd.config import AppSettings, validate_adm_model_dir
from src.iot_shepherd.adm.pipeline import analyze_pcap_file

st.set_page_config(page_title="Traffic Analysis | IoT Shepherd", page_icon="📡", layout="wide")
settings = AppSettings()

ok_model, model_msg = validate_adm_model_dir(settings.adm_model_dir)
if not ok_model:
    st.error("ADM model not found. " + model_msg)
    st.info("Fix: place the saved model folder at the configured path or set IOT_SHEPHERD_ADM_MODEL_DIR.")
    st.stop()


st.title("Traffic Analysis (ADM)")
st.caption("Upload a PCAP, extract features, run the ADM model, and produce an Incident Card.")

uploaded = st.file_uploader("Upload a PCAP file", type=["pcap", "pcapng"])
max_packets_val = st.number_input("Max packets to process (optional)", min_value=0, value=0, step=1000)
max_packets = None if max_packets_val == 0 else int(max_packets_val)

c1, c2 = st.columns([1, 1])
with c1:
    run_btn = st.button("Analyze PCAP", type="primary", disabled=uploaded is None)
with c2:
    if st.button("Clear latest Incident Card", use_container_width=True):
        st.session_state.pop("latest_incident", None)
        st.success("Cleared latest incident from session.")
        st.rerun()

if uploaded and run_btn:
    with st.status("Running ADM...", expanded=True) as status:
        try:
            status.write("Extracting features...")
            incident = analyze_pcap_file(
                uploaded_file=uploaded,
                runs_dir=settings.runs_dir,
                model_dir=settings.adm_model_dir,
                max_packets=max_packets,
            )
            st.session_state["latest_incident"] = incident
            # Persist pointers for agent tools / UI convenience
            try:
                (Path(settings.runs_dir) / "latest_pcap.txt").write_text(str(incident.get("pcap_path","")), encoding="utf-8")
            except Exception:
                pass
            try:
                (Path(settings.runs_dir) / "latest_incident.json").write_text(json.dumps(incident, indent=2), encoding="utf-8")
            except Exception:
                pass

            status.update(label="ADM complete", state="complete")
            st.success("Analysis complete.")
        except Exception as e:
            status.update(label="ADM failed", state="error")
            st.error(f"{type(e).__name__}: {e}")

if "latest_incident" in st.session_state:
    incident = st.session_state["latest_incident"]
    st.markdown("### Incident Summary")
    s = incident.get("summary", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total samples", s.get("total_samples", 0))
    c2.metric("Anomalous samples", s.get("anomalous_samples", 0))
    c3.metric("Anomaly rate (%)", f"{s.get('anomaly_percent', 0.0):.2f}")
    c4.metric("Dominant attack", s.get("dominant_attack") or "None")

    top = incident.get("top_attacks", []) or []
    if top:
        df = pd.DataFrame(top)
        fig = px.bar(df, x="attack_type", y="count", title="Top detected attack types (by count)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Artifacts")
    art = incident.get("artifacts", {}) or {}
    st.write("**Run ID:**", incident.get("run_id"))
    st.write("**PCAP:**", incident.get("pcap_path"))
    st.write("**Predictions CSV:**", art.get("results_csv"))
    st.write("**Report TXT:**", art.get("report_txt"))
    st.write("**Incident Card JSON:**", art.get("incident_json"))

    with st.expander("Incident Card (JSON)"):
        st.json(incident)

    st.info(
        "Go to **Ask IoT Shepherd → Agentic Incident Guidance** to load this Incident Card and generate mitigation guidance.\n"
        "Go to **Ask IoT Shepherd → Manuals Q&A** for manuals-only questions."
    )