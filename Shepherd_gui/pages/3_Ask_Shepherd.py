from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, List

import streamlit as st

from src.iot_shepherd.config import AppSettings
from src.iot_shepherd.cgm.qa import answer_from_manuals
from src.iot_shepherd.agent.shepherd_agent import build_shepherd_agent
from src.iot_shepherd.utils.io import safe_filename, write_upload_to_path


# --------------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Ask IoT Shepherd | IoT Shepherd",
    page_icon="🛡️",
    layout="wide",
)

settings = AppSettings()

st.title("Ask IoT Shepherd")
st.caption(
    "Use **Manuals-only (CGM)** for stable documentation Q&A, or **Agentic mode** where the Shepherd Agent "
    "uses an Incident Card to autonomously retrieve manuals evidence and produce mitigation guidance. "
    "Agentic behavior is **always admin-controlled**."
)

# --------------------------------------------------------------------------------------
# Session keys (stable)
# --------------------------------------------------------------------------------------
if "manual_messages" not in st.session_state:
    st.session_state["manual_messages"] = []  # List[Dict[str,str]]

if "agent_messages" not in st.session_state:
    st.session_state["agent_messages"] = []  # List[Dict[str,str]]

# Active Incident Card
# - stored as dict in-memory for convenience
# - stored to disk as runs/latest_incident.json by Traffic Analysis page already
if "active_incident_card" not in st.session_state:
    st.session_state["active_incident_card"] = None  # Optional[Dict[str,Any]]
if "active_incident_source" not in st.session_state:
    st.session_state["active_incident_source"] = None  # Optional[str]


# --------------------------------------------------------------------------------------
# Helpers (robust / Revelation-style)
# --------------------------------------------------------------------------------------
def _normalize_resp(resp: Any) -> str:
    """Normalize responses across Agno versions."""
    if isinstance(resp, str):
        return resp
    if hasattr(resp, "content"):
        return getattr(resp, "content")
    if hasattr(resp, "response"):
        return getattr(resp, "response")
    if isinstance(resp, dict) and "content" in resp:
        return str(resp["content"])
    return str(resp)


def _agent_call(agent: Any, message: str) -> str:
    """Call agent across method drift: run/respond/chat/invoke."""
    for method in ("run", "respond", "chat", "invoke"):
        if hasattr(agent, method):
            try:
                resp = getattr(agent, method)(message)
            except TypeError:
                resp = getattr(agent, method)(message=message)
            out = _normalize_resp(resp)
            return out
    return "⚠️ Agent invocation method not found. Please update Agno to a compatible version."


def _set_active_incident(incident: Dict[str, Any], source: str) -> None:
    st.session_state["active_incident_card"] = incident
    st.session_state["active_incident_source"] = source


def _clear_active_incident() -> None:
    st.session_state["active_incident_card"] = None
    st.session_state["active_incident_source"] = None


def _try_load_latest_incident_from_disk() -> Optional[Dict[str, Any]]:
    """Load runs/latest_incident.json if present."""
    try:
        p = Path(settings.runs_dir) / "latest_incident.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _compact_incident_text(incident: Dict[str, Any]) -> str:
    """Return the FULL incident card (no truncation), as requested.

    The paper flow uses an Incident Card as the primary artifact. For robustness and transparency,
    we inject the complete JSON into the agent prompt.
    """
    # Avoid markdown code-fences here. Some local models react to ```json blocks by emitting structured JSON
    # (including tool-call-like blobs) instead of invoking tools. We wrap the JSON in explicit tags instead.
    try:
        payload = json.dumps(incident, indent=2, ensure_ascii=False)
    except Exception:
        payload = str(incident)
    return "<INCIDENT_CARD_JSON>\n" + payload + "\n</INCIDENT_CARD_JSON>\n"
def _render_chat(messages_key: str) -> None:
    """Render Streamlit chat history."""
    for m in st.session_state.get(messages_key, []):
        role = m.get("role", "assistant")
        content = m.get("content", "")
        with st.chat_message(role):
            st.markdown(content)


# --------------------------------------------------------------------------------------
# Mode selector
# --------------------------------------------------------------------------------------
mode = st.radio(
    "Mode",
    options=[
        "📚 Manuals-only Q&A (CGM, non-agent)",
        "🛡️ Agentic Incident Guidance (Shepherd Agent)",
    ],
    horizontal=True,
)

st.divider()

# ======================================================================================
# Mode 1: Manuals-only Q&A (CGM)
# ======================================================================================
if mode.startswith("📚"):
    st.subheader("Manuals-only Q&A (CGM)")
    st.caption("This mode **does not use any agent**. Answers are grounded in indexed manuals.")

    # Controls
    c1, c2 = st.columns([1, 1])
    with c1:
        k = st.number_input("Top-k evidence chunks", min_value=1, max_value=20, value=5, step=1, help="How many manual chunks to retrieve.")
    with c2:
        if st.button("Clear manuals chat", use_container_width=True):
            st.session_state["manual_messages"] = []
            st.rerun()

    _render_chat("manual_messages")

    question = st.chat_input("Ask a question about your IoT manuals…")
    if question:
        st.session_state["manual_messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving manuals evidence and generating answer…"):
                try:
                    out = answer_from_manuals(
                        question=question,
                        chroma_dir=settings.chroma_dir,
                        embed_model=settings.embed_model,
                        llm_model=settings.llm_model,
                        ollama_host=settings.ollama_host,
                        k=int(k),
                    )
                    answer = out.get("answer", "").strip() or "(no answer returned)"
                    # Show answer
                    st.markdown(answer)

                    # Optional evidence block
                    with st.expander("Evidence (manual chunks)"):
                        ev = out.get("evidence", []) or []
                        if not ev:
                            st.info("No evidence chunks were retrieved. If you just reset the index, upload and re-index PDFs first.")
                        else:
                            # Show structured evidence (id, source, page, score, excerpt)
                            st.json(ev)

                    st.session_state["manual_messages"].append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")
                    st.session_state["manual_messages"].append(
                        {"role": "assistant", "content": f"⚠️ Error: {type(e).__name__}: {e}"}
                    )

# ======================================================================================
# Mode 2: Agentic mode (Shepherd Agent)
# ======================================================================================
else:
    st.subheader("Agentic Incident Guidance (Shepherd Agent)")
    st.caption(
        "Agentic mode uses an **Active Incident Card** to autonomously generate manuals retrieval queries and "
        "produce evidence-grounded mitigation guidance."
    )

    # -----------------------------
    # Agent toggles (admin control)
    # -----------------------------
    t1, t2, t3 = st.columns([1, 1, 2])

    with t1:
        enable_web = st.checkbox(
            "Enable web search (optional)",
            value=bool(st.session_state.get("agent_enable_web", False)),
            help="Agent may use external search ONLY if enabled and ONLY when manuals evidence is insufficient.",
        )
        st.session_state["agent_enable_web"] = enable_web

    with t2:
        allow_adm = st.checkbox(
            "Allow agent to run ADM (advanced)",
            value=bool(st.session_state.get("agent_allow_adm", False)),
            help="If enabled, the agent may call adm_analyze_pcap(). A provided PCAP path is enforced to prevent path hallucination.",
        )
        st.session_state["agent_allow_adm"] = allow_adm

    with t3:
        st.caption("Agent model")
        st.code(settings.llm_model, language="text")
        tools_txt = "CGM retrieve"
        if allow_adm:
            tools_txt = "ADM analyze PCAP + " + tools_txt
        if enable_web:
            tools_txt += " + Web search"
        st.caption(f"Tools available: {tools_txt}")

    @st.cache_resource(show_spinner=False)
    def _get_agent_cached(llm_model: str, ollama_host: str, enable_web_search: bool, allow_adm_calls: bool):
        return build_shepherd_agent(
            llm_model=llm_model,
            ollama_host=ollama_host,
            enable_web_search=enable_web_search,
            allow_adm=allow_adm_calls,
        )

    agent = _get_agent_cached(settings.llm_model, settings.ollama_host, bool(enable_web), bool(allow_adm))

    st.divider()

    # -----------------------------
    # Active Incident Card panel
    # -----------------------------
    st.markdown("### Active Incident Card")
    active = st.session_state.get("active_incident_card")
    src = st.session_state.get("active_incident_source")

    a1, a2, a3 = st.columns([1, 1, 2])

    with a1:
        if st.button("Load latest from Traffic Analysis", use_container_width=True):
            latest = st.session_state.get("latest_incident")
            if latest is None:
                latest = _try_load_latest_incident_from_disk()
            if latest:
                _set_active_incident(latest, source="latest_adm_run")
                st.success("Loaded latest Incident Card.")
            else:
                st.warning("No latest incident found. Run Traffic Analysis (ADM) first.")
            st.rerun()

    with a2:
        if st.button("Clear active Incident Card", use_container_width=True):
            _clear_active_incident()
            st.success("Cleared active Incident Card.")
            st.rerun()

    with a3:
        uploaded_ic = st.file_uploader("Upload incident_card.json", type=["json"], key="upload_ic_json")
        if uploaded_ic is not None:
            try:
                incident = json.loads(uploaded_ic.getvalue().decode("utf-8"))
                _set_active_incident(incident, source=f"upload:{uploaded_ic.name}")
                st.success(f"Loaded Incident Card from upload: {uploaded_ic.name}")
                st.rerun()
            except Exception as e:
                st.error(f"Could not parse JSON: {type(e).__name__}: {e}")

    if active:
        st.success(f"✅ Active Incident Card loaded ({src or 'unknown source'}).")
        with st.expander("View Incident Card JSON"):
            st.json(active)
    else:
        st.info("No active Incident Card loaded. Load the latest from Traffic Analysis or upload an incident_card.json.")

    st.divider()

    # -----------------------------
    # Optional: provide a PCAP path for the agent (ADM tool)
    # -----------------------------
    st.markdown("### Optional: Provide a PCAP for the agent (ADM tool)")
    st.caption(
        "If **Allow agent to run ADM** is enabled, you can upload a PCAP here. "
        "The ADM tool will use this provided PCAP path."
    )

    override_ptr = Path(settings.runs_dir) / "agent_pcap_override.txt"
    cur_override = None
    try:
        if override_ptr.exists():
            cur_override = override_ptr.read_text(encoding="utf-8", errors="ignore").strip() or None
    except Exception:
        cur_override = None

    p1, p2, p3 = st.columns([2, 1, 1])
    with p1:
        uploaded_pcap = st.file_uploader(
            "Upload PCAP for agent ADM tool (optional)",
            type=["pcap", "pcapng"],
            key="agent_tool_pcap",
            disabled=not bool(allow_adm),
        )
    with p2:
        max_packets = st.number_input(
            "Max packets (optional)",
            min_value=0,
            value=0,
            step=1000,
            help="0 = no limit",
            disabled=not bool(allow_adm),
        )
    with p3:
        if st.button("Clear provided PCAP", use_container_width=True, disabled=not bool(cur_override)):
            try:
                override_ptr.unlink(missing_ok=True)  # py3.10+: ok
            except Exception:
                pass
            st.success("Cleared provided PCAP override.")
            st.rerun()

    if cur_override:
        st.info(f"Current provided PCAP path (enforced): {cur_override}")
    elif allow_adm:
        st.warning("No PCAP is currently provided to the agent. Upload one above if you want the agent to run ADM.")

    def _save_agent_pcap_override(uploaded) -> Optional[str]:
        if uploaded is None:
            return None
        safe = safe_filename(getattr(uploaded, "name", "capture.pcap"))
        # Make the stored filename stable and unique to avoid accidental overwrites.
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if "." in safe:
            base, ext = safe.rsplit(".", 1)
            safe = f"{base}_{stamp}.{ext}"
        else:
            safe = f"{safe}_{stamp}"
        dest_dir = Path(settings.runs_dir) / "_agent_inputs"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / safe
        dest = write_upload_to_path(uploaded, dest)
        try:
            override_ptr.write_text(str(dest), encoding="utf-8")
            # Keep legacy pointer too, for visibility.
            (Path(settings.runs_dir) / "latest_pcap.txt").write_text(str(dest), encoding="utf-8")
        except Exception:
            pass
        return str(dest)

    # -----------------------------
    # Actions
    # -----------------------------
    b1, b2, b3 = st.columns([1, 1, 1])

    with b1:
        if st.button("Agent: analyze provided PCAP and propose mitigation", type="primary", use_container_width=True, disabled=not bool(allow_adm)):
            if uploaded_pcap is not None:
                _save_agent_pcap_override(uploaded_pcap)
            if not override_ptr.exists():
                st.warning("Upload a PCAP first (or ensure a provided PCAP override exists).")
            else:
                mp = int(max_packets) if int(max_packets) > 0 else None
                user_msg = "Analyze the provided PCAP with ADM and propose mitigation guidance."
                st.session_state["agent_messages"].append({"role": "user", "content": user_msg})
                with st.chat_message("user"):
                    st.markdown(user_msg)
                with st.chat_message("assistant"):
                    with st.spinner("Agent is running ADM + manuals retrieval…"):
                        # We include the enforced path for transparency (tool will still enforce it).
                        pcap_for_display = (override_ptr.read_text(encoding="utf-8", errors="ignore").strip() or "")
                        prompt = (
                            "You have a PCAP file provided by the admin. "
                            "Call adm_analyze_pcap(pcap_path=..., max_packets=...) to generate an Incident Card, "
                            "then follow your workflow: generate retrieval queries, call cgm_retrieve, and write mitigation.\n\n"
                            f"Provided PCAP path (must be used): {pcap_for_display}\n"
                            f"Max packets: {mp if mp is not None else 'None'}\n"
                        )
                        ans = _agent_call(agent, prompt)
                        st.markdown(ans)
                # After agent run, load latest incident to set as active (if produced)
                latest = _try_load_latest_incident_from_disk()
                if latest:
                    _set_active_incident(latest, source="agent_adm_tool")
                st.session_state["agent_messages"].append({"role": "assistant", "content": ans})
                st.rerun()

    with b2:
        if st.button("Generate mitigation (from active Incident Card)", use_container_width=True):
            if not active:
                st.warning("Load an Incident Card first.")
            else:
                user_msg = "Generate mitigation guidance for the active incident." 
                st.session_state["agent_messages"].append({"role": "user", "content": user_msg})
                with st.chat_message("user"):
                    st.markdown(user_msg)
                with st.chat_message("assistant"):
                    with st.spinner("Agent is retrieving manuals evidence and writing guidance…"):
                        compact = _compact_incident_text(active)
                        prompt = (
                            "An Incident Card is provided below. Follow your workflow strictly: "
                            "extract incident signals, formulate retrieval queries based on the incident, call cgm_retrieve, "
                            "and write operator-ready mitigation guidance with chunk citations.\n\n"
                            f"{compact}"
                        )
                        ans = _agent_call(agent, prompt)
                        st.markdown(ans)
                st.session_state["agent_messages"].append({"role": "assistant", "content": ans})
                st.rerun()

    with b3:
        if st.button("Clear agent chat", use_container_width=True):
            st.session_state["agent_messages"] = []
            st.rerun()

    st.divider()

    # -----------------------------
    # Agent chat (incident + question)
    # -----------------------------
    st.markdown("### Agent chat (Incident + Question)")
    st.caption(
        "Ask a specific question about the active incident. "
        "If no Incident Card is loaded, switch to Manuals-only mode."
    )

    _render_chat("agent_messages")

    q = st.chat_input("Ask the agent about this incident…")
    if q:
        if not st.session_state.get("active_incident_card"):
            st.warning("No active Incident Card loaded. Load one first or switch to Manuals-only mode.")
        else:
            st.session_state["agent_messages"].append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)

            compact = _compact_incident_text(st.session_state["active_incident_card"])
            prompt = (
                "You are given an Incident Card and an administrator question.\n"
                "Follow your workflow. Use manuals evidence (cgm_retrieve) and cite chunk IDs.\n"
                "Do NOT call adm_analyze_pcap unless the admin explicitly requests re-analysis.\n\n"
                f"{compact}\n"
                f"[ADMIN_QUESTION]\n{q}\n"
            ).strip()

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    compact = _compact_incident_text(st.session_state["active_incident_card"])
                    prompt = (
                        "You are given an Incident Card and an administrator question. "
                        "Follow your workflow strictly. Use local manuals evidence via cgm_retrieve "
                        "and cite chunk IDs. If manuals are insufficient and web search is enabled, you may use it as last resort.\n\n"
                        f"{compact}\n"
                        f"[ADMIN_QUESTION]\n{q}\n"
                    )
                    ans = _agent_call(agent, prompt)
                    st.markdown(ans)

            st.session_state["agent_messages"].append({"role": "assistant", "content": ans})
