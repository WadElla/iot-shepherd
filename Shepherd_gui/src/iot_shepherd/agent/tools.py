from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from agno.tools import tool

from ..config import AppSettings, validate_adm_model_dir
from ..adm.feature_extractor import extract_features_from_pcap
from ..adm.anomaly_detector import run_adm_inference
from ..cgm.retrieval import retrieve_context
from ..utils.paths import new_run_paths


@tool(
    name="cgm_retrieve",
    description="Retrieve top-k evidence chunks from the local IoT manuals knowledge base (Chroma). Returns a kb_context JSON payload.",
    show_result=False,
    requires_user_input=False,
    requires_confirmation=False,
    stop_after_tool_call=False,
)
def cgm_retrieve(query: str, k: int = 5) -> str:
    q = (query or "").strip()
    # Robustly parse k across LLM/tool-calling drift.
    # Some models may emit nested objects like {"type":"number","value":"5"}.
    kk = 5
    try:
        if k is None:
            kk = 5
        elif isinstance(k, dict):
            cand = k.get("value", None)
            kk = int(cand) if cand is not None else 5
        else:
            kk = int(k)
    except Exception:
        kk = 5
    if kk < 1:
        kk = 1

    if not q:
        return json.dumps(
            {
                "type": "kb_context",
                "ok": False,
                "error": "EMPTY_QUERY",
                "message": "Empty query provided. Provide a non-empty retrieval query string.",
                "question": "",
                "k": kk,
                "collection": None,
                "top_k": [],
                "chunk_count": 0,
                "total_chunks": 0,
            },
            ensure_ascii=False,
        )


    settings = AppSettings()
    payload = retrieve_context(
        query=q,
        chroma_dir=settings.chroma_dir,
        embed_model=settings.embed_model,
        k=kk,
    )

    results = payload.get("results", []) or []
    top_k = []
    for r in results:
        top_k.append(
            {
                "id": r.get("id"),
                "source": r.get("source"),
                "page": r.get("page"),
                "score": r.get("score"),
                "excerpt": (r.get("excerpt") or "").strip(),
            }
        )

    package = {
        "type": "kb_context",
        "ok": bool(payload.get("ok", False)),
        "error": payload.get("error"),
        "message": None,
        "total_chunks": payload.get("total_chunks"),
        "question": payload.get("question", q),
        "k": kk,
        "collection": payload.get("collection"),
        "top_k": top_k,
        "chunk_count": len(top_k),
    }

    if not package["ok"]:
        if package["error"] == "KB_EMPTY":
            package["message"] = "Knowledge Base is empty (0 chunks). Upload and index manuals first."
        elif package["error"] == "NO_MATCHES":
            package["message"] = "No chunks matched the query. Refine the query and retry."
        else:
            package["message"] = "Retrieval failed. Refine the query and retry."

    return json.dumps(package, ensure_ascii=False)


@tool(
    name="adm_analyze_pcap",
    description="Run end-to-end ADM on a PCAP path (on disk) and return an incident_card JSON payload.",
    show_result=False,
    requires_user_input=False,
    requires_confirmation=False,
    stop_after_tool_call=False,
)
def adm_analyze_pcap(pcap_path: str = "", max_packets: Optional[int] = None) -> str:
    settings = AppSettings()

    ok_model, msg = validate_adm_model_dir(settings.adm_model_dir)
    if not ok_model:
        return json.dumps(
            {"type": "incident_card", "ok": False, "error": "MODEL_MISSING", "message": msg, "pcap_path": pcap_path},
            ensure_ascii=False,
        )

    # IMPORTANT: To prevent LLMs from hallucinating random PCAP paths, we support a trusted override.
    # If an override is set by the UI (agent page), we always use it, ignoring the LLM-provided path.
    override_ptr = Path(settings.runs_dir) / "agent_pcap_override.txt"
    src: Optional[Path] = None
    if override_ptr.exists():
        try:
            cand = Path(override_ptr.read_text(encoding="utf-8", errors="ignore").strip())
            if cand.exists() and cand.is_file():
                src = cand
        except Exception:
            src = None

    if src is None:
        raw = (pcap_path or "").strip()
        src = Path(raw) if raw else None

    # Fallback to latest PCAP pointer if provided path is missing/invalid
    if src is None or (src.exists() and src.is_dir()) or (src is not None and not src.exists()):
        ptr = Path(settings.runs_dir) / "latest_pcap.txt"
        if ptr.exists():
            try:
                cand = Path(ptr.read_text(encoding="utf-8").strip())
                if cand.exists() and cand.is_file():
                    src = cand
            except Exception:
                pass

    if src is None:
        return json.dumps(
            {
                "type": "incident_card",
                "ok": False,
                "error": "MISSING_PCAP_PATH",
                "message": "No PCAP path provided. Provide a valid file path or run ADM in the UI first.",
                "pcap_path": pcap_path,
            },
            ensure_ascii=False,
        )

    if not src.exists():
        return json.dumps(
            {"type": "incident_card", "ok": False, "error": "PCAP_NOT_FOUND", "message": f"PCAP not found: {src}", "pcap_path": str(src)},
            ensure_ascii=False,
        )

    if src.is_dir():
        return json.dumps(
            {"type": "incident_card", "ok": False, "error": "PCAP_IS_DIRECTORY", "message": f"PCAP path is a directory: {src}", "pcap_path": str(src)},
            ensure_ascii=False,
        )

    rp = new_run_paths(runs_dir=Path(settings.runs_dir), original_filename=src.name)

    try:
        shutil.copy2(src, rp.pcap_path)
    except Exception as e:
        return json.dumps(
            {"type": "incident_card", "ok": False, "error": "PCAP_COPY_FAILED", "message": f"{type(e).__name__}: {e}", "pcap_path": str(src)},
            ensure_ascii=False,
        )

    try:
        fe_stats = extract_features_from_pcap(pcap_path=rp.pcap_path, out_csv=rp.features_csv, max_packets=max_packets)
    except Exception as e:
        return json.dumps(
            {"type": "incident_card", "ok": False, "error": "FEATURE_EXTRACTION_FAILED", "message": f"{type(e).__name__}: {e}", "pcap_path": str(rp.pcap_path)},
            ensure_ascii=False,
        )

    try:
        incident = run_adm_inference(
            features_csv=rp.features_csv,
            model_dir=Path(settings.adm_model_dir),
            results_csv=rp.results_csv,
            report_txt=rp.report_txt,
            incident_json=rp.incident_json,
        )
    except Exception as e:
        return json.dumps(
            {"type": "incident_card", "ok": False, "error": "ADM_INFERENCE_FAILED", "message": f"{type(e).__name__}: {e}", "pcap_path": str(rp.pcap_path)},
            ensure_ascii=False,
        )

    incident["ok"] = True
    incident["run_id"] = rp.run_id
    incident["pcap_path"] = str(rp.pcap_path)
    incident["feature_extraction"] = fe_stats

    try:
        (Path(settings.runs_dir) / "latest_pcap.txt").write_text(str(rp.pcap_path), encoding="utf-8")
        (Path(settings.runs_dir) / "latest_incident.json").write_text(json.dumps(incident, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    return json.dumps(incident, ensure_ascii=False)
