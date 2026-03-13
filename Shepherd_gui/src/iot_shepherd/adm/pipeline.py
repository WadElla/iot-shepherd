from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from .feature_extractor import extract_features_from_pcap
from .anomaly_detector import run_adm_inference
from ..utils.paths import new_run_paths
from ..utils.io import write_upload_to_path
from ..config import validate_adm_model_dir


def analyze_pcap_file(
    uploaded_file,
    runs_dir: Path,
    model_dir: Path,
    max_packets: int | None = None,
) -> Dict[str, Any]:
    """ADM pipeline: PCAP upload -> features -> BERT inference -> incident card + artifacts.

    Notes:
    - We DO NOT manipulate tshark paths here (PyShark handles backend discovery).
    - We DO validate that the saved model exists before inference.
    """
    rp = new_run_paths(runs_dir=runs_dir, original_filename=getattr(uploaded_file, "name", "capture.pcap"))

    # Persist uploaded PCAP into the run folder
    write_upload_to_path(uploaded_file, rp.pcap_path)

    # Pointer for agent/tools fallback
    try:
        (Path(runs_dir) / "latest_pcap.txt").write_text(str(rp.pcap_path), encoding="utf-8")
    except Exception:
        pass

    # Feature extraction (PyShark)
    fe_stats = extract_features_from_pcap(
        pcap_path=rp.pcap_path,
        out_csv=rp.features_csv,
        max_packets=max_packets,
    )

    # Validate model presence (fixes "model not in right place" issues)
    ok, msg = validate_adm_model_dir(model_dir)
    if not ok:
        raise FileNotFoundError(msg)

    # Inference + reporting
    incident = run_adm_inference(
        features_csv=rp.features_csv,
        model_dir=model_dir,
        results_csv=rp.results_csv,
        report_txt=rp.report_txt,
        incident_json=rp.incident_json,
    )

    incident["run_id"] = rp.run_id
    incident["pcap_path"] = str(rp.pcap_path)
    incident["feature_extraction"] = fe_stats

    # Convenience pointer for agentic mode / UI
    try:
        (Path(runs_dir) / "latest_incident.json").write_text(
            json.dumps(incident, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    return incident
