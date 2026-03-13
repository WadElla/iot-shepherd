from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import uuid
from .io import safe_filename

@dataclass
class RunPaths:
    run_id: str
    run_dir: Path
    pcap_dir: Path
    features_dir: Path
    outputs_dir: Path
    pcap_path: Path
    features_csv: Path
    results_csv: Path
    report_txt: Path
    incident_json: Path

def new_run_paths(runs_dir: Path, original_filename: str) -> RunPaths:
    run_id = uuid.uuid4().hex[:12]
    run_dir = runs_dir / run_id
    pcap_dir = run_dir / "pcap"
    features_dir = run_dir / "features"
    outputs_dir = run_dir / "outputs"

    pcap_path = pcap_dir / safe_filename(original_filename)
    features_csv = features_dir / "features.csv"
    results_csv = outputs_dir / "adm_predictions.csv"
    report_txt = outputs_dir / "bert_traffic_report.txt"
    incident_json = outputs_dir / "incident_card.json"

    for d in (pcap_dir, features_dir, outputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        pcap_dir=pcap_dir,
        features_dir=features_dir,
        outputs_dir=outputs_dir,
        pcap_path=pcap_path,
        features_csv=features_csv,
        results_csv=results_csv,
        report_txt=report_txt,
        incident_json=incident_json,
    )
