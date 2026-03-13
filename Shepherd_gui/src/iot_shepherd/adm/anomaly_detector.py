from __future__ import annotations

"""
IoT Shepherd — ADM (Anomaly Detection Module)

This module mirrors the working Revelation-style BERT inference loop and report generation,
and additionally emits an Incident Card JSON artifact for the Shepherd Agent.

Key properties:
- Uses fillna(0) to match the training-time preprocessing convention.
- Builds a per-row textual representation from feature:value pairs for BERT.
- Produces:
  (1) predictions CSV
  (2) human-readable interpretation report TXT
  (3) Incident Card JSON (summary + top attacks + endpoint metadata + artifact paths)
- Computes evaluation metrics ONLY when ground-truth labels are meaningfully present.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from collections import Counter
import json
from datetime import datetime, timezone

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


# -----------------------------
# Class mapping (Revelation-consistent)
# -----------------------------
ATTACK_LABEL_MAP: Dict[int, str] = {
    0: "Normal",
    1: "MITM",
    2: "Fingerprinting",
    3: "Ransomware",
    4: "Uploading",
    5: "SQL Injection",
    6: "DDoS_HTTP",
    7: "DDoS_TCP",
    8: "Password",
    9: "Port Scanning",
    10: "Vul Scanner",
    11: "Backdoor",
    12: "XSS",
    13: "DDoS_UDP",
    14: "DDoS_ICMP",
}


# -----------------------------
# Attack narratives (Revelation-consistent)
# (name, category, severity, behavior, implication, recommendation)
# -----------------------------
ATTACK_CATEGORIES: Dict[int, Tuple[str, str, str, str, str, str]] = {
    1: (
        "MITM",
        "Interception / Impersonation",
        "High",
        "Man-in-the-Middle (MITM) attacks attempt to intercept or alter communications between devices, often through ARP spoofing or rogue gateways.",
        "Sensitive data, including credentials or configuration commands, can be silently stolen or manipulated.",
        "Isolate the affected subnet, inspect ARP tables and DHCP leases, and enforce certificate validation for encrypted traffic.",
    ),
    2: (
        "Fingerprinting",
        "Reconnaissance",
        "Low",
        "The attacker actively profiles connected devices or services to determine software versions, open ports, and vulnerabilities.",
        "This is often the prelude to a targeted exploit or vulnerability scan.",
        "Restrict unnecessary service exposure, use device fingerprint obfuscation where possible, and log abnormal scan behavior.",
    ),
    3: (
        "Ransomware",
        "Malware Deployment",
        "Critical",
        "Malicious code designed to encrypt device data or lock access until a ransom is paid, possibly spreading laterally across the network.",
        "Can disrupt industrial systems or critical home infrastructure, causing major data loss or service outages.",
        "Immediately isolate affected hosts, restore from clean backups, and assess lateral movement. Report incidents if needed.",
    ),
    4: (
        "Uploading",
        "Unauthorized File Transfer",
        "Medium",
        "Traffic patterns consistent with unexpected file uploads, possibly exploiting unsecured endpoints (e.g., FTP, HTTP PUT).",
        "Could result in remote code execution or storage of malicious tools on IoT gateways.",
        "Audit upload endpoints, enforce authentication, inspect logs for unusual payloads, and restrict executable file transfers.",
    ),
    5: (
        "SQL Injection",
        "Injection Exploits",
        "High",
        "Malicious SQL statements are injected into application inputs to exfiltrate data or manipulate backend databases.",
        "Can compromise user records, leak credentials, or allow remote access to control systems.",
        "Sanitize all input fields, apply least privilege DB access, and use parameterized queries or ORMs.",
    ),
    6: (
        "DDoS_HTTP",
        "Volumetric Denial-of-Service",
        "High",
        "Large volumes of HTTP requests target web servers or REST APIs to exhaust resources.",
        "May lead to downtime of dashboards, control panels, or external APIs used by IoT systems.",
        "Deploy rate limiting, reverse proxies, or cloud-based DDoS protection. Identify attacker IPs for blacklisting.",
    ),
    7: (
        "DDoS_TCP",
        "Connection Flooding",
        "High",
        "TCP SYN/ACK floods or spoofed session establishment to overwhelm TCP stacks.",
        "Often affects routers, embedded Linux devices, or gateway firewalls, causing memory exhaustion.",
        "Enable SYN cookies, tune OS TCP parameters, and monitor for high-rate IPs with excessive half-open connections.",
    ),
    8: (
        "Password",
        "Credential Brute Force",
        "Medium",
        "Numerous login attempts with different passwords, targeting SSH, Telnet, HTTP, or MQTT services.",
        "Can lead to device hijacking or access to administrative interfaces if default or weak passwords are used.",
        "Enforce strong credentials, lock accounts after failures, and monitor authentication logs for abuse patterns.",
    ),
    9: (
        "Port Scanning",
        "Service Discovery",
        "Low",
        "Sequential or randomized probes across TCP/UDP ports to discover active services.",
        "Indicates an attacker is mapping the network to find exploitable targets.",
        "Block unsolicited scanning at the perimeter, use port knockers or segmentation to protect sensitive devices.",
    ),
    10: (
        "Vul Scanner",
        "Exploit Scanning",
        "Medium",
        "Automated tools (e.g., Nmap NSE, Nikto, Nessus) scan for known vulnerabilities or misconfigurations.",
        "These tools often identify weak firmware, default credentials, or exposed admin panels.",
        "Patch exposed services, disable unused ports, and fingerprint the scanning tool for possible attribution.",
    ),
    11: (
        "Backdoor",
        "Persistent Access",
        "Critical",
        "Hidden channels or implants enabling an attacker to regain access even after reboots or remediation.",
        "Can allow full device control and long-term compromise of the IoT environment.",
        "Hunt for unrecognized binaries, reverse shells, or custom beaconing behavior. Reflash firmware if needed.",
    ),
    12: (
        "XSS",
        "Web Exploits",
        "Medium",
        "Malicious JavaScript is injected into web-based IoT dashboards or user-facing interfaces.",
        "Can hijack sessions, modify settings, or exfiltrate sensitive values from browser sessions.",
        "Sanitize all inputs and outputs on web UIs. Apply Content Security Policies (CSP) and browser hardening.",
    ),
    13: (
        "DDoS_UDP",
        "Amplification / Flooding",
        "High",
        "High-speed UDP floods, often spoofed, target vulnerable services like NTP, DNS, or SSDP.",
        "Can saturate bandwidth, disrupt video streams, or bring down IoT controllers.",
        "Disable unused UDP services, rate-limit critical ports, and monitor for abnormal packet size bursts.",
    ),
    14: (
        "DDoS_ICMP",
        "Ping Flood / Tunnel Abuse",
        "High",
        "High-frequency ICMP traffic including echo requests or ICMP tunneling for covert channels.",
        "Impacts CPU usage, routing tables, or can be used to bypass firewalls.",
        "Rate-limit ICMP, block unneeded types (e.g., redirect, router discovery), and alert on anomalous RTTs.",
    ),
}


# -----------------------------
# Utilities
# -----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon / macOS Metal
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _dominant_non_normal(pred_counts: Counter) -> Tuple[Optional[int], int]:
    candidates = [(cid, cnt) for cid, cnt in pred_counts.items() if cid != 0]
    if not candidates:
        return None, 0
    cid, cnt = max(candidates, key=lambda x: x[1])
    return cid, cnt


def _safe_int_label(x) -> Optional[int]:
    """Best-effort coercion of labels to int; returns None when coercion fails."""
    try:
        if pd.isna(x):
            return None
        # Handles floats like 0.0 / 11.0 and strings like "11"
        return int(float(str(x).strip()))
    except Exception:
        return None


def _build_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors the Revelation-style row->text conversion:
    - Drop noisy/high-cardinality fields if present.
    - Rename Attack_label -> Label when present.
    - Exclude Label/Attack_type and endpoint fields from the text input.
    """
    # Drop irrelevant or high-cardinality fields (if present)
    columns_to_remove = ["tcp.payload", "http.file_data", "arp.src.proto_ipv4"]
    df = df.drop(columns=[c for c in columns_to_remove if c in df.columns], errors="ignore")

    # Normalize label column name if the dataset uses Attack_label
    if "Attack_label" in df.columns and "Label" not in df.columns:
        df = df.rename(columns={"Attack_label": "Label"})

    # Exclude label and endpoint identifiers from the text representation
    exclude_for_text = {"Label", "Attack_type", "ip.src_host", "ip.dst_host"}
    text_cols = [c for c in df.columns if c not in exclude_for_text]

    def row_to_text(row) -> str:
        parts: List[str] = []
        for c in text_cols:
            v = row.get(c, 0)
            parts.append(f"{c}: {v}")
        return " ".join(parts)

    df["text"] = df.apply(row_to_text, axis=1)
    return df


def _clean_meta(v) -> str:
    """
    UI-facing cleanup for endpoint/protocol metadata fields.
    Note: We keep training-compatible fillna(0) in the dataframe;
    here we only convert obvious "missing" markers for display.
    """
    if v is None:
        return "N/A"
    if isinstance(v, float) and pd.isna(v):
        return "N/A"
    # Many extracted string fields become 0 after fillna(0)
    if v == 0 or v == 0.0 or v == "0":
        return "N/A"
    s = str(v).strip()
    if s == "" or s.lower() == "nan" or s.lower() == "none":
        return "N/A"
    return s


def _labels_are_meaningful(df: pd.DataFrame) -> bool:
    """
    Decide whether we should compute evaluation metrics.
    For typical PCAP uploads, labels are absent; for dataset-style CSVs,
    labels and Attack_type are usually present.
    """
    if "Label" not in df.columns:
        return False

    # Coerce labels; keep only those that look like integers
    coerced = df["Label"].apply(_safe_int_label)
    valid = coerced.dropna()
    if valid.empty:
        return False

    # If Attack_type exists, it is very likely a dataset CSV (not a PCAP upload)
    if "Attack_type" in df.columns:
        return True

    # Otherwise, require at least 2 distinct labels to avoid misleading reports
    # (e.g., a column full of zeros accidentally created)
    return valid.nunique() >= 2


# -----------------------------
# Main entrypoint
# -----------------------------
def run_adm_inference(
    features_csv: Path,
    model_dir: Path,
    results_csv: Path,
    report_txt: Path,
    incident_json: Path,
    device: str | None = None,
) -> Dict[str, Any]:
    """
    Run BERT inference (Revelation-consistent) and emit:
    - predictions CSV
    - interpretation report TXT
    - Incident Card JSON
    """
    if not features_csv.exists():
        raise FileNotFoundError(f"Features CSV not found: {features_csv}")

    if not model_dir.exists():
        raise FileNotFoundError(
            f"ADM model directory not found: {model_dir}\n"
            f"Place your trained BERT model under: {model_dir}"
        )

    # Load dataset
    df = pd.read_csv(features_csv, low_memory=False)

    # Training convention: fill missing values with 0
    df = df.fillna(0)
    df = _build_text(df)

    # Determine whether we have meaningful labels
    compute_eval = _labels_are_meaningful(df)
    true_labels: Optional[List[int]] = None
    if compute_eval:
        coerced = df["Label"].apply(_safe_int_label)
        true_labels = [int(x) for x in coerced.tolist() if x is not None]

    # Device selection
    if device is None:
        dev = _device_auto()
    else:
        dev = torch.device(device)

    # Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(str(model_dir))
    model = BertForSequenceClassification.from_pretrained(str(model_dir))
    model.to(dev)
    model.eval()

    predictions: List[int] = []
    results: List[Dict[str, Any]] = []

    # Prediction loop
    for _, row in df.iterrows():
        inputs = tokenizer(
            str(row["text"]),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits

        pred = int(logits.argmax().item())
        predictions.append(pred)

        rec: Dict[str, Any] = {
            "text": row["text"],
            "predicted_label": pred,
            "predicted_attack_type": ATTACK_LABEL_MAP.get(pred, "Unknown"),
            "ip.src_host": _clean_meta(row.get("ip.src_host")),
            "ip.dst_host": _clean_meta(row.get("ip.dst_host")),
            "mqtt.topic": _clean_meta(row.get("mqtt.topic")),
            "dns.qry.name": _clean_meta(row.get("dns.qry.name")),
            "mbtcp.unit_id": _clean_meta(row.get("mbtcp.unit_id")),
            "frame.time": _clean_meta(row.get("frame.time")),
        }

        # Only attach true label when evaluation is meaningful
        if compute_eval:
            rec["true_label"] = _safe_int_label(row.get("Label"))

        results.append(rec)

    # Save prediction CSV
    result_df = pd.DataFrame(results)
    result_df.sort_values(by="predicted_attack_type", inplace=True)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(results_csv, index=False)

    # Optional evaluation
    eval_block: Dict[str, Any] = {}
    if compute_eval and true_labels is not None:
        # Align true labels length with predictions (in case of None filtering)
        # If we filtered Nones above, fall back to per-row coercion instead
        coerced_full = df["Label"].apply(_safe_int_label).tolist()
        aligned_true = [0 if x is None else int(x) for x in coerced_full]

        try:
            acc = accuracy_score(aligned_true, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                aligned_true,
                predictions,
                average="weighted",
                zero_division=0,
            )
            rep = classification_report(
                aligned_true,
                predictions,
                labels=list(ATTACK_LABEL_MAP.keys()),
                target_names=[ATTACK_LABEL_MAP[i] for i in range(15)],
                zero_division=0,
            )
            eval_block = {
                "accuracy": float(acc),
                "precision_weighted": float(precision),
                "recall_weighted": float(recall),
                "f1_weighted": float(f1),
                "classification_report": rep,
            }
        except Exception as e:
            # Never fail ADM output due to metric formatting issues
            eval_block = {"warning": f"Evaluation skipped due to error: {e}"}

    # Summary stats
    pred_counts = Counter(predictions)
    total = len(predictions)
    normal_count = int(pred_counts.get(0, 0))
    anomaly_count = int(total - normal_count)
    anomaly_percent = float((anomaly_count / total * 100.0) if total else 0.0)

    dom_id, dom_count = _dominant_non_normal(pred_counts)
    dom_name = ATTACK_LABEL_MAP.get(dom_id, None) if dom_id is not None else None
    dom_pct = float((dom_count / anomaly_count * 100.0) if anomaly_count and dom_count else 0.0)

    # Build report TXT (Revelation-style richness)
    report_lines: List[str] = []
    report_lines.append("==== BERT-Based IoT Traffic Interpretation Report ====\n")
    report_lines.append(f"Generated at: {utc_now_iso()}")
    report_lines.append(f"Total traffic samples analyzed: {total}")
    report_lines.append(f"Normal traffic: {normal_count} ({(normal_count / total):.2%})" if total else "Normal traffic: 0")
    report_lines.append(f"Anomalous traffic: {anomaly_count} ({anomaly_percent:.2f}%)\n")

    if compute_eval and eval_block:
        report_lines.append("=== Evaluation (labels present) ===")
        if "warning" in eval_block:
            report_lines.append(eval_block["warning"])
        else:
            report_lines.append(f"Accuracy: {eval_block['accuracy']:.4f}")
            report_lines.append(f"Precision (weighted): {eval_block['precision_weighted']:.4f}")
            report_lines.append(f"Recall (weighted): {eval_block['recall_weighted']:.4f}")
            report_lines.append(f"F1 (weighted): {eval_block['f1_weighted']:.4f}\n")
            report_lines.append("Classification Report:\n")
            report_lines.append(eval_block["classification_report"])
        report_lines.append("")

    report_lines.append("=== Category-wise Breakdown and Analysis ===\n")
    for class_id in range(15):
        count = int(pred_counts.get(class_id, 0))
        if count == 0:
            continue

        if class_id == 0:
            pct_total = float((count / total * 100.0) if total else 0.0)
            report_lines.append(f"[Normal]")
            report_lines.append(f"- Count: {count} ({pct_total:.2f}% of total)")
            report_lines.append("- No action needed. Traffic appears regular.\n")
            continue

        pct_anom = float((count / anomaly_count * 100.0) if anomaly_count else 0.0)
        # Narrative fields
        name, category, severity, behavior, implication, recommendation = ATTACK_CATEGORIES.get(
            class_id,
            (ATTACK_LABEL_MAP.get(class_id, "Unknown"), "Unknown", "Unknown", "", "", ""),
        )
        report_lines.append(f"[{name}]")
        report_lines.append(
            f"This section summarizes how {name} activity manifested in the analyzed traffic. "
            f"A total of {count} samples ({pct_anom:.2f}% of anomalous traffic) were classified under this type."
        )
        report_lines.append(f"- Count: {count} ({pct_anom:.2f}% of anomalies)")
        report_lines.append(f"- Category: {category}")
        report_lines.append(f"- Severity: {severity}")
        report_lines.append(f"- Behavior: {behavior}")
        report_lines.append(f"- Risk: {implication}")
        report_lines.append(f"- Recommendation: {recommendation}\n")

    report_lines.append("=== Interpretation Summary ===")
    if anomaly_count == 0:
        report_lines.append("No anomalies detected. The traffic appears clean.")
    elif anomaly_percent < 5:
        report_lines.append("Low anomaly rate detected. Possibly benign or low-risk activities.")
    elif anomaly_percent < 20:
        report_lines.append("Moderate anomaly level. Potential early-stage attacks.")
    else:
        report_lines.append("High anomaly volume detected. Immediate investigation recommended.")

    if dom_name:
        report_lines.append(f"\nDominant Detected Attack: {dom_name} ({dom_count} samples, {dom_pct:.2f}% of anomalies)")

    # Endpoint metadata per attack (for report + incident card)
    report_lines.append("\n=== Unique Endpoints and Metadata per Attack ===\n")
    endpoints_by_attack: Dict[str, List[str]] = {}

    for class_id in sorted(pred_counts.keys()):
        if class_id == 0:
            continue

        name = ATTACK_LABEL_MAP.get(class_id, "Unknown")
        attack_df = result_df[result_df["predicted_label"] == class_id]

        unique_conns = set()
        unique_mqtt = set()
        unique_dns = set()
        unique_modbus = set()

        for _, r in attack_df.iterrows():
            src = r.get("ip.src_host", "N/A")
            dst = r.get("ip.dst_host", "N/A")
            if isinstance(src, str) and isinstance(dst, str) and src != "N/A" and dst != "N/A":
                unique_conns.add(f"{src} → {dst}")

            t = r.get("mqtt.topic", "N/A")
            if isinstance(t, str) and t != "N/A":
                unique_mqtt.add(t)

            q = r.get("dns.qry.name", "N/A")
            if isinstance(q, str) and q != "N/A":
                unique_dns.add(q)

            u = r.get("mbtcp.unit_id", "N/A")
            if isinstance(u, str) and u != "N/A":
                unique_modbus.add(u)

        capped_pairs = list(sorted(unique_conns))[:10]
        endpoints_by_attack[name] = capped_pairs

        report_lines.append(f"[{name}]")
        report_lines.append("This section summarizes all unique endpoints and protocol metadata observed for this category.")
        report_lines.append(f"- Unique IP Pairs: {len(unique_conns)}")
        for pair in sorted(unique_conns):
            report_lines.append(f"  - {pair}")
        if unique_mqtt:
            report_lines.append(f"- MQTT Topics: {len(unique_mqtt)}")
            for topic in sorted(unique_mqtt):
                report_lines.append(f"  - {topic}")
        if unique_dns:
            report_lines.append(f"- DNS Queries: {len(unique_dns)}")
            for query in sorted(unique_dns):
                report_lines.append(f"  - {query}")
        if unique_modbus:
            report_lines.append(f"- Modbus Unit IDs: {len(unique_modbus)}")
            for unit_id in sorted(unique_modbus):
                report_lines.append(f"  - {unit_id}")
        report_lines.append("")

    report_txt.parent.mkdir(parents=True, exist_ok=True)
    report_txt.write_text("\n".join(report_lines), encoding="utf-8")

    # Build Incident Card JSON
    top_attacks: List[Dict[str, Any]] = []
    for cid, cnt in pred_counts.most_common():
        if cid == 0:
            continue
        name = ATTACK_LABEL_MAP.get(cid, "Unknown")
        category = None
        severity = None
        if cid in ATTACK_CATEGORIES:
            _, category, severity, _, _, _ = ATTACK_CATEGORIES[cid]

        top_attacks.append(
            {
                "attack_type": name,
                "count": int(cnt),
                "percent_of_anomalies": float((cnt / anomaly_count * 100.0) if anomaly_count else 0.0),
                "category": category,
                "severity": severity,
            }
        )

    incident: Dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "summary": {
            "total_samples": int(total),
            "normal_samples": int(normal_count),
            "anomalous_samples": int(anomaly_count),
            "anomaly_percent": float(anomaly_percent),
            "dominant_attack": dom_name,
            "dominant_attack_count": int(dom_count),
            "dominant_attack_percent_of_anomalies": float(dom_pct),
        },
        "top_attacks": top_attacks[:8],
        "endpoints_by_attack": endpoints_by_attack,
        "artifacts": {
            "results_csv": str(results_csv),
            "report_txt": str(report_txt),
            "incident_json": str(incident_json),
        },
        "evaluation": eval_block if (compute_eval and eval_block) else None,
        "notes": "Incident Card summarizes ADM outputs. Model text preprocessing matches training (fillna(0)).",
    }

    incident_json.parent.mkdir(parents=True, exist_ok=True)
    incident_json.write_text(json.dumps(incident, indent=2), encoding="utf-8")

    return incident