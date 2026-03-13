from __future__ import annotations

"""
IoT Shepherd - ADM Feature Extractor (PCAP -> CSV)

Goal
- Extract a stable, model-compatible set of packet-level / protocol features from a PCAP,
  then write them to CSV for the BERT ADM inference stage.
- This extractor is intentionally aligned with the working style used in Revelation:
  - defaults for missing fields are 0 (matching common training-time preprocessing)
  - does NOT inject ground-truth labels for uploaded PCAPs (to avoid misleading evaluation)
  - focuses on deterministic, reproducible outputs

Notes
- Requires tshark to be installed (PyShark backend).
- For unlabeled operational use, keep include_labels=False (default).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import pyshark


# Fields that are expensive / noisy and are dropped before model text construction in the ADM.
# We do not extract them by default.
_HEAVY_FIELDS = {
    "tcp.payload",
    "http.file_data",
    "arp.src.proto_ipv4",
}

# A robust, model-friendly feature set,
# excluding label columns by default.
FEATURES: List[str] = [
    # time + endpoints
    "frame.time",
    "ip.src_host",
    "ip.dst_host",

    # ARP
    "arp.dst.proto_ipv4",
    "arp.opcode",
    "arp.hw.size",

    # ICMP
    "icmp.checksum",
    "icmp.seq_le",
    "icmp.transmit_timestamp",
    "icmp.unused",

    # HTTP (excluding http.file_data by default)
    "http.content_length",
    "http.request.uri.query",
    "http.request.method",
    "http.referer",
    "http.request.full_uri",
    "http.request.version",
    "http.response",
    "http.tls_port",

    # TCP (excluding tcp.payload by default)
    "tcp.ack",
    "tcp.ack_raw",
    "tcp.checksum",
    "tcp.connection.fin",
    "tcp.connection.rst",
    "tcp.connection.syn",
    "tcp.connection.synack",
    "tcp.dstport",
    "tcp.flags",
    "tcp.flags.ack",
    "tcp.len",
    "tcp.options",
    "tcp.seq",
    "tcp.srcport",

    # UDP
    "udp.port",
    "udp.stream",
    "udp.time_delta",

    # DNS
    "dns.qry.name",
    "dns.qry.name.len",
    "dns.qry.qu",
    "dns.qry.type",
    "dns.retransmission",
    "dns.retransmit_request",
    "dns.retransmit_request_in",

    # MQTT
    "mqtt.conack.flags",
    "mqtt.conflag.cleansess",
    "mqtt.conflags",
    "mqtt.hdrflags",
    "mqtt.len",
    "mqtt.msg_decoded_as",
    "mqtt.msg",
    "mqtt.msgtype",
    "mqtt.proto_len",
    "mqtt.protoname",
    "mqtt.topic",
    "mqtt.topic_len",
    "mqtt.ver",

    # Modbus/TCP
    "mbtcp.len",
    "mbtcp.trans_id",
    "mbtcp.unit_id",
]


@dataclass(frozen=True)
class ExtractorOptions:
    max_packets: Optional[int] = None
    tshark_display_filter: Optional[str] = None
    include_labels: bool = False
    attack_type: str = "Unknown"   # only used when include_labels=True
    attack_label: int = 0          # only used when include_labels=True


def _safe_getattr(obj, attr: str, default=0):
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def _pkt_time_ts(pkt) -> float:
    try:
        return float(pkt.sniff_time.timestamp())
    except Exception:
        return 0.0


def _extract_one_packet(pkt) -> Dict[str, Any]:
    """Extract a feature dict from a single PyShark packet using safe fallbacks."""
    f: Dict[str, Any] = {k: 0 for k in FEATURES}

    # frame.time
    f["frame.time"] = _pkt_time_ts(pkt)

    # IP endpoints
    try:
        if hasattr(pkt, "ip"):
            f["ip.src_host"] = _safe_getattr(pkt.ip, "src", 0)
            f["ip.dst_host"] = _safe_getattr(pkt.ip, "dst", 0)
    except Exception:
        pass

    # ARP
    try:
        if hasattr(pkt, "arp"):
            f["arp.dst.proto_ipv4"] = _safe_getattr(pkt.arp, "dst_proto_ipv4", 0)
            f["arp.opcode"] = _safe_getattr(pkt.arp, "opcode", 0)
            f["arp.hw.size"] = _safe_getattr(pkt.arp, "hw_size", 0)
    except Exception:
        pass

    # ICMP
    try:
        if hasattr(pkt, "icmp"):
            f["icmp.checksum"] = _safe_getattr(pkt.icmp, "checksum", 0)
            f["icmp.seq_le"] = _safe_getattr(pkt.icmp, "seq_le", 0)
            f["icmp.transmit_timestamp"] = _safe_getattr(pkt.icmp, "transmit_timestamp", 0)
            f["icmp.unused"] = _safe_getattr(pkt.icmp, "unused", 0)
    except Exception:
        pass

    # HTTP
    try:
        if hasattr(pkt, "http"):
            f["http.content_length"] = _safe_getattr(pkt.http, "content_length", 0)
            f["http.request.uri.query"] = _safe_getattr(pkt.http, "request_uri_query", 0)
            f["http.request.method"] = _safe_getattr(pkt.http, "request_method", 0)
            f["http.referer"] = _safe_getattr(pkt.http, "referer", 0)
            f["http.request.full_uri"] = _safe_getattr(pkt.http, "request_full_uri", 0)
            f["http.request.version"] = _safe_getattr(pkt.http, "request_version", 0)
            f["http.response"] = _safe_getattr(pkt.http, "response", 0)
            f["http.tls_port"] = _safe_getattr(pkt.http, "tls_port", 0)
    except Exception:
        pass

    # TCP
    try:
        if hasattr(pkt, "tcp"):
            f["tcp.ack"] = _safe_getattr(pkt.tcp, "ack", 0)
            f["tcp.ack_raw"] = _safe_getattr(pkt.tcp, "ack_raw", 0)
            f["tcp.checksum"] = _safe_getattr(pkt.tcp, "checksum", 0)
            f["tcp.connection.fin"] = _safe_getattr(pkt.tcp, "connection_fin", 0)
            f["tcp.connection.rst"] = _safe_getattr(pkt.tcp, "connection_rst", 0)
            f["tcp.connection.syn"] = _safe_getattr(pkt.tcp, "connection_syn", 0)
            f["tcp.connection.synack"] = _safe_getattr(pkt.tcp, "connection_synack", 0)
            f["tcp.dstport"] = _safe_getattr(pkt.tcp, "dstport", 0)
            f["tcp.flags"] = _safe_getattr(pkt.tcp, "flags", 0)
            f["tcp.flags.ack"] = _safe_getattr(pkt.tcp, "flags_ack", 0)
            f["tcp.len"] = _safe_getattr(pkt.tcp, "len", 0)
            f["tcp.options"] = _safe_getattr(pkt.tcp, "options", 0)
            f["tcp.seq"] = _safe_getattr(pkt.tcp, "seq", 0)
            f["tcp.srcport"] = _safe_getattr(pkt.tcp, "srcport", 0)
    except Exception:
        pass

    # UDP
    try:
        if hasattr(pkt, "udp"):
            f["udp.port"] = _safe_getattr(pkt.udp, "port", 0)
            f["udp.stream"] = _safe_getattr(pkt.udp, "stream", 0)
            f["udp.time_delta"] = _safe_getattr(pkt.udp, "time_delta", 0)
    except Exception:
        pass

    # DNS
    try:
        if hasattr(pkt, "dns"):
            f["dns.qry.name"] = _safe_getattr(pkt.dns, "qry_name", 0)
            f["dns.qry.name.len"] = _safe_getattr(pkt.dns, "qry_name_len", 0)
            f["dns.qry.qu"] = _safe_getattr(pkt.dns, "qry_qu", 0)
            f["dns.qry.type"] = _safe_getattr(pkt.dns, "qry_type", 0)
            f["dns.retransmission"] = _safe_getattr(pkt.dns, "retransmission", 0)
            f["dns.retransmit_request"] = _safe_getattr(pkt.dns, "retransmit_request", 0)
            f["dns.retransmit_request_in"] = _safe_getattr(pkt.dns, "retransmit_request_in", 0)
    except Exception:
        pass

    # MQTT
    try:
        if hasattr(pkt, "mqtt"):
            f["mqtt.conack.flags"] = _safe_getattr(pkt.mqtt, "conack_flags", 0)
            f["mqtt.conflag.cleansess"] = _safe_getattr(pkt.mqtt, "conflag_cleansess", 0)
            f["mqtt.conflags"] = _safe_getattr(pkt.mqtt, "conflags", 0)
            f["mqtt.hdrflags"] = _safe_getattr(pkt.mqtt, "hdrflags", 0)
            f["mqtt.len"] = _safe_getattr(pkt.mqtt, "len", 0)
            f["mqtt.msg_decoded_as"] = _safe_getattr(pkt.mqtt, "msg_decoded_as", 0)
            f["mqtt.msg"] = _safe_getattr(pkt.mqtt, "msg", 0)
            f["mqtt.msgtype"] = _safe_getattr(pkt.mqtt, "msgtype", 0)
            f["mqtt.proto_len"] = _safe_getattr(pkt.mqtt, "proto_len", 0)
            f["mqtt.protoname"] = _safe_getattr(pkt.mqtt, "protoname", 0)
            f["mqtt.topic"] = _safe_getattr(pkt.mqtt, "topic", 0)
            f["mqtt.topic_len"] = _safe_getattr(pkt.mqtt, "topic_len", 0)
            f["mqtt.ver"] = _safe_getattr(pkt.mqtt, "ver", 0)
    except Exception:
        pass

    # Modbus/TCP
    try:
        if hasattr(pkt, "mbtcp"):
            f["mbtcp.len"] = _safe_getattr(pkt.mbtcp, "len", 0)
            f["mbtcp.trans_id"] = _safe_getattr(pkt.mbtcp, "trans_id", 0)
            f["mbtcp.unit_id"] = _safe_getattr(pkt.mbtcp, "unit_id", 0)
    except Exception:
        pass

    return f


def extract_features_from_pcap(
    pcap_path: Path,
    out_csv: Path,
    max_packets: int | None = None,
    tshark_display_filter: str | None = None,
    include_labels: bool = False,
    attack_type: str = "Unknown",
    attack_label: int = 0,
) -> Dict[str, Any]:
    """
    Extract features from a PCAP into a CSV.

    Args:
      pcap_path: input PCAP path on disk
      out_csv: CSV output path
      max_packets: cap packets processed (for demos / speed)
      tshark_display_filter: optional tshark display filter
      include_labels: ONLY for synthetic/debug runs (adds Attack_type/Attack_label columns)
      attack_type/attack_label: values used if include_labels=True

    Returns:
      stats dict for UI/logging
    """
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP not found: {pcap_path}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cap = pyshark.FileCapture(
        str(pcap_path),
        keep_packets=False,
        use_json=True,
        include_raw=False,
        display_filter=tshark_display_filter,
    )

    rows: List[Dict[str, Any]] = []
    total = 0

    try:
        for pkt in cap:
            total += 1
            row = _extract_one_packet(pkt)
            if include_labels:
                row["Attack_type"] = attack_type
                row["Attack_label"] = int(attack_label)
            rows.append(row)
            if max_packets and total >= max_packets:
                break
    finally:
        try:
            cap.close()
        except Exception:
            pass

    df = pd.DataFrame(rows)

    # Match common preprocessing used in training/inference
    df = df.fillna(0)

    # Ensure stable column order
    ordered_cols = list(FEATURES) + (["Attack_type", "Attack_label"] if include_labels else [])
    df = df.reindex(columns=ordered_cols)

    df.to_csv(out_csv, index=False)

    return {
        "pcap_path": str(pcap_path),
        "out_csv": str(out_csv),
        "packets_processed": int(total),
        "fields_extracted": int(len(df.columns)),
        "rows_written": int(len(df)),
        "display_filter": tshark_display_filter,
        "labels_included": bool(include_labels),
    }
