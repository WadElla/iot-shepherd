import pyshark
import pandas as pd

# Full list of features
ALL_FEATURES = [
    "frame.time", "ip.src_host", "ip.dst_host",
    "arp.dst.proto_ipv4", "arp.opcode", "arp.hw.size", "arp.src.proto_ipv4",
    "icmp.checksum", "icmp.seq_le", "icmp.transmit_timestamp", "icmp.unused",
    "http.file_data", "http.content_length", "http.request.uri.query", 
    "http.request.method", "http.referer", "http.request.full_uri",
    "http.request.version", "http.response", "http.tls_port",
    "tcp.ack", "tcp.ack_raw", "tcp.checksum", "tcp.connection.fin", 
    "tcp.connection.rst", "tcp.connection.syn", "tcp.connection.synack",
    "tcp.dstport", "tcp.flags", "tcp.flags.ack", "tcp.len", "tcp.options", 
    "tcp.payload", "tcp.seq", "tcp.srcport",
    "udp.port", "udp.stream", "udp.time_delta",
    "dns.qry.name", "dns.qry.name.len", "dns.qry.qu", "dns.qry.type",
    "dns.retransmission", "dns.retransmit_request", "dns.retransmit_request_in",
    "mqtt.conack.flags", "mqtt.conflag.cleansess", "mqtt.conflags",
    "mqtt.hdrflags", "mqtt.len", "mqtt.msg_decoded_as", "mqtt.msg",
    "mqtt.msgtype", "mqtt.proto_len", "mqtt.protoname", "mqtt.topic",
    "mqtt.topic_len", "mqtt.ver",
    "mbtcp.len", "mbtcp.trans_id", "mbtcp.unit_id",
    "Attack_type", "Attack_label"  # Added attack type and attack label
]

def extract_pcap_features(pcap_file, attack_type):
    
    cap = pyshark.FileCapture(pcap_file)  # Capture all packet types
    feature_list = []
    
    attack_label = 0 if attack_type.lower() == "normal" else 11  #11 is backdoor attack type # Here, you specifiy the type of attack type based on the dataset. 

    for packet in cap:
        features = {key: 0 for key in ALL_FEATURES} 

        try:
            features["frame.time"] = packet.sniff_time.timestamp()
            features["Attack_type"] = attack_type  # Assign attack type label
            features["Attack_label"] = attack_label  # Assign attack label

            # IP Layer
            if hasattr(packet, "ip"):
                features["ip.src_host"] = packet.ip.src
                features["ip.dst_host"] = packet.ip.dst
            
            # ARP Layer
            if hasattr(packet, "arp"):
                features["arp.dst.proto_ipv4"] = getattr(packet.arp, "dst_proto_ipv4", 0)
                features["arp.opcode"] = getattr(packet.arp, "opcode", 0)
                features["arp.hw.size"] = getattr(packet.arp, "hw_size", 0)
                features["arp.src.proto_ipv4"] = getattr(packet.arp, "src_proto_ipv4", 0)

            # ICMP Layer
            if hasattr(packet, "icmp"):
                features["icmp.checksum"] = getattr(packet.icmp, "checksum", 0)
                features["icmp.seq_le"] = getattr(packet.icmp, "seq_le", 0)
                features["icmp.transmit_timestamp"] = getattr(packet.icmp, "transmit_timestamp", 0)
                features["icmp.unused"] = getattr(packet.icmp, "unused", 0)

            # HTTP Layer
            if hasattr(packet, "http"):
                features["http.file_data"] = getattr(packet.http, "file_data", 0)
                features["http.content_length"] = getattr(packet.http, "content_length", 0)
                features["http.request.uri.query"] = getattr(packet.http, "request_uri_query", 0)
                features["http.request.method"] = getattr(packet.http, "request_method", 0)
                features["http.referer"] = getattr(packet.http, "referer", 0)
                features["http.request.full_uri"] = getattr(packet.http, "request_full_uri", 0)
                features["http.request.version"] = getattr(packet.http, "request_version", 0)
                features["http.response"] = getattr(packet.http, "response", 0)
                features["http.tls_port"] = getattr(packet.http, "tls_port", 0)

            # TCP Layer
            if hasattr(packet, "tcp"):
                features["tcp.ack"] = getattr(packet.tcp, "ack", 0)
                features["tcp.ack_raw"] = getattr(packet.tcp, "ack_raw", 0)
                features["tcp.checksum"] = getattr(packet.tcp, "checksum", 0)
                features["tcp.connection.fin"] = getattr(packet.tcp, "connection_fin", 0)
                features["tcp.connection.rst"] = getattr(packet.tcp, "connection_rst", 0)
                features["tcp.connection.syn"] = getattr(packet.tcp, "connection_syn", 0)
                features["tcp.connection.synack"] = getattr(packet.tcp, "connection_synack", 0)
                features["tcp.dstport"] = getattr(packet.tcp, "dstport", 0)
                features["tcp.flags"] = getattr(packet.tcp, "flags", 0)
                features["tcp.flags.ack"] = getattr(packet.tcp, "flags_ack", 0)
                features["tcp.len"] = getattr(packet.tcp, "len", 0)
                features["tcp.options"] = getattr(packet.tcp, "options", 0)
                features["tcp.payload"] = getattr(packet.tcp, "payload", 0)
                features["tcp.seq"] = getattr(packet.tcp, "seq", 0)
                features["tcp.srcport"] = getattr(packet.tcp, "srcport", 0)

            # UDP Layer
            if hasattr(packet, "udp"):
                features["udp.port"] = getattr(packet.udp, "port", 0)
                features["udp.stream"] = getattr(packet.udp, "stream", 0)
                features["udp.time_delta"] = getattr(packet.udp, "time_delta", 0)

            # DNS Layer
            if hasattr(packet, "dns"):
                features["dns.qry.name"] = getattr(packet.dns, "qry_name", 0)
                features["dns.qry.name.len"] = getattr(packet.dns, "qry_name_len", 0)
                features["dns.qry.qu"] = getattr(packet.dns, "qry_qu", 0)
                features["dns.qry.type"] = getattr(packet.dns, "qry_type", 0)

            # MQTT Layer
            if hasattr(packet, "mqtt"):
                features["mqtt.conack.flags"] = getattr(packet.mqtt, "conack_flags", 0)
                features["mqtt.topic"] = getattr(packet.mqtt, "topic", 0)
                features["mqtt.msgtype"] = getattr(packet.mqtt, "msgtype", 0)

            # Modbus TCP Layer
            if hasattr(packet, "mbtcp"):
                features["mbtcp.len"] = getattr(packet.mbtcp, "len", 0)
                features["mbtcp.trans_id"] = getattr(packet.mbtcp, "trans_id", 0)
                features["mbtcp.unit_id"] = getattr(packet.mbtcp, "unit_id", 0)

            feature_list.append(features)

        except Exception as e:
            print(f"Error processing packet: {e}")
            continue

    cap.close()
    return pd.DataFrame(feature_list)

# Usage
pcap_file = "Backdoor_attack.pcap" # Location of the pcap file
attack_type = "Backdoor"  # Specify the attack type
df = extract_pcap_features(pcap_file, attack_type)

# Save to CSV
df.to_csv("test.csv", index=False)
print("Feature extraction complete. Saved to 'pcap_features.csv'.")

# Display first few rows
print(df.head(12))