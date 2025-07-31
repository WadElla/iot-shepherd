import os
import re
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../rag_module')))
from query import query_rag

BERT_REPORT_PATH = "bert_traffic_report.txt"
OUTPUT_PATH = "integration/remediation_guidance.txt"


def extract_attack_sections(report_path):
    """Parses report and extracts attacks, metadata, and percentages."""
    with open(report_path, 'r') as f:
        lines = f.readlines()

    sections = {}
    current_attack = None
    collecting = None

    for i, line in enumerate(lines):
        line = line.strip()

        # Detect start of an attack block (but skip [Normal])
        if line.startswith("[") and not line.startswith("[Normal]"):
            current_attack = line.strip("[]")
            sections[current_attack] = {
                "ip_pairs": [],
                "mqtt_topics": [],
                "dns_queries": [],
                "modbus_ids": [],
                "count": 0,
                "percent": 0.0
            }

        elif current_attack:
            if line.startswith("- Count:"):
                # Example: - Count: 1330 (5.34%)
                match = re.match(r"- Count:\s+(\d+)\s+\(([\d\.]+)%\)", line)
                if match:
                    sections[current_attack]["count"] = int(match.group(1))
                    sections[current_attack]["percent"] = float(match.group(2))

            elif "Unique IP Pairs:" in line:
                collecting = "ip_pairs"
            elif "MQTT Topics:" in line:
                collecting = "mqtt_topics"
            elif "DNS Queries:" in line:
                collecting = "dns_queries"
            elif "Modbus Unit IDs:" in line:
                collecting = "modbus_ids"
            elif line.startswith("-") and collecting:
                val = line.strip("- ").strip()
                if val:
                    sections[current_attack][collecting].append(val)

    return sections


def build_automated_prompt(attack_type, metadata):
    prompt = f"What steps should an IoT administrator take to mitigate {attack_type} threats"

    if metadata["ip_pairs"]:
        prompt += f" involving IPs like {', '.join(metadata['ip_pairs'][:3])}"
    if metadata["mqtt_topics"]:
        prompt += f", MQTT topics: {', '.join(metadata['mqtt_topics'])}"
    if metadata["dns_queries"]:
        prompt += f", and DNS queries: {', '.join(metadata['dns_queries'])}"
    if metadata["modbus_ids"]:
        prompt += f", Modbus Unit IDs: {', '.join(metadata['modbus_ids'])}"

    return prompt + "?"


def run_automated_mode(sections):
    print("🤖 Running in AUTO mode...\n")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, 'w') as out:
        for attack, meta in sections.items():
            prompt = build_automated_prompt(attack, meta)
            print(f"🛡️  Querying for {attack}: {prompt}")
            response = query_rag(prompt)
            out.write(f"### [{attack}]\nPrompt: {prompt}\nLLM Answer: {response.strip()}\n\n")

    print(f"\n✅ All responses saved to: {OUTPUT_PATH}")


def run_interactive_mode(sections):
    attacks = sorted(sections.items(), key=lambda x: -x[1].get("percent", 0))
    while True:
        print("\n🧠 Detected Attack Types (sorted by severity):")
        for idx, (attack, meta) in enumerate(attacks, start=1):
            print(f"{idx}. {attack} ({meta['percent']:.2f}%)")

        try:
            choice = int(input("\nSelect an attack to query (or 0 to exit): "))
            if choice == 0:
                break
            selected_attack, metadata = attacks[choice - 1]
        except (ValueError, IndexError):
            print("❌ Invalid selection. Try again.")
            continue

        print("\n📌 Available Metadata:")
        print(f"- IP Pairs: {', '.join(metadata['ip_pairs'][:3]) or 'None'}")
        print(f"- MQTT Topics: {', '.join(metadata['mqtt_topics']) or 'None'}")
        print(f"- DNS Queries: {', '.join(metadata['dns_queries']) or 'None'}")
        print(f"- Modbus IDs: {', '.join(metadata['modbus_ids']) or 'None'}")

        question = input("\nType your question: ").strip()
        if not question.endswith("?"):
            question += "?"

        enrich = input("➕ Include metadata in the query? (y/n): ").lower().startswith("y")
        if enrich:
            if metadata["ip_pairs"]:
                question += f" IPs: {', '.join(metadata['ip_pairs'][:3])}."
            if metadata["mqtt_topics"]:
                question += f" MQTT Topics: {', '.join(metadata['mqtt_topics'])}."
            if metadata["dns_queries"]:
                question += f" DNS Queries: {', '.join(metadata['dns_queries'])}."
            if metadata["modbus_ids"]:
                question += f" Modbus Unit IDs: {', '.join(metadata['modbus_ids'])}."

        print(f"\n🔍 Querying LLM with: {question}")
        response = query_rag(question)
        print(f"\n🤖 LLM Response:\n{response.strip()}")

        save = input("💾 Save this answer? (y/n): ").strip().lower() == "y"
        if save:
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            with open(OUTPUT_PATH, 'a') as out:
                out.write(f"### [{selected_attack}]\nPrompt: {question}\nLLM Answer: {response.strip()}\n\n")

        again = input("\n🔁 Ask about another attack? (y/n): ").strip().lower()
        if again != "y":
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="Run in fully automated mode.")
    args = parser.parse_args()

    if not os.path.exists(BERT_REPORT_PATH):
        print(f"❌ Missing file: {BERT_REPORT_PATH}")
        return

    sections = extract_attack_sections(BERT_REPORT_PATH)
    if not sections:
        print("✅ No actionable anomalies found.")
        return

    if args.auto:
        run_automated_mode(sections)
    else:
        run_interactive_mode(sections)


if __name__ == "__main__":
    main()
