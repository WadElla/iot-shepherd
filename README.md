# IoT Shepherd

**IoT Shepherd** is a unified framework for secure and intelligent IoT management. It combines anomaly detection on IoT network traffic with context-aware question answering using Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG).

This project includes tools for:
- Embedding IoT manuals into a vector database
- Answering IoT-related admin queries with contextual accuracy
- Detecting anomalous traffic using a fine-tuned BERT model
- Automatically triggering remediation prompts using the LLM system

---

## 📁 Directory Structure

```
IoT-Shepherd/
│
├── README.md
├── requirements.txt
│
├── rag_module/
│   ├── populate_database.py
│   ├── query.py
│   ├── performance.py
│   ├── get_embedding_function.py
│   ├── data/
│   └── chroma/
│
├── Bert/
│   ├── feature_extraction.py
│   ├── bert_multiclass_train.py
│   ├── bert_multiclass_test.py
│   ├── performance_metrics.py
│   ├── saved_model/
│   └── bert_traffic_report.txt
│
├── integration/
│   ├── remediation_orchestrator.py
│   └── remediation_guidance.txt
```

---

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/iot-shepherd.git
cd iot-shepherd
```

2. **Create a virtual environment (optional)**
```bash
conda create -n shepherd python=3.10
conda activate shepherd
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

4. **Start Ollama models**
```bash
ollama run llama3.2
ollama run nomic-embed-text
```

---

## 🚀 Usage

### 1. Populate the Vector Database with IoT Manuals
```bash
cd rag_module/
python populate_database.py --reset
```

### 2. Ask a Contextual Question
```bash
python query.py "How do I reset a smart plug that won’t connect to WiFi?"
```

### 3. Extract Features from PCAP File
```bash
cd ../Bert/
python feature_extraction.py  # Edit PCAP file and attack type inside the script
```

### 4. Train the Anomaly Detection Model
```bash
python bert_multiclass_train.py
```

### 5. Generate Report from Predictions
```bash
python bert_multiclass_test.py
```

### 6. Generate Remediation Advice from Detected Threats

**Interactive mode:**
```bash
cd ../integration/
python remediation_orchestrator.py
```

**Automated mode:**
```bash
python remediation_orchestrator.py --auto
```

---
