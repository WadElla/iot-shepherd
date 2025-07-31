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

> **Note:** This project requires [Ollama](https://ollama.com) for running local LLMs and embedding models. To install Ollama, follow the official guide here: [https://ollama.com/download](https://ollama.com/download). It supports macOS, Linux, and Windows (via WSL).

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

4. **Start Ollama Server and Pull Required Models**

First, start the Ollama server (must be running in the background):

```bash
ollama serve
```

Then, in a separate terminal, pull the required models:

```bash
ollama pull llama3:2
ollama pull nomic-embed-text
```

✅ Once the server is running and models are pulled, you're ready to run the project. The Python code will automatically connect to the local Ollama API.

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

### 6. **Generate Remediation Advice from Detected Threats**

The system supports two modes of remediation guidance, both powered by context-aware retrieval from IoT manuals:

* **Interactive Mode**: You will be prompted to manually select from the list of detected attacks. For each selected threat, the system uses a Retrieval-Augmented Generation (RAG) pipeline to provide actionable remediation instructions based on the detected metadata.

* **Automated Mode**: The system analyzes the `bert_traffic_report.txt` file, identifies all detected attack types, ranks them by their percentage of total traffic, and auto-generates remediation guidance for each in descending order of urgency.

#### Interactive Mode:

```bash
cd ../integration/
python remediation_orchestrator.py
```

#### Automated Mode:

```bash
cd ../integration/
python remediation_orchestrator.py --auto
```

> ✅ Use `--auto` to automatically process all detected threats based on priority, or run without flags to choose interactively at runtime.

---
