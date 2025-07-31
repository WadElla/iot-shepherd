import os
import time
import psutil
import pandas as pd
from docx import Document
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates
import subprocess
from transformers import AutoTokenizer
import torch

from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM as Ollama
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

# ----------------------------
# NVML GPU Monitoring
# ----------------------------
def initialize_nvml():
    """Initialize NVML."""
    nvmlInit()


def shutdown_nvml():
    """Shutdown NVML."""
    nvmlShutdown()


def get_gpu_usage_nvml(gpu_index=0):
    """Retrieve GPU memory and utilization using NVML."""
    handle = nvmlDeviceGetHandleByIndex(gpu_index)  # Assumes GPU 0; adjust index if needed
    memory_info = nvmlDeviceGetMemoryInfo(handle)
    utilization = nvmlDeviceGetUtilizationRates(handle)
    return {
        "gpu_memory_used": memory_info.used / (1024 ** 2),  # Convert bytes to MB
        "gpu_memory_total": memory_info.total / (1024 ** 2),
        "gpu_utilization": utilization.gpu,  # GPU utilization percentage
    }


# ----------------------------
# NVIDIA-SMI GPU Monitoring
# ----------------------------
def get_gpu_usage_nvidia_smi():
    """Retrieve GPU usage metrics using NVIDIA-SMI with robust parsing."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        lines = result.strip().split("\n")
        for line in lines:
            fields = line.split(",")
            if len(fields) == 3:
                memory_used = float(fields[0].strip())
                memory_total = float(fields[1].strip())
                utilization = float(fields[2].strip())
                return {
                    "gpu_memory_used": memory_used,
                    "gpu_memory_total": memory_total,
                    "gpu_utilization": utilization,
                }
    except Exception as e:
        print("Error while retrieving GPU usage with NVIDIA-SMI:", e)
        return {"gpu_memory_used": 0, "gpu_memory_total": 0, "gpu_utilization": 0}


# ----------------------------
# PyTorch GPU Monitoring
# ----------------------------
def get_gpu_usage_torch(device=0):
    """Retrieve GPU utilization using PyTorch."""
    if torch.cuda.is_available():
        memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # MB
        max_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
        gpu_utilization = memory_reserved / max_memory * 100 if max_memory > 0 else 0
        return {"gpu_memory_reserved": memory_reserved, "gpu_utilization": gpu_utilization}
    else:
        return {"gpu_memory_reserved": 0, "gpu_utilization": 0}


# ----------------------------
# Core Functionality
# ----------------------------
def load_questions_from_docx(file_path):
    """Load questions and answers from a Word document."""
    document = Document(file_path)
    qa_dict = {}
    for paragraph in document.paragraphs:
        if ":" in paragraph.text:
            question, answer = paragraph.text.split(":", 1)
            qa_dict[question.strip()] = answer.strip()
    return qa_dict


def calculate_response_size(response_text, model_name="gpt2"):
    """Calculate the total size of a response in bytes and token count."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer.tokenize(response_text)
        token_sizes = [len(token.encode("utf-8")) for token in tokens]  # Size of each token in bytes
        total_size = sum(token_sizes)  # Total size in bytes
        return total_size, len(tokens)  # Return total size and token count
    except Exception as e:
        print(f"Error while loading tokenizer for model {model_name}: {e}")
        return 0, 0


def evaluate_answer(generated_answer, reference_answer):
    """Evaluate the generated answers using various metrics."""
    reference_tokens = reference_answer.split()
    generated_tokens = generated_answer.split()

    # BERTScore
    P, R, F1 = bert_score([generated_answer], [reference_answer], lang="en", verbose=False)

    # ROUGE Score
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge_scorer_obj.score(reference_answer, generated_answer)
    
    # BLEU Score with Smoothing
    smoothie = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothie)

    # METEOR Score
    meteor = meteor_score([reference_tokens], generated_tokens)
    
    return {
        "bert_Precision": P.mean().item(),
        "bert_Recall": R.mean().item(),
        "bert_F1": F1.mean().item(),
        "rouge1": rouge_scores['rouge1'].fmeasure,
        "rouge2": rouge_scores['rouge2'].fmeasure,
        "rougeL": rouge_scores['rougeL'].fmeasure,
        "bleu": bleu_score,
        "meteor": meteor
    }


def main():
    # Initialize NVML
    initialize_nvml()

    # Load questions and answers from the document
    questions_answers = load_questions_from_docx("Evaluation/troubleshoot.docx")
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    model = Ollama(model="llava")  

    # Lists to store evaluation results and system metrics
    results = []
    system_metrics = []

    for question, reference_answer in questions_answers.items():
        process = psutil.Process()

        # Measure system performance for "with context"
        memory_before = process.memory_info().rss / (1024 ** 2)  # Memory in MB
        gpu_before_nvml = get_gpu_usage_nvml()
        gpu_before_smi = get_gpu_usage_nvidia_smi()
        gpu_before_torch = get_gpu_usage_torch()
        cpu_times_before = process.cpu_times()
        psutil_cpu_usage_before = psutil.cpu_percent(interval=None)  # psutil CPU usage
        start_time = time.time()

        # Execute the "with context" scenario
        results_db = db.similarity_search_with_score(question, k=3)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results_db])
        prompt_with_context = f"""
        Answer the question based only on the following context:

        {context_text}

        ---

        Answer the question based on the above context: {question}
        """
        response_text = model.invoke(prompt_with_context)

        end_time = time.time()
        memory_after = process.memory_info().rss / (1024 ** 2)
        cpu_times_after = process.cpu_times()
        psutil_cpu_usage_after = psutil.cpu_percent(interval=None)  # psutil CPU usage
        gpu_after_nvml = get_gpu_usage_nvml()
        gpu_after_smi = get_gpu_usage_nvidia_smi()
        gpu_after_torch = get_gpu_usage_torch()

        # Calculate metrics for "with context"
        elapsed_time_with_context = end_time - start_time
        memory_diff_with_context = memory_after - memory_before
        total_cpu_time_with_context = (cpu_times_after.user - cpu_times_before.user) + \
                                      (cpu_times_after.system - cpu_times_before.system)
        estimated_cpu_usage_with_context = (total_cpu_time_with_context / elapsed_time_with_context) * 100
        psutil_cpu_usage_with_context = (psutil_cpu_usage_after + psutil_cpu_usage_before) / 2
        rag_response_size, rag_token_count = calculate_response_size(response_text)

        gpu_memory_used_diff_nvml = gpu_after_nvml["gpu_memory_used"] - gpu_before_nvml["gpu_memory_used"]
        gpu_utilization_diff_nvml = gpu_after_nvml["gpu_utilization"] - gpu_before_nvml["gpu_utilization"]

        gpu_memory_used_diff_smi = gpu_after_smi["gpu_memory_used"] - gpu_before_smi["gpu_memory_used"]
        gpu_utilization_diff_smi = gpu_after_smi["gpu_utilization"] - gpu_before_smi["gpu_utilization"]

        gpu_utilization_torch_diff = gpu_after_torch["gpu_utilization"] - gpu_before_torch["gpu_utilization"]

        rag_scores = evaluate_answer(response_text, reference_answer)

        # Measure system performance for "without context"
        memory_before = process.memory_info().rss / (1024 ** 2)
        gpu_before_nvml = get_gpu_usage_nvml()
        gpu_before_smi = get_gpu_usage_nvidia_smi()
        gpu_before_torch = get_gpu_usage_torch()
        cpu_times_before = process.cpu_times()
        psutil_cpu_usage_before = psutil.cpu_percent(interval=None)
        start_time = time.time()

        # Execute the "without context" scenario
        response_text_no_context = model.invoke(f"Please provide an answer to the following question: {question}")

        end_time = time.time()
        memory_after = process.memory_info().rss / (1024 ** 2)
        cpu_times_after = process.cpu_times()
        psutil_cpu_usage_after = psutil.cpu_percent(interval=None)
        gpu_after_nvml = get_gpu_usage_nvml()
        gpu_after_smi = get_gpu_usage_nvidia_smi()
        gpu_after_torch = get_gpu_usage_torch()

        # Calculate metrics for "without context"
        elapsed_time_no_context = end_time - start_time
        memory_diff_no_context = memory_after - memory_before
        total_cpu_time_no_context = (cpu_times_after.user - cpu_times_before.user) + \
                                    (cpu_times_after.system - cpu_times_before.system)
        estimated_cpu_usage_no_context = (total_cpu_time_no_context / elapsed_time_no_context) * 100
        psutil_cpu_usage_no_context = (psutil_cpu_usage_after + psutil_cpu_usage_before) / 2
        no_context_response_size, no_context_token_count = calculate_response_size(response_text_no_context)

        gpu_memory_used_diff_nvml_no_context = gpu_after_nvml["gpu_memory_used"] - gpu_before_nvml["gpu_memory_used"]
        gpu_utilization_diff_nvml_no_context = gpu_after_nvml["gpu_utilization"] - gpu_before_nvml["gpu_utilization"]

        gpu_memory_used_diff_smi_no_context = gpu_after_smi["gpu_memory_used"] - gpu_before_smi["gpu_memory_used"]
        gpu_utilization_diff_smi_no_context = gpu_after_smi["gpu_utilization"] - gpu_before_smi["gpu_utilization"]

        gpu_utilization_torch_diff_no_context = gpu_after_torch["gpu_utilization"] - gpu_before_torch["gpu_utilization"]

        no_context_scores = evaluate_answer(response_text_no_context, reference_answer)

        # Store results and metrics
        result = {
            "question": question,
            "reference_answer": reference_answer,
            "rag_answer": response_text,
            "no_context_answer": response_text_no_context,
            **{f"rag_{key}": value for key, value in rag_scores.items()},
            **{f"no_context_{key}": value for key, value in no_context_scores.items()}
        }
        results.append(result)

        metric_entry = {
            "question": question,
            "rag_memory_diff": memory_diff_with_context,
            "rag_elapsed_time": elapsed_time_with_context,
            "rag_estimated_cpu_usage": estimated_cpu_usage_with_context,
            "rag_psutil_cpu_usage": psutil_cpu_usage_with_context,
            "rag_response_size": rag_response_size,
            "rag_token_count": rag_token_count,
            "rag_gpu_memory_used_nvml": gpu_memory_used_diff_nvml,
            "rag_gpu_utilization_nvml": gpu_utilization_diff_nvml,
            "rag_gpu_memory_used_smi": gpu_memory_used_diff_smi,
            "rag_gpu_utilization_smi": gpu_utilization_diff_smi,
            "rag_gpu_utilization_torch_diff": gpu_utilization_torch_diff,
            "no_context_memory_diff": memory_diff_no_context,
            "no_context_elapsed_time": elapsed_time_no_context,
            "no_context_estimated_cpu_usage": estimated_cpu_usage_no_context,
            "no_context_psutil_cpu_usage": psutil_cpu_usage_no_context,
            "no_context_response_size": no_context_response_size,
            "no_context_token_count": no_context_token_count,
            "no_context_gpu_memory_used_nvml": gpu_memory_used_diff_nvml_no_context,
            "no_context_gpu_utilization_nvml": gpu_utilization_diff_nvml_no_context,
            "no_context_gpu_memory_used_smi": gpu_memory_used_diff_smi_no_context,
            "no_context_gpu_utilization_smi": gpu_utilization_diff_smi_no_context,
            "no_context_gpu_utilization_torch_diff": gpu_utilization_torch_diff_no_context,
        }
        system_metrics.append(metric_entry)

    # Save evaluation results
    #df_results = pd.DataFrame(results)
    #df_results.to_csv("Evaluation/evaluation_troubleshoot.csv", index=False)

    # Save system metrics
    df_metrics = pd.DataFrame(system_metrics)
    #df_metrics.to_csv("Evaluation/system_metrics_gemma2.csv", index=False)

    # Calculate and save averages
    averages = df_metrics.mean(numeric_only=True).to_dict()
    pd.DataFrame([averages]).to_csv("Evaluation/system_metrics_averages_llava.csv", index=False)

    # Shutdown NVML
    shutdown_nvml()

    print("Evaluation complete. Results and metrics saved.")


if __name__ == "__main__":
    main()
