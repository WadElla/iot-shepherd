import time
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification

# Load the saved model and tokenizer
model = BertForSequenceClassification.from_pretrained('./saved_model')
tokenizer = BertTokenizer.from_pretrained('./saved_model')

# Device identification (GPU if available, otherwise CPU/MPS)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

# Load and preprocess the test dataset
test_df = pd.read_csv('pcap_features.csv', low_memory=False)

# Drop columns that are not used during inference and rename as needed
columns_to_remove = ['ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4', 'tcp.payload', 'http.file_data']
test_df.drop(columns=columns_to_remove, inplace=True)
test_df.rename(columns={'attack_label': 'label'}, inplace=True)

# Combine the remaining features into a single text string for BERT
test_df['text'] = test_df.apply(
    lambda row: ' '.join([f"{col}: {row[col]}" for col in test_df.columns if col not in ['label', 'attack_type']]),
    axis=1
)
true_labels = test_df['label'].tolist()

print("Unique True Labels:", set(true_labels))

def predict_and_evaluate(test_df, model, tokenizer, true_labels, output_file):
    predictions = []
    results = []
    
    # Start timing the inference process
    start_time = time.time()
    
    for index, row in test_df.iterrows():
        # Tokenize the input text
        inputs = tokenizer(row['text'], return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
        predictions.append(predicted_class_id)
        
        results.append({
            'text': row['text'],
            'predicted_label': predicted_class_id
        })
    
    end_time = time.time()
    total_time = end_time - start_time
    num_samples = len(test_df)
    latency_per_sample = total_time / num_samples if num_samples > 0 else float('inf')
    throughput = num_samples / total_time if total_time > 0 else 0
    
    # Save predictions to CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    
    # Evaluate predictions
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"\nTotal Inference Time: {total_time:.4f} seconds")
    print(f"Number of Samples Processed: {num_samples}")
    print(f"Average Inference Latency per Sample: {latency_per_sample:.6f} seconds")
    print(f"Inference Throughput (samples/second): {throughput:.2f}")

if __name__ == "__main__":
    predict_and_evaluate(test_df, model, tokenizer, true_labels, 'anomaly_detection_test_results.csv')
