import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize

# Step 1: Load and Prepare the Dataset
df = pd.read_csv('final_dataset.csv')

# Drop specific columns you don't want to use
columns_to_remove = ['ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4', 'tcp.payload', 'http.file_data']
df.drop(columns=columns_to_remove, inplace=True)

# Rename 'Attack_label' to 'Label'
df.rename(columns={'Attack_label': 'Label'}, inplace=True)

# Combine relevant features (excluding Label and Attack_type) into a single string
df['text'] = df.apply(lambda row: ' '.join([f"{col}: {row[col]}" for col in df.columns if col not in ['Label', 'Attack_type']]), axis=1)

# Convert 'Label' to integer
df['label'] = df['Label'].astype(int)

# Step 2: Split the Dataset into Training and Testing Sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Step 3: Tokenization using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the data for both training and testing sets
train_encodings = tokenizer(train_df['text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')
test_encodings = tokenizer(test_df['text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')

train_labels = train_df['label'].tolist()
test_labels = test_df['label'].tolist()

# Custom Dataset Class for PyTorch
class IoTDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create the training and testing datasets
train_dataset = IoTDataset(train_encodings, train_labels)
test_dataset = IoTDataset(test_encodings, test_labels)

# Step 4: Device Identification (using GPU if available)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Step 5: Model Setup (BERT for sequence classification)
num_classes = len(df['label'].unique())  # Number of unique classes
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
model.to(device)

# Step 6: Define Custom Metrics for Multiclass
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Overall accuracy
    accuracy = accuracy_score(labels, preds)
    
    # Weighted precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    
    # Class-specific metrics
    report = classification_report(labels, preds, output_dict=True)
    
    print("\nClassification Report:")
    print(classification_report(labels, preds))
    
    # Save class-specific metrics to CSV
    class_report_df = pd.DataFrame(report).transpose()
    class_report_df.to_csv("performance/classification_report.csv", index=True)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_report': report
    }

# Step 7: Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    fp16=torch.cuda.is_available()
)

# Step 8: Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Step 9: Training
trainer.train()

# Step 10: Evaluation
eval_result = trainer.evaluate()

# Save overall evaluation metrics to CSV
eval_metrics_df = pd.DataFrame([eval_result])
eval_metrics_df.to_csv("performance/evaluation_metrics.csv", index=False)

print(f"Evaluation results: {eval_result}")

# Step 11: Visualization

# 1. Performance Metrics Over Epochs
train_logs = [log for log in trainer.state.log_history if 'eval_accuracy' in log]
epochs = [log['epoch'] for log in train_logs]
accuracies = [log['eval_accuracy'] for log in train_logs]
f1_scores = [log['eval_f1'] for log in train_logs]
precision_scores = [log['eval_precision'] for log in train_logs]
recall_scores = [log['eval_recall'] for log in train_logs]

plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracies, label="Accuracy", marker='o')
plt.plot(epochs, f1_scores, label="F1 Score", marker='o')
plt.plot(epochs, precision_scores, label="Precision", marker='o')
plt.plot(epochs, recall_scores, label="Recall", marker='o')
plt.xlabel("Epochs")
plt.ylabel("Performance")
plt.title("Model Performance Over Epochs")
plt.legend()
plt.grid()
plt.savefig("performance/Performance_Metrics_with_Precision_Recall.pdf", format="pdf", dpi=300)
plt.close()

# 2. Confusion Matrix
predictions = trainer.predict(test_dataset).predictions.argmax(-1)
cm = confusion_matrix(test_labels, predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal"] + [f"Class {i}" for i in range(1, num_classes)], yticklabels=["Normal"] + [f"Class {i}" for i in range(1, num_classes)])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig("performance/Confusion_Matrix.pdf", format="pdf", dpi=300)
plt.close()

# 3. ROC Curve
labels_binarized = label_binarize(test_labels, classes=list(range(num_classes)))
predictions_probs = trainer.predict(test_dataset).predictions

plt.figure(figsize=(10, 8))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(labels_binarized[:, i], predictions_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Multiclass Classification")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("ROC_Curve.pdf", format="pdf", dpi=300)
plt.close()

# Step 12: Save the Trained Model and Tokenizer
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')
