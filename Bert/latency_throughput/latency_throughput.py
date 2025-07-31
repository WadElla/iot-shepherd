# holistic_metrics.py

# Input the metrics obtained from individual scripts:
extraction_time = float(input("Enter total feature extraction time (seconds): "))
num_packets = int(input("Enter number of packets processed during feature extraction: "))

inference_time = float(input("Enter total inference time (seconds): "))
num_samples = int(input("Enter number of samples processed during inference: "))

# Total pipeline time is the sum of feature extraction and inference times.
total_pipeline_time = extraction_time + inference_time

# Overall throughput is based on the inference stage's sample count.
overall_throughput = num_samples / total_pipeline_time if total_pipeline_time > 0 else 0

# Average latency per sample for the full pipeline:
average_pipeline_latency_per_sample = total_pipeline_time / num_samples if num_samples > 0 else float('inf')

print("\nHolistic Pipeline Metrics:")
print(f"Total Pipeline Time (PCAP to Anomaly Detection): {total_pipeline_time:.4f} seconds")
print(f"Overall Throughput (samples/second): {overall_throughput:.2f}")
print(f"Average Pipeline Latency per Sample: {average_pipeline_latency_per_sample:.6f} seconds")
