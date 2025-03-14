# combined_evaluation.py
import time
from sklearn.metrics import f1_score

def simulate_model_response(true_label, prompt):
    # Simulate a model response (for demonstration)
    time.sleep(0.4)
    response = "Simulated response"
    predicted_label = true_label  # for demonstration, assume perfect prediction
    tokens_used = len(response.split())
    return response, predicted_label, tokens_used

def combined_evaluation(true_labels, prompts):
    predictions = []
    latencies = []
    for true_label, prompt in zip(true_labels, prompts):
        start = time.time()
        response, pred_label, tokens = simulate_model_response(true_label, prompt)
        latency = time.time() - start
        predictions.append(pred_label)
        latencies.append(latency)
    overall_f1 = f1_score(true_labels, predictions, average='weighted')
    avg_latency = sum(latencies) / len(latencies)
    print(f"Overall F1 Score: {overall_f1:.4f}")
    print(f"Average Latency: {avg_latency:.2f} seconds")
    return overall_f1, avg_latency

# Example usage:
true_labels = [1, 0, 1]
prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
combined_evaluation(true_labels, prompts)
