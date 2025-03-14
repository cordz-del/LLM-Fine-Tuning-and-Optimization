# evaluation_metrics.py
from sklearn.metrics import accuracy_score, f1_score

def evaluate_classification(true_labels, predictions):
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    return acc, f1

# Example usage:
true_labels = [0, 1, 1, 0, 1]
predictions = [0, 1, 0, 0, 1]
evaluate_classification(true_labels, predictions)
