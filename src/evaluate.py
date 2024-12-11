import numpy as np

# Load metrics
metrics = np.load('./results/training_metrics.npy', allow_pickle=True)

# Print summary
for metric in metrics:
    print(f"Hidden Nodes: {metric['hidden_nodes']}, "
          f"Accuracy: {metric['accuracy']:.4f}, "
          f"Precision: {metric['precision']:.4f}, "
          f"Recall: {metric['recall']:.4f}, "
          f"MSE: {metric['mse']:.4f}")
