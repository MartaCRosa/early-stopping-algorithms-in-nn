import numpy as np
import matplotlib.pyplot as plt
from model import create_model
from sklearn.metrics import precision_score, recall_score, accuracy_score, mean_squared_error

# Load processed data
processed_data_path = './data/prepared/mnist_20_final.npz'
data = np.load(processed_data_path)
x_train = data['x_train']
y_train = data['y_train']
x_val = data['x_val']
y_val = data['y_val']
x_test = data['x_test']
y_test = data['y_test']

# Parameters
input_dim = x_train.shape[1]
output_dim = 10
hidden_dimensions = 32 #64, 128, 256, 512

# To track metrics
results = []

for hidden_nodes in hidden_dimensions:
    print(f"\nTraining model with {hidden_nodes} hidden nodes...")
    model = create_model(input_dim, hidden_nodes, output_dim)

    # Train the model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=20,
        batch_size=32,
        verbose=1
    )

    # Evaluate on test data
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    mse = mean_squared_error(y_true, y_pred)

    # Store results
    results.append({
        'hidden_nodes': hidden_nodes,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'mse': mse,
    })

    print(f"Hidden Nodes: {hidden_nodes}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, MSE: {mse:.4f}")

# Save metrics
np.save('./results/training_metrics.npy', results)

# Plot Error Trends
plt.figure(figsize=(10, 6))
for res in results:
    plt.plot(res['mse'], label=f"{res['hidden_nodes']} nodes")
plt.title("Error Trends (MSE) Across Epochs")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.savefig('./results/error_trends.png')
plt.show()
