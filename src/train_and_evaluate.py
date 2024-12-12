import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
import time
from model import create_model
from keras.utils import to_categorical 

# Load prepared dataset
prepared_data_path = './data/prepared/mnist_14_final.npz'
data = np.load(prepared_data_path)
x_train = data['x_train']
y_train = data['y_train']
x_val = data['x_val']
y_val = data['y_val']
x_test = data['x_test']
y_test = data['y_test']

# One-hot encode the target labels
y_train = to_categorical(y_train, num_classes=10)
y_val = to_categorical(y_val, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Parameters
input_dim = x_train.shape[1]  # The dimension also needs to be 1D
output_dim = 10  # Digits 0-9

# Run one experiment at a time
hidden_nodes = 32  # 64, 128, 256, 512
batch_size = 32  # 64, 128
epochs = 150

results = []

start_time = time.time()

model = create_model(input_dim, hidden_nodes, output_dim)

# Train the model
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

time_taken = time.time() - start_time

# Evaluate on test data
y_pred = model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_labels, y_pred_labels)
precision = precision_score(y_test_labels, y_pred_labels, average='weighted')
recall = recall_score(y_test_labels, y_pred_labels, average='weighted')
mse = mean_squared_error(y_test_labels, y_pred_labels)

# Store results
results.append({
    #'hidden_nodes': hidden_nodes,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'mse': mse,
    'time': time_taken
})

# Calculate standard deviation for each metric across the runs
accuracy_std = np.std([res['accuracy'] for res in results])
precision_std = np.std([res['precision'] for res in results])
recall_std = np.std([res['recall'] for res in results])
mse_std = np.std([res['mse'] for res in results])

# Generate a unique identifier for the current experiment parameters
experiment_name = f"res_RS14_bn{batch_size}_lr0.0005"
filename_npy = f"results_{experiment_name}.npy"
filename_txt = f"results_{experiment_name}.txt"

# Save results as numpy array
np.save(filename_npy, results)

# Print the results
print(f"Results for {hidden_nodes} hidden nodes:")
print(f"  Accuracy ~ Std: {accuracy:.4f} ~ {accuracy_std:.4f}")
print(f"  Precision ~ Std: {precision:.4f} ~ {precision_std:.4f}")
print(f"  Recall ~ Std: {recall:.4f} ~ {recall_std:.4f}")
print(f"  MSE ~ Std: {mse:.4f} ~ {mse_std:.4f}")
print(f"  Time Taken: {time_taken:.2f} seconds")

# Save results to a file
with open(filename_txt, 'w') as result_file:
    result_file.write("=====================\n")
    #result_file.write(f"Input Dimensions: {input_dim}\n")
    #result_file.write(f"Batch Size: {batch_size}\n")
    #result_file.write(f"Hidden Layer Nodes: {hidden_nodes}\n")
    result_file.write(f"Time Taken: {time_taken:.2f} seconds\n")
    result_file.write(f"Accuracy ~ Std: {accuracy:.4f} ~ {accuracy_std:.4f}\n")
    result_file.write(f"Precision ~ Std: {precision:.4f} ~ {precision_std:.4f}\n")
    result_file.write(f"Recall~ Std: {recall:.4f} ~ {recall_std:.4f}\n")
    result_file.write(f"MSE ~ Std: {mse:.4f} ~ {mse_std:.4f}\n")

# Plot Error Trends (MSE)
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), history.history['loss'], label='Training Loss')
plt.plot(range(epochs), history.history['val_loss'], label='Validation Loss')
plt.title(f"Loss Trend Across Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f'./results/plots/loss_{experiment_name}.png')
plt.show()
