import numpy as np
from keras.utils import to_categorical
from model import create_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
import matplotlib.pyplot as plt
import time

prepared_data_path = './data/prepared/mnist_14_final.npz'
data = np.load(prepared_data_path)
x_train = data['x_train']
y_train = data['y_train']
x_val = data['x_val']
y_val = data['y_val']
x_test = data['x_test']
y_test = data['y_test']

# One-hot encoding the target labels
y_train = to_categorical(y_train, num_classes=10)
y_val = to_categorical(y_val, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

input_dim = x_train.shape[1]
output_dim = 10

hidden_nodes = 32
batch_size = 32
epochs = 150
pq_alpha = 2  # PQ stopping threshold
training_strip_length = 5  # Length of training strips for calculating progress

results = []
start_time = time.time()

model = create_model(input_dim, hidden_nodes, output_dim)

history = {
    'accuracy': [],
    'val_accuracy': [],
    'loss': [],
    'val_loss': []
}

best_val_loss = np.inf  # Track best validation loss

# Calculate training progress (PQ) variables
training_strip_losses = []

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")

    # Train for one epoch
    train_history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=1,
        validation_data=(x_val, y_val),
        verbose=1
    )

    # Retrieve loss values
    train_loss = train_history.history['loss'][0]
    val_loss = train_history.history['val_loss'][0]
    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)

    # Update the strip losses
    training_strip_losses.append(train_loss)

    # Ensure strip has `training_strip_length` elements
    if len(training_strip_losses) > training_strip_length:
        training_strip_losses.pop(0)

    # Compute training progress
    if len(training_strip_losses) == training_strip_length:
        min_loss_in_strip = min(training_strip_losses)
        avg_loss_in_strip = np.mean(training_strip_losses)
        progress = 1000 * (avg_loss_in_strip / min_loss_in_strip - 1)

        # Calculate PQ
        pq = generalization_loss / progress if progress > 0 else np.inf
        print(f"Generalization Loss: {generalization_loss:.2f}% | Training Progress: {progress:.2f} | PQ: {pq:.2f}")

        # Stop if PQ exceeds threshold
        if pq > pq_alpha:
            print(f"PQ exceeded threshold ({pq_alpha}), stopping training.")
            break

    best_val_loss = min(best_val_loss, val_loss)

time_taken = time.time() - start_time
last_epoch = epoch + 1

# Evaluate the model
y_pred = model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_labels, y_pred_labels)
precision = precision_score(y_test_labels, y_pred_labels, average='weighted')
recall = recall_score(y_test_labels, y_pred_labels, average='weighted')
mse = mean_squared_error(y_test_labels, y_pred_labels)

# Print results
print(f"\nResults for {hidden_nodes} hidden nodes:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  MSE: {mse:.4f}")
print(f"  Time Taken: {time_taken:.2f}s")
print(f"  Last Epoch: {last_epoch}")

# Save metrics to file
experiment_name = f"PQ_alpha_{pq_alpha}_hn_{hidden_nodes}"
filename_txt = f"./results/metrics/{experiment_name}.txt"
with open(filename_txt, 'w') as result_file:
    result_file.write("=====================\n")
    result_file.write(f"Hidden Layer Nodes: {hidden_nodes}\n")
    result_file.write(f"Last Epoch: {last_epoch}\n")
    result_file.write(f"Accuracy: {accuracy:.4f}\n")
    result_file.write(f"Precision: {precision:.4f}\n")
    result_file.write(f"Recall: {recall:.4f}\n")
    result_file.write(f"MSE: {mse:.4f}\n")
    result_file.write(f"Time Taken: {time_taken:.2f}s\n")
