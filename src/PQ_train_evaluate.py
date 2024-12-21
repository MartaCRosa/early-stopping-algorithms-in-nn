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
pq_alpha = 2  # Progressive quitting stopping threshold in %
training_strip_length = 5  # =k length of training strips to calculate progress

results = []
start_time = time.time()

model = create_model(input_dim, hidden_nodes, output_dim)

# Initialize variables for PQ early stopping
best_val_loss = np.inf  # Tracks best validation loss
history = {'val_accuracy': [], 'loss': [], 'val_loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'mse': []}
training_strip_losses = []  # To later alculate training progress (P_k)

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")

    # Train for each epoch
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
    val_acc = train_history.history['val_accuracy'][0]
    
    # Compute validation predictions and metrics
    val_preds = np.argmax(model.predict(x_val), axis=1)
    val_true = np.argmax(y_val, axis=1)
    val_precision = precision_score(val_true, val_preds, average='weighted')
    val_recall = recall_score(val_true, val_preds, average='weighted')
    val_mse = mean_squared_error(val_true, val_preds)

    # Append to history
    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['accuracy'].append(val_acc)
    history['precision'].append(val_precision)
    history['recall'].append(val_recall)
    history['mse'].append(val_mse)

    # Update the strip losses
    training_strip_losses.append(train_loss)

    # When the strip already has k elements it deletes the oldest loss [0]
    if len(training_strip_losses) > training_strip_length:
        training_strip_losses.pop(0)

    # Compute training progress
    if len(training_strip_losses) == training_strip_length:
        min_loss_in_strip = min(training_strip_losses)
        avg_loss_in_strip = sum(training_strip_losses) / training_strip_length
        P_k = 1000 * (avg_loss_in_strip / min_loss_in_strip - 1)  # Training progress

        # Calculate PQ
        generalization_loss = 100 * (val_loss / min_loss_in_strip - 1)
        pq = generalization_loss / P_k
        print(f"Generalization Loss: {generalization_loss:.2f}% | Training Progress: {P_k:.2f} | PQ: {pq:.2f}")

        # Stop if PQ exceeds threshold
        if pq > pq_alpha:
            print(f"PQ exceeded threshold ({pq_alpha}), stopping training.")
            break

    #best_val_loss = min(best_val_loss, val_loss)

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

# Standard deviations for metrics across epochs
accuracy_std = np.std(history['accuracy'])
precision_std = np.std(history['precision'])
recall_std = np.std(history['recall'])
mse_std = np.std(history['mse'])

# Print the results
print(f"\nResults for {hidden_nodes} hidden nodes:")
print(f"  Accuracy ~ Std: {accuracy:.4f} ~ {accuracy_std:.4f}")
print(f"  Precision ~ Std: {precision:.4f} ~ {precision_std:.4f}")
print(f"  Recall ~ Std: {recall:.4f} ~ {recall_std:.4f}")
print(f"  MSE ~ Std: {mse:.4f} ~ {mse_std:.4f}")
print(f"  Time Taken: {time_taken:.2f} seconds")
print(f"  Last Epoch: {last_epoch}")

# Save metrics
experiment_name = f"PQ_alpha_{pq_alpha}_hn_{hidden_nodes}"

filename_txt = f"./results/metrics/PQ/{experiment_name}.txt"
with open(filename_txt, 'w') as result_file:
    result_file.write("=====================\n")
    result_file.write(f"Hidden Layer Nodes: {hidden_nodes}\n")
    result_file.write(f"Time Taken: {time_taken:.2f} seconds\n")
    result_file.write(f"Last Epoch: {last_epoch}\n")
    result_file.write(f"Accuracy ~ Std: {accuracy:.4f} ~ {accuracy_std:.4f}\n")
    result_file.write(f"Precision ~ Std: {precision:.4f} ~ {precision_std:.4f}\n")
    result_file.write(f"Recall ~ Std: {recall:.4f} ~ {recall_std:.4f}\n")
    result_file.write(f"MSE ~ Std: {mse:.4f} ~ {mse_std:.4f}\n")

# Plot Error Trends
plt.figure(figsize=(10, 6))
plt.plot(range(len(history['loss'])), history['loss'], label='Training Loss')
plt.plot(range(len(history['val_loss'])), history['val_loss'], label='Validation Loss')
plt.title("Loss Trend Across Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f'./results/plots/PQ/{experiment_name}.png')
plt.show()
