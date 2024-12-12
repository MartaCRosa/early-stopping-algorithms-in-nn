import numpy as np
from keras.utils import to_categorical 
from model_research import create_model
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

input_dim = x_train.shape[1]  # The input of the NN is the training data shaped in 1D
output_dim = 10  # The output are the digits 0-9

hidden_nodes = 32  # 32 64, 128, 256, 512
batch_size = 32  # 32 64, 128
epochs = 150

results = []
start_time = time.time()

model = create_model(input_dim, hidden_nodes, output_dim)

# Train the model using the predefined method in TensorFlow/Keras
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1  # Show progress bars
)

time_taken = time.time() - start_time

# Evaluate on test data
y_pred = model.predict(x_test)  # 2D NumPy array containing the predicted probabilities for each class (one row per sample) based on the trained model
y_pred_labels = np.argmax(y_pred, axis=1)  # 1D array containing the predicted class labels by finding the index of the maximum probability in each row (axis=1 to make 1D)
y_test_labels = np.argmax(y_test, axis=1)  # 1D array containing the true class labels by converting the one-hot encoding into their class label representation

accuracy = accuracy_score(y_test_labels, y_pred_labels)
precision = precision_score(y_test_labels, y_pred_labels, average='weighted')  
recall = recall_score(y_test_labels, y_pred_labels, average='weighted')
mse = mean_squared_error(y_test_labels, y_pred_labels)
# weighted ensures that the contribution of each class to the final score is weighted by the number of samples in that class (i.e. some classes may have more samples than others)

results.append({
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'mse': mse,
    'time': time_taken
})

# Append per-epoch metrics to lists, from history, to be able to calculate standard deviation across epochs
epoch_accuracies = history.history['accuracy']
epoch_precisions = history.history.get('precision', [])
epoch_recalls = history.history.get('recall', [])
epoch_mses = history.history.get('mse', [])

accuracy_std = np.std(epoch_accuracies)
precision_std = np.std(epoch_precisions) if epoch_precisions else 0
recall_std = np.std(epoch_recalls) if epoch_recalls else 0
mse_std = np.std(epoch_mses) if epoch_mses else 0

# Print the results
print(f"Results for {hidden_nodes} hidden nodes:")
print(f"  Accuracy ~ Std: {accuracy:.4f} ~ {accuracy_std:.4f}")
print(f"  Precision ~ Std: {precision:.4f} ~ {precision_std:.4f}")
print(f"  Recall ~ Std: {recall:.4f} ~ {recall_std:.4f}")
print(f"  MSE ~ Std: {mse:.4f} ~ {mse_std:.4f}")
print(f"  Time Taken: {time_taken:.2f} seconds")

experiment_name = f"res_RS14_bn{batch_size}_lr0.001"
filename_npy = f"./hyperparameter_research/results/RS14/{experiment_name}.npy"
filename_txt = f"./hyperparameter_research/results/metrics/RS14/{experiment_name}.txt"

# Save results as numpy array
np.save(filename_npy, results)

# Save results to a file
with open(filename_txt, 'w') as result_file:
    result_file.write("=====================\n")
    result_file.write(f"Time Taken: {time_taken:.2f} seconds\n")
    result_file.write(f"Accuracy ~ Std: {accuracy:.4f} ~ {accuracy_std:.4f}\n")
    result_file.write(f"Precision ~ Std: {precision:.4f} ~ {precision_std:.4f}\n")
    result_file.write(f"Recall~ Std: {recall:.4f} ~ {recall_std:.4f}\n")
    result_file.write(f"MSE ~ Std: {mse:.4f} ~ {mse_std:.4f}\n")

# Plot Error Trends
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), history.history['loss'], label='Training Loss')
plt.plot(range(epochs), history.history['val_loss'], label='Validation Loss')
plt.title(f"Loss Trend Across Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f'./hyperparameter_research/results/plots/RS14/loss_{experiment_name}.png')
plt.show()
