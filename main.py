import numpy as np

def check_processed_data(file_path):
    # Load the processed .npz file
    data = np.load(file_path)
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']

    # Print dataset sizes
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    print(f"Input dimensions: {x_train.shape[1]} (Flattened)")

    return x_train, y_train, x_test, y_test

# Example usage
data_file = './data/processed/mnist_20_resized.npz'
x_train, y_train, x_test, y_test = check_processed_data(data_file)