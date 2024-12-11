import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.transform import resize

# Paths to processed data
processed_data_path = './data/processed/mnist_20_resized.npz'

# Load processed dataset
data = np.load(processed_data_path)
x_train_resized = data['x_train']
y_train = data['y_train']
x_test_resized = data['x_test']
y_test = data['y_test']

# Function to subset data
def subset_data(x, y, n_train=10000, n_test=2500, random_seed=42):
    """
    Subsets the dataset to a smaller number of training and test samples.
    """
    np.random.seed(random_seed)

    # Randomly shuffle indices for training and test sets
    train_indices = np.random.choice(x.shape[0], n_train, replace=False)
    test_indices = np.random.choice(y.shape[0], n_test, replace=False)

    # Subset the data
    x_train_subset = x[train_indices]
    y_train_subset = y[train_indices]
    x_test_subset = x[test_indices]
    y_test_subset = y[test_indices]

    return x_train_subset, y_train_subset, x_test_subset, y_test_subset

# Subset the data
x_train_subset, y_train_subset, x_test_subset, y_test_subset = subset_data(
    x_train_resized, y_train, n_train=10000, n_test=2500
)

# Split data into training and validation sets (80-20)
def split_train_val(x_train, y_train, val_ratio=0.2, random_seed=42):
    """
    Splits the training data into train and validation sets.
    """
    x_train_final, x_val_final, y_train_final, y_val_final = train_test_split(
        x_train, y_train, test_size=val_ratio, random_state=random_seed
    )
    return x_train_final, y_train_final, x_val_final, y_val_final

# Perform train/validation split
x_train_final, y_train_final, x_val_final, y_val_final = split_train_val(
    x_train_subset, y_train_subset
)

# One-hot encoding function
def one_hot_encode(labels, num_classes=10):
    """
    Converts numerical labels into one-hot encoded vectors.
    """
    encoded = np.zeros((labels.shape[0], num_classes))
    for i, label in enumerate(labels):
        encoded[i, label] = 1
    return encoded

# One-hot encode the labels after all splits
y_train_encoded = one_hot_encode(y_train_final)
y_val_encoded = one_hot_encode(y_val_final)
y_test_encoded = one_hot_encode(y_test_subset)

# Save final datasets
np.savez_compressed('./data/prepared/mnist_20_final.npz', 
                    x_train=x_train_final, y_train=y_train_final, 
                    x_val=x_val_final, y_val=y_val_final, 
                    x_test=x_test_subset, y_test=y_test_subset)

# Print results
print("Data preparation completed and saved:")
print(f"Training set: {x_train_final.shape[0]} samples")
print(f"Validation set: {x_val_final.shape[0]} samples")
print(f"Test set: {x_test_subset.shape[0]} samples")
