import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.transform import resize

resized_data_path = './data/processed/mnist_14_resized.npz'

data = np.load(resized_data_path)
x_train_resized = data['x_train']
y_train = data['y_train']
x_test_resized = data['x_test']
y_test = data['y_test']

def subset_data(x, y, n_train=10000, n_test=2500, random_seed=42):  # Choice of number 42 is common practice
    np.random.seed(random_seed)  # This makes the random selection reproducible

    # Randomly shuffle indices for training and test sets
    train_indices = np.random.choice(x.shape[0], n_train, replace=False)  # Randomly selects n_train indices (without replacement) from the range [0, x.shape[0]).
    test_indices = np.random.choice(y.shape[0], n_test, replace=False)

    x_train_subset = x[train_indices]
    y_train_subset = y[train_indices]
    x_test_subset = x[test_indices]
    y_test_subset = y[test_indices]

    return x_train_subset, y_train_subset, x_test_subset, y_test_subset

# Subset the data
x_train_subset, y_train_subset, x_test_subset, y_test_subset = subset_data(x_train_resized, y_train, n_train=10000, n_test=2500)

# Split training data into training and validation sets (80-20) function
def split_train_val(x_train, y_train, val_ratio=0.2, random_seed=42):

    x_train_final, x_val_final, y_train_final, y_val_final = train_test_split(x_train, y_train, test_size=val_ratio, random_state=random_seed)  # train_test_split is a function that slits a dataset into 2, in this case, randomly

    return x_train_final, y_train_final, x_val_final, y_val_final

# Actual train/validation split
x_train_final, y_train_final, x_val_final, y_val_final = split_train_val(x_train_subset, y_train_subset)

# Save final datasets
np.savez_compressed('./data/prepared/mnist_14_final.npz', 
                    x_train=x_train_final, y_train=y_train_final, 
                    x_val=x_val_final, y_val=y_val_final, 
                    x_test=x_test_subset, y_test=y_test_subset)

# Confirm subsets dimensions
print(f"Training set: {x_train_final.shape[0]} samples")
print(f"Validation set: {x_val_final.shape[0]} samples")
print(f"Test set: {x_test_subset.shape[0]} samples")
