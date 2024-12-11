import pandas as pd
import numpy as np
from skimage.transform import resize

# Paths to input files (raw data)
train_path = './data/raw/mnist_train.csv'
test_path = './data/raw/mnist_test.csv'

# Load datasets
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Separate features (pixels) and labels
x_train = train_data.iloc[:, 1:].values  # All columns except the first (label)
y_train = train_data.iloc[:, 0].values   # First column (label)
x_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# Normalize pixel values to [0, 1]
x_train = x_train / 255.0   # 0 is black and 255 is white, in between there are greys
x_test = x_test / 255.0

# Function to resize images
def resize_images(images, new_size=(14, 14)):
    resized_images = np.zeros((images.shape[0], new_size[0] * new_size[1]))
    for i in range(images.shape[0]):
        img = images[i].reshape(28, 28)  # Original 28x28
        img_resized = resize(img, new_size, anti_aliasing=True)  # Resize
        resized_images[i] = img_resized.flatten()  # Flatten back to 1D
    return resized_images

# Resize train and test images
x_train_resized = resize_images(x_train, new_size=(20, 20))
x_test_resized = resize_images(x_test, new_size=(20, 20))

# Save the resized dataset to the processed folder
np.savez_compressed('./data/processed/mnist_resized.npz', 
                    x_train=x_train_resized, y_train=y_train, 
                    x_test=x_test_resized, y_test=y_test)

print("Resized dataset saved to './data/processed/...'")
