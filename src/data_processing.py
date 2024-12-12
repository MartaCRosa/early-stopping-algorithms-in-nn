import pandas as pd
import numpy as np
from skimage.transform import resize

train_path = './data/raw/mnist_train.csv'
test_path = './data/raw/mnist_test.csv'

# Use pandas to load/read files
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Separate features (pixels) from labels
x_train = train_data.iloc[:, 1:].values  # All columns except the first that is the label
y_train = train_data.iloc[:, 0].values   # Label
x_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# Normalize pixel values
x_train = x_train / 255.0   # 0 is black and 255 is white, in between there are greys
x_test = x_test / 255.0

def resize_images(images, new_size=(14, 14)):
    resized_images = np.zeros((images.shape[0], new_size[0] * new_size[1])) # images.shape[0] gives the number of images in the set; could be new_size[0]^2 because it's a square
    for i in range(images.shape[0]):
        img = images[i].reshape(28, 28)  # Each image in the dataset is reshaped to the original 28x28, in a 2D array
        img_resized = resize(img, new_size, anti_aliasing=True)  # Resizing, anti_aliasing helps reducing blurriness
        resized_images[i] = img_resized.flatten()  # Flatten back to 1D like the dataset, put it in the array with the index i
    return resized_images

# Resize the train and test images
x_train_resized = resize_images(x_train, new_size=(14, 14))
x_test_resized = resize_images(x_test, new_size=(14, 14))

# Save
np.savez_compressed('./data/processed/mnist_resized.npz', 
                    x_train=x_train_resized, y_train=y_train, 
                    x_test=x_test_resized, y_test=y_test)
