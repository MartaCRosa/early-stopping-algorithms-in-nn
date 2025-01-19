# Analysis of Early Stopping in Neural Networks: MNIST Dataset Classification

This project explores the impact of early stopping algorithms, specifically **Generalization Loss (GL)** and **Progressive Quitting (PQ)**, on neural networks trained for classifying MNIST dataset images. This is part of the coursework for the Neural Networks & Deep Learning course.

---

## Dataset and Preprocessing
The MNIST dataset is not included in this repository due to file size limitations. You can download the dataset from the following link:
- [MNIST Dataset (Train & Test CSV)](https://www.kaggle.com/competitions/digit-recognizer/data)

1. **Dataset**: The MNIST dataset, containing 28x28 grayscale images of digits 0–9, was resized to smaller dimensions for computational efficiency.
2. **Normalization**: Pixel values were normalized to the range [0, 1].
3. **Subset Selection**: Data was split into:
   - 10,000 training samples
   - 2,500 testing samples
4. **Preprocessing Steps**:
   - Reshaping, resizing, and flattening images into 1D arrays.
   - Splitting training data into training (80%) and validation (20%).
   - One-hot encoding the digit labels into 10 classes.

---

## Neural Network Architecture

- **Input Layer**: 14x14 images flattened into 196-dimensional vectors.
- **Hidden Layer**: Single layer with sigmoid activation, tested with nodes ranging from 16 to 256.
- **Output Layer**: Softmax activation for classification into 10 digit classes.
- **Optimizer**: Adam (Adaptive Moment Estimation).

---

## Hyperparameter Tuning

Key hyperparameters were tuned using grid search:
- **Learning Rates**: 0.0005, 0.001, and 0.005
- **Batch Sizes**: 32, 64, and 128
- **Fixed Parameters**: 32 hidden nodes, 150 epochs
- **Best Accuracy Achieved**: 95.68% ± 2.79 on the 14x14 dataset with 32 batch size and 0.001 learning rate.

---

## Results and Analysis

- **Generalization Loss Results**:
  - Optimal α: 12 with 256 hidden nodes.
  - Achieved accuracy: 96.72% ± 2.1.
- **Progressive Quitting Results**:
  - Optimal α: 0.12 with 256 hidden nodes.
  - Achieved accuracy: 96.80% ± 1.8.

---

## Conclusion

- The best model achieved an accuracy of **96.88%**.

---

### References

- Prechelt, L. (2000). *Early Stopping - But When?*

