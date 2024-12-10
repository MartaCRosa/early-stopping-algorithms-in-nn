# nndl-project

## Dataset

The MNIST dataset is not included in this repository due to file size limitations. You can download the dataset from the following link:

- [MNIST Dataset (Train & Test CSV)](https://www.kaggle.com/competitions/digit-recognizer/data)


TOPIC 8:
Consider the raw images from the MNIST dataset as input. This is a classification problem 
with C classes, where C=10. Extract a global dataset of N pairs, and divide it appropriately 
into training and test sets (consider at least 10,000 elements for the training set and 2,500 for 
the test set). Use resilient backpropagation (RProp) as the weight update algorithm (batch 
update). Study the learning process of a neural network (e.g., epochs required for learning, 
error trend on training and validation sets, accuracy on the test set) with a single layer of 
internal nodes and using the sigmoid activation function, varying the early-stopping 
criterion. Referring to the article “Early Stopping — But When? Lutz Prechelt, 1999”, 
consider the two early-stopping algorithms GL and PQ with different values for the 
parameter α\alpha. Study networks with a different number of internal nodes (at least five 
different dimensions). If necessary, due to computational time and memory constraints, you 
can reduce the dimensions of the raw MNIST dataset images (e.g., using the imresize 
function in MATLAB).

