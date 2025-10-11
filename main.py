import numpy as np

from data import X, y
from activations import Sigmoid, ReLU, Softmax
from layers import DenseLayer
from losses import CategoricalCrossEntropy, SparseCategoricalCrossEntropy, BinaryCrossEntropy

# Create Dense layer with 2 input features and 3 output values
dense1 = DenseLayer(2, 3)
# Create ReLU activation (to be used with Dense layer)
activation1 = ReLU()

# Create second Dense layer with 3 input features (as we take output of previous layer here) and 3 output values
dense2 = DenseLayer(3, 3)
# Create Softmax activation (to be used with Dense layer)
activation2 = Softmax()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Make a forward pass through activation function
# It takes the output of first dense layer
activation1.forward(dense1.output)

# Make a forward pass through second Dense layer
# It takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Make a forward pass through activation function
# It takes in output of second dense layer
activation2.forward(dense2.output)

cat_loss = CategoricalCrossEntropy()
loss = cat_loss.calculate(activation2.output, y)

# The output of softmax layer (activation2.output) is a 2D array of shape (n_samples, n_classes), where each row contains the predicted probabilities for each class.
# The predicted class for each sample is the index of the maximum probability in that row.
y_pred = np.argmax(activation2.output, axis=1)

# If true labels y are one-hot encoded (shape (n_samples, n_classes)), convert them to integer class indices:
if len(y.shape) == 2:
    y_true = np.argmax(y, axis=1)
else:
    y_true = y

accuracy = np.mean(y_pred == y_true)

print("loss: ", loss)
print("Accuracy: ", accuracy)
