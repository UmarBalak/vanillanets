import numpy as np
from activations import Softmax
from losses import CategoricalCrossEntropy

class Activation_Softmax_Loss_CategoricalCrossentropy:
    """
    Combined Softmax activation and cross-entropy loss for faster backward step
    """
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        
        # Calculate gradient (predicted probability - true label)
        # We subtract 1 from the predicted probability at the index of the true label
        self.dinputs[range(samples), y_true] -= 1
        
        # Normalize gradient
        # If we don't normalize, larger batch sizes will result in larger gradients, 
        # making training unstable.
        self.dinputs = self.dinputs / samples