import numpy as np

class Linear:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        # The derivative is 1, so 1 * dvalues = dvalues
        self.dinputs = dvalues.copy()

class Sigmoid:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        
    def backward(self, dvalues):
        # Derivative of Sigmoid is: output * (1 - output)
        self.dinputs = dvalues * (1 - self.output) * self.output
        
class ReLU:

    def forward(self, inputs):
        # Remember inputs for backward pass
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Since we need to modify the original variable, we make a copy first
        self.dinputs = dvalues.copy()
        
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


class LeakyReLU:

    def forward(self, inputs):
        # Remember inputs for backward pass
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, 0.1 * inputs)

    def backward(self, dvalues):
        # Make a copy of values first
        self.dinputs = dvalues.copy()
        
        # Multiply gradient by 0.1 where input values were negative or zero
        self.dinputs[self.inputs <= 0] *= 0.1

class Tanh:
    """
    NumPy provides direct, built-in function for Tanh 
    because it is a standard mathematical function (hyperbolic tangent), 
    just like sine or cosine.
    """

    def forward(self, inputs):
        """
        # self.output = (np.exp(inputs) - np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs))

        # Manual formula for tanh works mathematically for small/typical inputs 
        # but produces nan for large values because np.exp(inputs) overflows (is too large for float storage), 
        # and the denominator also overflows, leading to division by infinity or undefined math.
        """
        
        # The objective is to utilize NumPy for implementing the neural network, rather than hardcoding operations using pure Python.
        self.inputs = inputs
        self.output = np.tanh(inputs)

    def backward(self, dvalues):
            # Derivative of Tanh is: 1 - (output)^2
            self.dinputs = dvalues * (1 - self.output ** 2)

class Softmax:

    def forward(self, inputs):
        self.inputs = inputs
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # Create uninitialized array to hold the gradients
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            
            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
