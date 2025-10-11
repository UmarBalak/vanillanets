import numpy as np

class Linear:

    def forward(self, inputs):
        self.outputs = inputs

class Sigmoid:

    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
        
class ReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class LeakyReLU:

    def forward(self, inputs):
        self.outputs = np.where(inputs > 0, inputs, 0.1 * inputs)

class Tanh:
    """
    NumPy provides direct, built-in function for Tanh 
    because it is a standard mathematical function (hyperbolic tangent), 
    just like sine or cosine.
    """

    def forward(self, inputs):
        """
        # self.outputs = (np.exp(inputs) - np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs))

        # Manual formula for tanh works mathematically for small/typical inputs 
        # but produces nan for large values because np.exp(inputs) overflows (is too large for float storage), 
        # and the denominator also overflows, leading to division by infinity or undefined math.
        """
        
        # The objective is to utilize NumPy for implementing the neural network, rather than hardcoding operations using pure Python.

        self.outputs = np.tanh(inputs)

class Softmax:

    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # subtracting 'max value' to prevent the exponents values from becoming very large

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

