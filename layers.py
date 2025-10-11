import numpy as np 
np.random.seed(42) 

# class DenseLayer:

#     def __init__(self, n_inputs, n_neurons):
#         # Initialize weights and biases
#         # In general, neural networks work best with values between -1 and +1
#         self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # initialize weights randomly from Gaussian distribution, (already transposed, so don't need to take transpose each time)
#         self.biases = np.zeros((1, n_neurons)) # 1 -> number of rows, n_neurons -> number of columns

#     def forward(self, inputs):
#         # Calculate output values from inputs, weights and biases
#         self.output = np.dot(inputs, self.weights) + self.biases

import numpy as np

class DenseLayer:
    def __init__(self, n_inputs, n_neurons, *, activation='relu', init='auto',
                 distribution='normal', bias_init='zeros', seed=None):
        """
        n_inputs: fan-in
        n_neurons: fan-out
        activation: 'relu', 'leaky_relu', 'tanh', 'sigmoid', 'softmax', 'linear'
        init: 'auto' | 'he' | 'xavier'
        distribution: 'normal' or 'uniform'
        bias_init: 'zeros' or float (small constant)
        seed: optional int for reproducibility
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        fan_in, fan_out = n_inputs, n_neurons

        # choose initializer
        if init == 'auto':
            if activation in ('relu', 'leaky_relu'):
                init = 'he'
            else:
                init = 'xavier'

        if init == 'he':
            # std = sqrt(2 / fan_in)
            if distribution == 'normal':
                std = np.sqrt(2.0 / fan_in)
                self.weights = rng.normal(0.0, std, size=(fan_in, fan_out))
            else:
                # limit = sqrt(6 / fan_in)
                limit = np.sqrt(6.0 / fan_in)
                self.weights = rng.uniform(-limit, limit, size=(fan_in, fan_out))

        elif init == 'xavier':
            # normal: std = sqrt(2 / (fan_in + fan_out))
            # uniform: limit = sqrt(6 / (fan_in + fan_out))
            denom = (fan_in + fan_out)
            if distribution == 'normal':
                std = np.sqrt(2.0 / denom)
                self.weights = rng.normal(0.0, std, size=(fan_in, fan_out))
            else:
                limit = np.sqrt(6.0 / denom)
                self.weights = rng.uniform(-limit, limit, size=(fan_in, fan_out))

        else:
            raise ValueError("init must be 'auto', 'he', or 'xavier'")

        # biases: prefer zeros; optionally small positive for ReLU to reduce dead units
        if bias_init == 'zeros':
            self.biases = np.zeros((1, fan_out))
        elif isinstance(bias_init, (int, float)):
            self.biases = np.full((1, fan_out), float(bias_init))
        else:
            raise ValueError("bias_init must be 'zeros' or a numeric constant")

        self.output = None

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases