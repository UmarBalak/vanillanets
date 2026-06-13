"""
VanillaNets v1.0.0 - A transparent, NumPy-only neural network library.

A from-scratch implementation of core neural network components using only Python and NumPy.
Every component is written explicitly with clarity prioritized over convenience.

Main Classes:
    - Model: Sequential model for building neural networks
    - DenseLayer: Fully connected layers
    - Activation functions: Linear, ReLU, LeakyReLU, Tanh, Sigmoid, Softmax
    - Loss functions: BinaryCrossEntropy, CategoricalCrossEntropy, SparseCategoricalCrossEntropy, MeanSquaredError
    - Optimizers: Optimizer_SGD, Optimizer_Adam
    - Metrics: Accuracy, Precision, Recall, F1Score, ConfusionMatrix, R2Score, MAE, RMSE

Example Usage:
    >>> from vanillanets import Model, DenseLayer
    >>> from vanillanets.activations import ReLU, Sigmoid
    >>> from vanillanets.losses import BinaryCrossEntropy
    >>> from vanillanets.optimizers import Optimizer_Adam
    >>> from vanillanets.metrics import Accuracy
    >>>
    >>> model = Model()
    >>> model.add(DenseLayer(30, 64))
    >>> model.add(ReLU())
    >>> model.add(DenseLayer(64, 1))
    >>> model.add(Sigmoid())
    >>>
    >>> model.set(
    ...     loss=BinaryCrossEntropy(),
    ...     optimizer=Optimizer_Adam(learning_rate=0.01),
    ...     metrics={'accuracy': Accuracy()}
    ... )
    >>> model.finalize()
    >>>
    >>> model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))
    >>> predictions = model.predict(X_test)
"""

__version__ = "1.0.0"
__author__ = "Umar Balak"
__license__ = "MIT"

# Core model and layer
from .model import Model
from .layers import DenseLayer

# Activation functions
from .activations import (
    Linear,
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    Softmax,
)

# Loss functions
from .losses import (
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    SparseCategoricalCrossEntropy,
    MeanSquaredError,
)

# Optimizers
from .optimizers import Optimizer_SGD, Optimizer_Adam

# Metrics
from .metrics import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
    R2Score,
    MAE,
    RMSE,
)

# Legacy accuracy
from .accuracy import Accuracy as Accuracy_Legacy

__all__ = [
    # Core
    "Model",
    "DenseLayer",
    # Activations
    "Linear",
    "ReLU",
    "LeakyReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    # Losses
    "BinaryCrossEntropy",
    "CategoricalCrossEntropy",
    "SparseCategoricalCrossEntropy",
    "MeanSquaredError",
    # Optimizers
    "Optimizer_SGD",
    "Optimizer_Adam",
    # Metrics
    "Accuracy",
    "Precision",
    "Recall",
    "F1Score",
    "ConfusionMatrix",
    "R2Score",
    "MAE",
    "RMSE",
    "Accuracy_Legacy",
]
