# VanillaNets v1.0.0

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/dependency-numpy%202.3.3%2B-green)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Production Ready ✓](https://img.shields.io/badge/status-production%20ready-success)]()

**A transparent, NumPy-only neural network library designed for learning and experimentation.**

VanillaNets is a from-scratch implementation of core neural network components using only Python and NumPy. Every component is written explicitly with clarity prioritized over convenience, making the entire system transparent, easy to inspect, and perfect for understanding how neural networks operate under the hood.

Whether you're a student learning fundamentals, a researcher prototyping new ideas, or an educator building curriculum, VanillaNets provides a crystal-clear window into neural network mechanics without framework abstractions.

---

## Features

### Core Architecture
* **Dense Layers** - Fully connected layers with efficient forward and backward passes
* **Advanced Weight Initialization** - He, Xavier (Glorot), normal and uniform distributions for optimized training
* **Flexible Bias Initialization** - Zeros or small positive constants to reduce dead units

### Activation Functions (with derivatives)
* Linear, ReLU & LeakyReLU
* Tanh & Sigmoid
* Softmax (with fused Softmax+CrossEntropy backward pass optimization)

### Loss Functions
* Binary Cross-Entropy (for binary classification)
* Categorical Cross-Entropy (for multiclass classification)
* Sparse Categorical Cross-Entropy (for integer-encoded labels)
* Mean Squared Error (for regression)

### Optimizers
* **SGD** - Stochastic Gradient Descent with momentum and learning rate decay
* **Adam** - Adaptive Moment Estimation with adaptive learning rates per parameter

### Metrics & Evaluation
* **Classification:** Accuracy, Precision, Recall, F1 Score, Confusion Matrix
* **Regression:** R² Score, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)

### Model API
* Sequential model building (`model.add()`)
* Flexible metrics interface (single metric or multiple metrics as dict/list)
* Training with `fit()` and optional validation data
* Inference with `predict()`
* Batch evaluation with `evaluate()`

### Performance Optimizations
* Fused Softmax + Categorical Cross-Entropy backward pass (faster training)
* Efficient NumPy vectorization throughout
* Memory-conscious layer implementations


---

## Quick Start

### Example 1: Binary Classification

```python
from model import Model
from layers import DenseLayer
from activations import ReLU, Sigmoid
from losses import BinaryCrossEntropy
from optimizers import Optimizer_Adam
from metrics import Accuracy

# Build model
model = Model()
model.add(DenseLayer(30, 64))
model.add(ReLU())
model.add(DenseLayer(64, 1))
model.add(Sigmoid())

# Compile with loss, optimizer, and metrics
model.set(
    loss=BinaryCrossEntropy(),
    optimizer=Optimizer_Adam(learning_rate=0.01),
    metrics={'accuracy': Accuracy()}
)
model.finalize()

# Train the model
model.fit(X_train, y_train, epochs=100, print_every=10, 
          validation_data=(X_val, y_val))

# Evaluate on test set
loss, metrics = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Accuracy: {metrics['accuracy']:.4f}")

# Make predictions
predictions = model.predict(X_new)
```

### Example 2: Multiclass Classification

```python
from activations import ReLU, Softmax
from losses import CategoricalCrossEntropy

# Build model
model = Model()
model.add(DenseLayer(784, 128))
model.add(ReLU())
model.add(DenseLayer(128, 64))
model.add(ReLU())
model.add(DenseLayer(64, 10))
model.add(Softmax())

# Compile
model.set(
    loss=CategoricalCrossEntropy(),
    optimizer=Optimizer_Adam(learning_rate=0.05),
    metrics={'accuracy': Accuracy()}
)
model.finalize()

# Train
model.fit(X_train, y_train, epochs=50, print_every=5)
```

### Example 3: Regression

```python
from activations import Linear, ReLU
from losses import MeanSquaredError
from metrics import RMSE, MAE

# Build model
model = Model()
model.add(DenseLayer(8, 64))
model.add(ReLU())
model.add(DenseLayer(64, 1))
model.add(Linear())

# Compile with multiple metrics
model.set(
    loss=MeanSquaredError(),
    optimizer=Optimizer_Adam(learning_rate=0.01),
    metrics={'rmse': RMSE(), 'mae': MAE()}
)
model.finalize()

# Train and evaluate
model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))
loss, metrics = model.evaluate(X_test, y_test)
print(f"Test RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
```

---

## Examples

Full working examples included:
* `binary_classification.py` - Breast cancer classification
* `multiclass_classification.py` - Handwritten digit recognition
* `regression.py` - California housing price prediction

Run any example:
```bash
python binary_classification.py
python multiclass_classification.py
python regression.py
```

---

## Testing

Run the comprehensive test suite (pytest required):

```bash
pip install pytest
pytest test_nn_lib.py -v
```

### Test Coverage

Comprehensive unit and integration tests cover:
- ✓ All activation functions (Linear, Sigmoid, ReLU, LeakyReLU, Tanh, Softmax) and their derivatives
- ✓ All loss functions (BCE, CCE, SparseCCE, MSE) with gradient validation
- ✓ Dense layer forward/backward passes
- ✓ Optimizer updates (SGD momentum, Adam adaptive rates)
- ✓ Fused Softmax+CrossEntropy optimization
- ✓ All metrics (classification & regression)
- ✓ Model training, evaluation, and prediction workflows
- ✓ Edge cases and numerical stability

---

## Design Philosophy

VanillaNets is built on the principle that **understanding requires transparency**:

- **No magic** ✓ Every computation is explicit; no hidden state or black-box frameworks
- **Learn by reading** ✓ Source code is the primary documentation
- **Experimentation-friendly** ✓ Modify any component without framework constraints
- **Pure NumPy** ✓ No external dependencies beyond NumPy for core functionality
- **Production-ready** ✓ Full test coverage, efficient implementations, stable API

## Use Cases

- **Education:** Perfect for coursework on neural networks and deep learning
- **Research Prototyping:** Experiment with new loss functions, activations, or optimization strategies
- **Interview Prep:** Implement solutions from scratch during ML engineering interviews
- **Curriculum Development:** Build course materials with fully transparent implementations
- **Algorithmic Learning:** Understand backpropagation, gradient descent, and optimizer mechanics

---

## Project Status

### v1.0.0 - Production Ready

**Fully Implemented & Tested:**
- ✓ Dense layer implementation with He, Xavier, normal, and uniform weight initialization
- ✓ All activation functions with proper gradient computation (Linear, ReLU, LeakyReLU, Tanh, Sigmoid, Softmax)
- ✓ All loss functions with backward passes (BCE, CCE, SparseCCE, MSE)
- ✓ SGD optimizer with momentum and learning rate decay
- ✓ Adam optimizer with adaptive learning rates
- ✓ Comprehensive metrics suite (Accuracy, Precision, Recall, F1, Confusion Matrix, R², MAE, RMSE)
- ✓ Full Model API (add, set, finalize, predict, evaluate, fit)
- ✓ Fused Softmax+CrossEntropy optimization for faster training
- ✓ Extensive test coverage (50+ test cases)
- ✓ Complete example applications (binary classification, multiclass classification, regression)
- ✓ Validation data support during training

### Future Enhancements (Post-v1.0)
- Convolutional (Conv2D) layers with pooling
- Recurrent layers (LSTM, GRU)
- Batch normalization and layer normalization
- Dropout regularization
- Custom layer support through base class
- Learning rate scheduling
- Distributed training utilities (multi-GPU)
- Quantization and pruning support

---

## License

MIT License - See [LICENSE](LICENSE) for full details.

You are free to use, modify, and distribute this software for any purpose (commercial or personal) with proper attribution.

---

## Acknowledgments

VanillaNets was built with a singular mission: to demystify neural networks for learners everywhere. This library stands on the shoulders of foundational work in deep learning by pioneers like Yann LeCun, Geoffrey Hinton, Yoshua Bengio, and the broader machine learning community.

Special thanks to:
- The NumPy team for creating an incredible numerical computing foundation
- All educators who emphasize understanding over black-box frameworks
- Contributors and users who provide feedback and improvements

## Citation

If you use VanillaNets in your research or teaching, please cite:

```bibtex
@software{vanillanets2026,
  title={VanillaNets: A Transparent Neural Network Library},
  author={Umar Balak},
  year={2026},
  url={https://github.com/UmarBalak/vanillanet-library}
}
```

---

## Resources & References

### Recommended Reading
- **Neural Networks from Scratch in Python**, co-authored by Harrison Kinsley and Daniel Kukieła
- **Deep Learning** by Goodfellow, I., Bengio, Y., & Courville, A.
- **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd Edition** by Aurélien Géron


### Related Projects
- [NumPy](https://numpy.org/) - Our computational foundation
- [nnfs](https://github.com/nnaisense/nnfs) - Dataset utilities
- [3blue1brown Neural Network Series](https://www.youtube.com/watch?v=aircAruvnKk) - Visual learning guide

---

**Built with ❤️ for learners by learners.**
