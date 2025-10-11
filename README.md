# VanillaNets

VanillaNets is a from-scratch implementation of the core building blocks of neural networks using only Python and NumPy.

The focus is on clarity over convenience, every component is written explicitly, making the entire system transparent, easy to inspect, and perfect for learning how things work under the hood.

### Features

* Dense layers with forward and backward passes (Advanced weight initialization techniques)
* Activation functions: Linear, ReLU, LeakyReLU, Tanh, Sigmoid, Softmax
* Losses: Binary Cross-Entropy, Categorical, Sparse Categorical
* Minimal, framework-free design (only NumPy)

### Why

* To truly understand how neural networks operate at the lowest level
* To build intuition for gradients, activations, and loss functions
* To create a clean foundation for adding more advanced topics like optimizers and regularization

### Project Status

* ✅ Implemented: Dense layers, activation functions, loss functions
* 🔜 Upcoming: Optimizers (SGD, Momentum), L2 Regularization, and a datasets module