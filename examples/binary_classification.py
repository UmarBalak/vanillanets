import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from vanillanets.layers import DenseLayer
from vanillanets.activations import ReLU, Sigmoid
from vanillanets.losses import BinaryCrossEntropy
from vanillanets.optimizers import Optimizer_Adam
from vanillanets.model import Model
from vanillanets.metrics import Accuracy, Precision, Recall, F1Score

# 1. Load and prepare the data
data = load_breast_cancer()
X, y = data.data, data.target
y = y.reshape(-1, 1)  # Binary cross entropy expects a 2D column vector

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build the Model
model = Model()
model.add(DenseLayer(30, 64))  # 30 input features
model.add(ReLU())
model.add(DenseLayer(64, 1))   # 1 binary output
model.add(Sigmoid())

# 3. Compile and Train
model.set(
    loss=BinaryCrossEntropy(),
    optimizer=Optimizer_Adam(learning_rate=0.01, decay=1e-4),
    metrics=[Accuracy(), Precision(), Recall(), F1Score()]
)
model.finalize()

print("Training on Breast Cancer Dataset...")
model.fit(X_train, y_train, epochs=10, print_every=1,
          validation_data=(X_test, y_test))

print("\nFinal test set performance:")
test_loss, test_metrics = model.evaluate(X_test, y_test)
print(f"loss: {test_loss:.3f}")
for name, value in test_metrics.items():
    print(f"{name}: {value:.3f}")