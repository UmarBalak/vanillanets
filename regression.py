import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from layers import DenseLayer
from activations import ReLU, Linear
from losses import MeanSquaredError
from optimizers import Optimizer_Adam
from model import Model
from metrics import R2Score, MAE, RMSE

# 1. Load and prepare the data
X, y = fetch_california_housing(return_X_y=True)
y = y.reshape(-1, 1)  # Ensure target is a column vector

# For a pure NumPy library, 20,640 samples in one massive matrix multiplication is heavy.
# We will grab a random subset of 15,000 samples to keep training fast.
indices = np.random.choice(len(X), 15000, replace=False)
X, y = X[indices], y[indices]

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build the Model
model = Model()
model.add(DenseLayer(8, 64))  # 8 input features
model.add(ReLU())
model.add(DenseLayer(64, 64))
model.add(ReLU())
model.add(DenseLayer(64, 1))  # 1 price output
model.add(Linear())

# 3. Compile and Train
model.set(
    loss=MeanSquaredError(),
    optimizer=Optimizer_Adam(learning_rate=0.01, decay=1e-3),
    metrics=[R2Score(), MAE(), RMSE()]
)
model.finalize()

print("Training on California Housing Subset...")
model.fit(X_train, y_train, epochs=100, print_every=10,
          validation_data=(X_test, y_test))

print("\nFinal test set performance:")
test_loss, test_metrics = model.evaluate(X_test, y_test)
print(f"loss: {test_loss:.3f}")
for name, value in test_metrics.items():
    print(f"{name}: {value:.3f}")