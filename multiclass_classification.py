import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from layers import DenseLayer
from activations import ReLU, Softmax
from losses import CategoricalCrossEntropy
from optimizers import Optimizer_Adam
from model import Model
from metrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix

# 1. Load and prepare the data
X, y = load_digits(return_X_y=True)
X = X / 16.0  # Scale pixel values to 0-1

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build the Model
model = Model()
model.add(DenseLayer(64, 128))  # 64 pixels in
model.add(ReLU())
model.add(DenseLayer(128, 10))  # 10 digits out
model.add(Softmax())

# 3. Compile and Train
model.set(
    loss=CategoricalCrossEntropy(),
    optimizer=Optimizer_Adam(learning_rate=0.05, decay=1e-4),
    metrics=[Accuracy(), Precision(), Recall(), F1Score()]
)
model.finalize()

print("Training on Handwritten Digits...")
model.fit(X_train, y_train, epochs=10, print_every=1,
          validation_data=(X_test, y_test))

print("\nFinal test set performance:")
test_loss, test_metrics = model.evaluate(X_test, y_test)
print(f"loss: {test_loss:.3f}")
for name, value in test_metrics.items():
    print(f"{name}: {value:.3f}")

# Confusion matrix on test set
cm = ConfusionMatrix().calculate(model.predict(X_test), y_test, num_classes=10)
print("\nConfusion matrix (rows=true, cols=predicted):")
print(cm)