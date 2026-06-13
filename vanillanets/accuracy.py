import numpy as np

class Accuracy:
    # Calculates accuracy based on predictions and ground truth
    def calculate(self, predictions, y):
        # Handle binary classification (single output)
        if predictions.shape[1] == 1:
            # Binary case: threshold at 0.5
            predictions = (predictions > 0.5) * 1
            predictions = predictions.flatten()

            # y for binary is shape (n, 1), NOT one-hot -> just flatten
            if len(y.shape) == 2:
                y = y.flatten()
        else:
            # Multiclass case: use argmax
            predictions = np.argmax(predictions, axis=1)

            # Handle one-hot encoded y
            if len(y.shape) == 2:
                y = np.argmax(y, axis=1)

        return np.mean(predictions == y)