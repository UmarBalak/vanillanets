import numpy as np


def _to_labels(predictions, y):
    """
    Convert raw model outputs and ground truth into 1D label arrays
    suitable for classification metrics.

    - predictions: (n, 1) -> threshold at 0.5 -> (n,)
                    (n, C) -> argmax over classes -> (n,)
    - y:           (n, 1) -> flatten -> (n,)        [binary, NOT one-hot]
                    (n, C) -> argmax over classes -> (n,) [one-hot, multiclass]
                    (n,)   -> unchanged
    """
    if predictions.shape[1] == 1:
        predictions = (predictions > 0.5).astype(int).flatten()
        if len(y.shape) == 2:
            y = y.flatten()
    else:
        predictions = np.argmax(predictions, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)

    return predictions, y.astype(int)


class Accuracy:
    """Fraction of correct predictions. Works for binary and multiclass."""

    def calculate(self, predictions, y):
        predictions, y = _to_labels(predictions, y)
        return np.mean(predictions == y)


class Precision:
    """
    Binary precision: TP / (TP + FP).
    For multiclass, computes macro-averaged precision (mean over classes).
    """

    def calculate(self, predictions, y):
        predictions, y = _to_labels(predictions, y)
        classes = np.unique(np.concatenate([predictions, y]))

        if len(classes) <= 2:
            tp = np.sum((predictions == 1) & (y == 1))
            fp = np.sum((predictions == 1) & (y == 0))
            return tp / (tp + fp + 1e-7)

        # macro-average over classes
        scores = []
        for c in classes:
            tp = np.sum((predictions == c) & (y == c))
            fp = np.sum((predictions == c) & (y != c))
            scores.append(tp / (tp + fp + 1e-7))
        return np.mean(scores)


class Recall:
    """
    Binary recall: TP / (TP + FN).
    For multiclass, computes macro-averaged recall (mean over classes).
    """

    def calculate(self, predictions, y):
        predictions, y = _to_labels(predictions, y)
        classes = np.unique(np.concatenate([predictions, y]))

        if len(classes) <= 2:
            tp = np.sum((predictions == 1) & (y == 1))
            fn = np.sum((predictions == 0) & (y == 1))
            return tp / (tp + fn + 1e-7)

        scores = []
        for c in classes:
            tp = np.sum((predictions == c) & (y == c))
            fn = np.sum((predictions != c) & (y == c))
            scores.append(tp / (tp + fn + 1e-7))
        return np.mean(scores)


class F1Score:
    """Harmonic mean of precision and recall."""

    def calculate(self, predictions, y):
        p = Precision().calculate(predictions, y)
        r = Recall().calculate(predictions, y)
        return 2 * p * r / (p + r + 1e-7)


class ConfusionMatrix:
    """
    Returns an (n_classes, n_classes) integer matrix where
    rows = true labels, columns = predicted labels.

    num_classes can be passed explicitly (recommended), otherwise
    it is inferred from the data (max label + 1), which may miss
    classes absent from a given batch.
    """

    def calculate(self, predictions, y, num_classes=None):
        predictions, y = _to_labels(predictions, y)
        n = num_classes if num_classes is not None else int(max(predictions.max(), y.max())) + 1

        cm = np.zeros((n, n), dtype=int)
        for true_label, pred_label in zip(y, predictions):
            cm[true_label, pred_label] += 1
        return cm


class R2Score:
    """Coefficient of determination for regression. 1.0 is a perfect fit."""

    def calculate(self, y_pred, y_true):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / (ss_tot + 1e-7)


class MAE:
    """Mean Absolute Error for regression."""

    def calculate(self, y_pred, y_true):
        return np.mean(np.abs(y_true - y_pred))


class RMSE:
    """Root Mean Squared Error for regression."""

    def calculate(self, y_pred, y_true):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))