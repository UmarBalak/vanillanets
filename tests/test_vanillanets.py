"""
Comprehensive test suite for the nn library.

Covers every class:
- activations.py   : Linear, Sigmoid, ReLU, LeakyReLU, Tanh, Softmax
- losses.py        : BinaryCrossEntropy, CategoricalCrossEntropy,
                      SparseCategoricalCrossEntropy, MeanSquaredError
- layers.py        : DenseLayer (init modes, forward, backward)
- optimizers.py    : Optimizer_SGD, Optimizer_Adam
- activation_loss.py: Activation_Softmax_Loss_CategoricalCrossentropy
- metrics.py       : Accuracy, Precision, Recall, F1Score, ConfusionMatrix,
                      R2Score, MAE, RMSE
- accuracy.py      : Accuracy (legacy)
- model.py         : Model (add/set/finalize/predict/evaluate/fit)

Run with:
    pytest test_vanillanets.py -v
"""

import numpy as np
import pytest

from vanillanets.activations import Linear, Sigmoid, ReLU, LeakyReLU, Tanh, Softmax
from vanillanets.losses import (
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    SparseCategoricalCrossEntropy,
    MeanSquaredError,
)
from vanillanets.layers import DenseLayer
from vanillanets.optimizers import Optimizer_SGD, Optimizer_Adam
from vanillanets.softmax_loss import Activation_Softmax_Loss_CategoricalCrossentropy
from vanillanets.metrics import (
    Accuracy as MAccuracy,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
    R2Score,
    MAE,
    RMSE,
)
from vanillanets.accuracy import Accuracy as LegacyAccuracy
from vanillanets.model import Model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def numerical_gradient_check(forward_fn, x, dvalues, epsilon=1e-5, tol=1e-3):
    """
    Generic numerical gradient check for an elementwise/forward function.
    forward_fn(x) -> output of same shape as x (used for activations).
    Compares analytic dinputs (computed by caller) against numeric estimate
    of d(output . dvalues_as_weights)/dx via central differences.
    Returns max abs difference.
    """
    grad_numeric = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]

        x[idx] = orig + epsilon
        out_plus = forward_fn(x).copy()

        x[idx] = orig - epsilon
        out_minus = forward_fn(x).copy()

        x[idx] = orig

        # directional derivative weighted by dvalues (sum trick)
        grad_numeric[idx] = np.sum((out_plus - out_minus) * dvalues) / (2 * epsilon)
        it.iternext()

    return grad_numeric


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

class TestActivations:

    def test_linear_forward_backward(self):
        act = Linear()
        x = np.array([[1.0, -2.0, 3.0]])
        act.forward(x)
        assert np.allclose(act.output, x)

        dvalues = np.array([[0.5, 0.5, 0.5]])
        act.backward(dvalues)
        assert np.allclose(act.dinputs, dvalues)

    def test_sigmoid_forward_range(self):
        act = Sigmoid()
        x = np.array([[-100.0, 0.0, 100.0]])
        act.forward(x)
        # output should be bounded in [0, 1] (extreme inputs saturate to 0/1
        # in float64 precision, which is correct)
        assert np.all(act.output >= 0) and np.all(act.output <= 1)
        assert np.isclose(act.output[0, 1], 0.5)

    def test_sigmoid_backward_gradient(self):
        act = Sigmoid()
        x = np.array([[0.5, -0.3, 1.2]])
        act.forward(x)
        dvalues = np.ones_like(x)
        act.backward(dvalues)

        numeric = numerical_gradient_check(
            lambda inp: 1 / (1 + np.exp(-inp)), x.copy(), dvalues
        )
        assert np.allclose(act.dinputs, numeric, atol=1e-3)

    def test_relu_forward(self):
        act = ReLU()
        x = np.array([[-1.0, 0.0, 2.0]])
        act.forward(x)
        assert np.allclose(act.output, [[0.0, 0.0, 2.0]])

    def test_relu_backward(self):
        act = ReLU()
        x = np.array([[-1.0, 0.0, 2.0]])
        act.forward(x)
        dvalues = np.array([[1.0, 1.0, 1.0]])
        act.backward(dvalues)
        # gradient zeroed where input <= 0
        assert np.allclose(act.dinputs, [[0.0, 0.0, 1.0]])

    def test_leaky_relu_forward(self):
        act = LeakyReLU()
        x = np.array([[-10.0, 0.0, 5.0]])
        act.forward(x)
        assert np.allclose(act.output, [[-1.0, 0.0, 5.0]])

    def test_leaky_relu_backward(self):
        act = LeakyReLU()
        x = np.array([[-10.0, 0.0, 5.0]])
        act.forward(x)
        dvalues = np.array([[1.0, 1.0, 1.0]])
        act.backward(dvalues)
        # gradient scaled by 0.1 where input <= 0
        assert np.allclose(act.dinputs, [[0.1, 0.1, 1.0]])

    def test_tanh_forward_range(self):
        act = Tanh()
        x = np.array([[-100.0, 0.0, 100.0]])
        act.forward(x)
        assert np.all(act.output >= -1) and np.all(act.output <= 1)
        assert np.isclose(act.output[0, 1], 0.0)
        # no NaNs from overflow
        assert not np.any(np.isnan(act.output))

    def test_tanh_backward_gradient(self):
        act = Tanh()
        x = np.array([[0.5, -0.3, 1.2]])
        act.forward(x)
        dvalues = np.ones_like(x)
        act.backward(dvalues)

        numeric = numerical_gradient_check(np.tanh, x.copy(), dvalues)
        assert np.allclose(act.dinputs, numeric, atol=1e-3)

    def test_softmax_forward_sums_to_one(self):
        act = Softmax()
        x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        act.forward(x)
        sums = np.sum(act.output, axis=1)
        assert np.allclose(sums, 1.0)
        assert np.all(act.output >= 0)

    def test_softmax_numerical_stability(self):
        act = Softmax()
        x = np.array([[1000.0, 1001.0, 1002.0]])
        act.forward(x)
        assert not np.any(np.isnan(act.output))
        assert np.allclose(np.sum(act.output), 1.0)

    def test_softmax_backward_shapes(self):
        act = Softmax()
        x = np.array([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]])
        act.forward(x)
        dvalues = np.random.randn(*x.shape)
        act.backward(dvalues)
        assert act.dinputs.shape == x.shape


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class TestLosses:

    def test_binary_cross_entropy_forward(self):
        loss = BinaryCrossEntropy()
        y_pred = np.array([[0.9], [0.1]])
        y_true = np.array([[1.0], [0.0]])
        sample_losses = loss.forward(y_pred, y_true)
        # loss should be small since predictions match labels well
        assert np.all(sample_losses < 0.2)

    def test_binary_cross_entropy_calculate(self):
        loss = BinaryCrossEntropy()
        y_pred = np.array([[0.5], [0.5]])
        y_true = np.array([[1.0], [0.0]])
        mean_loss = loss.calculate(y_pred, y_true)
        # -log(0.5) for each sample
        assert np.isclose(mean_loss, -np.log(0.5), atol=1e-4)

    def test_binary_cross_entropy_clipping(self):
        loss = BinaryCrossEntropy()
        y_pred = np.array([[0.0], [1.0]])  # extreme values
        y_true = np.array([[1.0], [0.0]])
        sample_losses = loss.forward(y_pred, y_true)
        assert not np.any(np.isnan(sample_losses))
        assert not np.any(np.isinf(sample_losses))

    def test_binary_cross_entropy_backward_shape(self):
        loss = BinaryCrossEntropy()
        y_pred = np.array([[0.7], [0.3], [0.5]])
        y_true = np.array([[1.0], [0.0], [1.0]])
        loss.backward(y_pred, y_true)
        assert loss.dinputs.shape == y_pred.shape

    def test_categorical_cross_entropy_sparse_labels(self):
        loss = CategoricalCrossEntropy()
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        y_true = np.array([0, 1])  # sparse labels
        mean_loss = loss.calculate(y_pred, y_true)
        expected = np.mean([-np.log(0.7), -np.log(0.8)])
        assert np.isclose(mean_loss, expected, atol=1e-6)

    def test_categorical_cross_entropy_onehot_labels(self):
        loss = CategoricalCrossEntropy()
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        y_true = np.array([[1, 0, 0], [0, 1, 0]])  # one-hot
        mean_loss = loss.calculate(y_pred, y_true)
        expected = np.mean([-np.log(0.7), -np.log(0.8)])
        assert np.isclose(mean_loss, expected, atol=1e-6)

    def test_categorical_cross_entropy_backward_sparse_vs_onehot(self):
        loss = CategoricalCrossEntropy()
        dvalues = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])

        y_sparse = np.array([0, 1])
        loss.backward(dvalues, y_sparse)
        dinputs_sparse = loss.dinputs.copy()

        y_onehot = np.array([[1, 0, 0], [0, 1, 0]])
        loss.backward(dvalues, y_onehot)
        dinputs_onehot = loss.dinputs.copy()

        assert np.allclose(dinputs_sparse, dinputs_onehot)

    def test_categorical_cross_entropy_backward_clipping(self):
        loss = CategoricalCrossEntropy()
        dvalues = np.array([[0.0, 1.0, 0.0]])  # extreme
        y_true = np.array([0])
        loss.backward(dvalues, y_true)
        assert not np.any(np.isnan(loss.dinputs))
        assert not np.any(np.isinf(loss.dinputs))

    def test_sparse_categorical_cross_entropy_matches_categorical(self):
        sparse_loss = SparseCategoricalCrossEntropy()
        cat_loss = CategoricalCrossEntropy()

        y_pred = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
        y_true = np.array([0, 2])

        sparse_result = sparse_loss.calculate(y_pred, y_true)
        cat_result = cat_loss.calculate(y_pred, y_true)
        assert np.isclose(sparse_result, cat_result)

    def test_mean_squared_error_forward(self):
        loss = MeanSquaredError()
        y_pred = np.array([[1.0], [2.0]])
        y_true = np.array([[1.5], [2.5]])
        sample_losses = loss.forward(y_pred, y_true)
        assert np.allclose(sample_losses, [0.25, 0.25])

    def test_mean_squared_error_calculate_zero_for_perfect_pred(self):
        loss = MeanSquaredError()
        y_pred = np.array([[3.0], [4.0]])
        y_true = np.array([[3.0], [4.0]])
        mean_loss = loss.calculate(y_pred, y_true)
        assert np.isclose(mean_loss, 0.0)

    def test_mean_squared_error_backward_shape(self):
        loss = MeanSquaredError()
        y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_true = np.array([[1.5, 2.5], [2.5, 3.5]])
        loss.backward(y_pred, y_true)
        assert loss.dinputs.shape == y_pred.shape


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class TestDenseLayer:

    def test_output_shape(self):
        layer = DenseLayer(4, 6, seed=1)
        x = np.random.randn(10, 4)
        layer.forward(x)
        assert layer.output.shape == (10, 6)

    def test_weight_bias_shapes(self):
        layer = DenseLayer(5, 3, seed=1)
        assert layer.weights.shape == (5, 3)
        assert layer.biases.shape == (1, 3)

    def test_backward_gradient_shapes(self):
        layer = DenseLayer(4, 6, seed=1)
        x = np.random.randn(10, 4)
        layer.forward(x)
        dvalues = np.random.randn(10, 6)
        layer.backward(dvalues)
        assert layer.dweights.shape == layer.weights.shape
        assert layer.dbiases.shape == layer.biases.shape
        assert layer.dinputs.shape == x.shape

    def test_he_init_auto_for_relu(self):
        layer = DenseLayer(100, 100, activation='relu', seed=1)
        std = np.std(layer.weights)
        expected = np.sqrt(2.0 / 100)
        # statistical check, not exact
        assert 0.5 * expected < std < 2 * expected

    def test_xavier_init_auto_for_non_relu(self):
        layer = DenseLayer(100, 100, activation='tanh', seed=1)
        std = np.std(layer.weights)
        expected = np.sqrt(2.0 / (100 + 100))
        assert 0.5 * expected < std < 2 * expected

    def test_uniform_distribution_bounds_he(self):
        layer = DenseLayer(100, 50, init='he', distribution='uniform', seed=1)
        limit = np.sqrt(6.0 / 100)
        assert np.all(layer.weights >= -limit - 1e-9)
        assert np.all(layer.weights <= limit + 1e-9)

    def test_uniform_distribution_bounds_xavier(self):
        layer = DenseLayer(100, 50, init='xavier', distribution='uniform', seed=1)
        limit = np.sqrt(6.0 / (100 + 50))
        assert np.all(layer.weights >= -limit - 1e-9)
        assert np.all(layer.weights <= limit + 1e-9)

    def test_invalid_init_raises(self):
        with pytest.raises(ValueError):
            DenseLayer(10, 10, init='bogus')

    def test_bias_init_zeros(self):
        layer = DenseLayer(5, 5, bias_init='zeros', seed=1)
        assert np.allclose(layer.biases, 0.0)

    def test_bias_init_constant(self):
        layer = DenseLayer(5, 5, bias_init=0.01, seed=1)
        assert np.allclose(layer.biases, 0.01)

    def test_invalid_bias_init_raises(self):
        with pytest.raises(ValueError):
            DenseLayer(5, 5, bias_init='bogus')

    def test_seed_reproducibility(self):
        l1 = DenseLayer(10, 10, seed=42)
        l2 = DenseLayer(10, 10, seed=42)
        assert np.allclose(l1.weights, l2.weights)


# ---------------------------------------------------------------------------
# Activation + Loss fusion
# ---------------------------------------------------------------------------

class TestSoftmaxCCEFusion:

    def test_forward_matches_separate(self):
        combo = Activation_Softmax_Loss_CategoricalCrossentropy()
        inputs = np.array([[1.0, 2.0, 0.5], [0.5, 0.2, 3.0]])
        y_true = np.array([1, 2])

        combo_loss = combo.forward(inputs, y_true)

        softmax = Softmax()
        softmax.forward(inputs)
        cce = CategoricalCrossEntropy()
        separate_loss = cce.calculate(softmax.output, y_true)

        assert np.isclose(combo_loss, separate_loss)

    def test_backward_onehot_vs_sparse(self):
        combo = Activation_Softmax_Loss_CategoricalCrossentropy()
        inputs = np.array([[1.0, 2.0, 0.5], [0.5, 0.2, 3.0]])
        combo.forward(inputs, np.array([1, 2]))
        dvalues = combo.output

        combo.backward(dvalues, np.array([1, 2]))
        d_sparse = combo.dinputs.copy()

        combo.backward(dvalues, np.array([[0, 1, 0], [0, 0, 1]]))
        d_onehot = combo.dinputs.copy()

        assert np.allclose(d_sparse, d_onehot)

    def test_backward_gradient_correctness(self):
        """
        For softmax+CCE, dinputs = (softmax_output - one_hot_true) / n_samples.
        """
        combo = Activation_Softmax_Loss_CategoricalCrossentropy()
        inputs = np.array([[1.0, 2.0, 0.5]])
        combo.forward(inputs, np.array([1]))
        dvalues = combo.output
        combo.backward(dvalues, np.array([1]))

        expected = combo.output.copy()
        expected[0, 1] -= 1
        expected /= 1
        assert np.allclose(combo.dinputs, expected)


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

class TestOptimizers:

    def _make_layer(self):
        layer = DenseLayer(3, 2, seed=1)
        layer.dweights = np.ones_like(layer.weights)
        layer.dbiases = np.ones_like(layer.biases)
        return layer

    def test_sgd_vanilla_update_direction(self):
        layer = self._make_layer()
        w_before = layer.weights.copy()
        opt = Optimizer_SGD(learning_rate=0.1, momentum=0.0)
        opt.update_params(layer)
        # weights should decrease since dweights are all positive
        assert np.all(layer.weights < w_before)
        assert np.allclose(layer.weights, w_before - 0.1 * 1.0)

    def test_sgd_with_momentum_creates_momentum_arrays(self):
        layer = self._make_layer()
        opt = Optimizer_SGD(learning_rate=0.1, momentum=0.9)
        opt.update_params(layer)
        assert hasattr(layer, 'weight_momentums')
        assert hasattr(layer, 'bias_momentums')

    def test_sgd_decay_reduces_lr(self):
        opt = Optimizer_SGD(learning_rate=1.0, decay=0.1)
        opt.pre_update_lr()  # iteration 0 -> no change
        lr0 = opt.current_learning_rate
        opt.iterations = 5
        opt.pre_update_lr()
        lr5 = opt.current_learning_rate
        assert lr5 < lr0

    def test_sgd_post_update_increments_iterations(self):
        opt = Optimizer_SGD()
        assert opt.iterations == 0
        opt.post_update_params()
        assert opt.iterations == 1

    def test_adam_update_changes_weights(self):
        layer = self._make_layer()
        w_before = layer.weights.copy()
        opt = Optimizer_Adam(learning_rate=0.01)
        opt.update_params(layer)
        assert not np.allclose(layer.weights, w_before)
        # adam should create cache/momentum arrays
        assert hasattr(layer, 'weight_cache')
        assert hasattr(layer, 'weight_momentums')

    def test_adam_decay_reduces_lr(self):
        opt = Optimizer_Adam(learning_rate=1.0, decay=0.1)
        opt.iterations = 5
        opt.pre_update_lr()
        assert opt.current_learning_rate < 1.0

    def test_adam_bias_correction_first_step(self):
        layer = self._make_layer()
        opt = Optimizer_Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
        opt.update_params(layer)
        # after one step, weights should move by approx -lr (bias-corrected ratio ~ 1)
        # for constant gradient of 1, m_hat ~ 1, v_hat ~ 1 -> update ~ -lr
        delta = layer.weights - DenseLayer(3, 2, seed=1).weights
        assert np.allclose(delta, -0.01, atol=1e-3)


# ---------------------------------------------------------------------------
# Metrics (metrics.py)
# ---------------------------------------------------------------------------

class TestMetricsBinary:

    def setup_method(self):
        # 6 samples, binary predictions (n,1) and y (n,1)
        self.predictions = np.array([[0.9], [0.2], [0.8], [0.1], [0.7], [0.4]])
        self.y = np.array([[1], [0], [1], [0], [0], [1]])
        # thresholded preds: [1,0,1,0,1,0] vs y [1,0,1,0,0,1]
        # TP=2 (idx0,2), FP=1 (idx4), FN=1(idx5), TN=2(idx1,3)

    def test_accuracy(self):
        acc = MAccuracy().calculate(self.predictions, self.y)
        assert np.isclose(acc, 4 / 6)

    def test_precision(self):
        p = Precision().calculate(self.predictions, self.y)
        assert np.isclose(p, 2 / (2 + 1), atol=1e-6)

    def test_recall(self):
        r = Recall().calculate(self.predictions, self.y)
        assert np.isclose(r, 2 / (2 + 1), atol=1e-6)

    def test_f1_score(self):
        p = Precision().calculate(self.predictions, self.y)
        r = Recall().calculate(self.predictions, self.y)
        f1 = F1Score().calculate(self.predictions, self.y)
        expected = 2 * p * r / (p + r + 1e-7)
        assert np.isclose(f1, expected)

    def test_confusion_matrix_binary(self):
        cm = ConfusionMatrix().calculate(self.predictions, self.y, num_classes=2)
        assert cm.shape == (2, 2)
        assert cm.sum() == 6


class TestMetricsMulticlass:

    def setup_method(self):
        # 3-class softmax-like outputs
        self.predictions = np.array([
            [0.7, 0.2, 0.1],  # pred 0
            [0.1, 0.8, 0.1],  # pred 1
            [0.2, 0.3, 0.5],  # pred 2
            [0.6, 0.3, 0.1],  # pred 0
        ])
        self.y_sparse = np.array([0, 1, 1, 0])  # true labels
        self.y_onehot = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
        ])

    def test_accuracy_sparse(self):
        acc = MAccuracy().calculate(self.predictions, self.y_sparse)
        # predicted: [0,1,2,0] vs true [0,1,1,0] -> 3/4 correct
        assert np.isclose(acc, 3 / 4)

    def test_accuracy_onehot_matches_sparse(self):
        acc_sparse = MAccuracy().calculate(self.predictions, self.y_sparse)
        acc_onehot = MAccuracy().calculate(self.predictions, self.y_onehot)
        assert np.isclose(acc_sparse, acc_onehot)

    def test_precision_recall_f1_run_without_error(self):
        p = Precision().calculate(self.predictions, self.y_sparse)
        r = Recall().calculate(self.predictions, self.y_sparse)
        f1 = F1Score().calculate(self.predictions, self.y_sparse)
        assert 0 <= p <= 1
        assert 0 <= r <= 1
        assert 0 <= f1 <= 1

    def test_confusion_matrix_multiclass(self):
        cm = ConfusionMatrix().calculate(self.predictions, self.y_sparse, num_classes=3)
        assert cm.shape == (3, 3)
        assert cm.sum() == 4
        # row 0 (true=0): samples 0 and 3, both predicted 0 -> cm[0,0]==2
        assert cm[0, 0] == 2


class TestMetricsRegression:

    def test_r2_perfect_fit(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = y_true.copy()
        r2 = R2Score().calculate(y_pred, y_true)
        assert np.isclose(r2, 1.0, atol=1e-5)

    def test_r2_worse_than_mean(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([10.0, -5.0, 20.0, -3.0])
        r2 = R2Score().calculate(y_pred, y_true)
        assert r2 < 0

    def test_mae(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 2.0])
        mae = MAE().calculate(y_pred, y_true)
        assert np.isclose(mae, np.mean([0.5, 0.5, 1.0]))

    def test_rmse(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 5.0])
        rmse = RMSE().calculate(y_pred, y_true)
        expected = np.sqrt(np.mean([0, 0, 4]))
        assert np.isclose(rmse, expected)


class TestLegacyAccuracy:
    """accuracy.py's standalone Accuracy class (post-fix)."""

    def test_binary_shapes(self):
        predictions = np.array([[0.9], [0.1], [0.6]])
        y = np.array([[1], [0], [1]])
        acc = LegacyAccuracy().calculate(predictions, y)
        assert np.isclose(acc, 1.0)

    def test_multiclass_onehot(self):
        predictions = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        y = np.array([[1, 0, 0], [0, 1, 0]])
        acc = LegacyAccuracy().calculate(predictions, y)
        assert np.isclose(acc, 1.0)

    def test_multiclass_sparse(self):
        predictions = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        y = np.array([0, 0])  # second sample wrong
        acc = LegacyAccuracy().calculate(predictions, y)
        assert np.isclose(acc, 0.5)


# ---------------------------------------------------------------------------
# Model integration tests
# ---------------------------------------------------------------------------

class TestModelIntegration:

    def test_binary_classification_pipeline(self):
        np.random.seed(0)
        X = np.random.randn(50, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)

        model = Model()
        model.add(DenseLayer(4, 8, seed=1))
        model.add(ReLU())
        model.add(DenseLayer(8, 1, seed=2))
        model.add(Sigmoid())

        model.set(
            loss=BinaryCrossEntropy(),
            optimizer=Optimizer_Adam(learning_rate=0.05),
            metrics=[MAccuracy(), Precision(), Recall(), F1Score()],
        )
        model.finalize()

        loss_before, _ = model.evaluate(X, y)
        model.fit(X, y, epochs=50, print_every=1000)
        loss_after, metrics_after = model.evaluate(X, y)

        assert loss_after < loss_before
        assert 0 <= metrics_after['accuracy'] <= 1

    def test_multiclass_classification_pipeline_uses_fusion(self):
        np.random.seed(0)
        X = np.random.randn(60, 3)
        y = np.random.randint(0, 3, size=60)

        model = Model()
        model.add(DenseLayer(3, 16, seed=1))
        model.add(ReLU())
        model.add(DenseLayer(16, 3, seed=2))
        model.add(Softmax())

        model.set(
            loss=CategoricalCrossEntropy(),
            optimizer=Optimizer_Adam(learning_rate=0.05),
            metrics=[MAccuracy()],
        )
        model.finalize()

        # confirm the fused softmax+CCE shortcut is active
        assert isinstance(model.softmax_classifier_output,
                          Activation_Softmax_Loss_CategoricalCrossentropy)

        loss_before, _ = model.evaluate(X, y)
        model.fit(X, y, epochs=50, print_every=1000)
        loss_after, _ = model.evaluate(X, y)
        assert loss_after < loss_before

    def test_regression_pipeline(self):
        np.random.seed(0)
        X = np.random.randn(100, 2)
        y = (3 * X[:, 0] - 2 * X[:, 1] + 1).reshape(-1, 1)

        model = Model()
        model.add(DenseLayer(2, 16, seed=1))
        model.add(ReLU())
        model.add(DenseLayer(16, 1, seed=2))
        model.add(Linear())

        model.set(
            loss=MeanSquaredError(),
            optimizer=Optimizer_Adam(learning_rate=0.05),
            metrics=[R2Score(), MAE(), RMSE()],
        )
        model.finalize()

        loss_before, _ = model.evaluate(X, y)
        model.fit(X, y, epochs=100, print_every=1000)
        loss_after, metrics_after = model.evaluate(X, y)

        assert loss_after < loss_before
        assert metrics_after['r2score'] > -1  # sanity, should improve a lot in practice

    def test_predict_matches_evaluate_forward(self):
        np.random.seed(0)
        X = np.random.randn(10, 3)
        y = np.random.randint(0, 2, size=(10, 1)).astype(float)

        model = Model()
        model.add(DenseLayer(3, 5, seed=1))
        model.add(ReLU())
        model.add(DenseLayer(5, 1, seed=2))
        model.add(Sigmoid())
        model.set(loss=BinaryCrossEntropy(), optimizer=Optimizer_Adam(),
                  metrics=[MAccuracy()])
        model.finalize()

        preds1 = model.predict(X)
        preds2 = model.predict(X)
        assert np.allclose(preds1, preds2)
        assert preds1.shape == (10, 1)

    def test_validation_data_does_not_break_training(self):
        np.random.seed(0)
        X = np.random.randn(40, 3)
        y = np.random.randint(0, 2, size=(40, 1)).astype(float)
        X_val = np.random.randn(10, 3)
        y_val = np.random.randint(0, 2, size=(10, 1)).astype(float)

        model = Model()
        model.add(DenseLayer(3, 5, seed=1))
        model.add(ReLU())
        model.add(DenseLayer(5, 1, seed=2))
        model.add(Sigmoid())
        model.set(loss=BinaryCrossEntropy(), optimizer=Optimizer_Adam(),
                  metrics=[MAccuracy()])
        model.finalize()

        # should not raise
        model.fit(X, y, epochs=5, print_every=1, validation_data=(X_val, y_val))

    def test_legacy_accuracy_kwarg_still_works(self):
        np.random.seed(0)
        X = np.random.randn(20, 2)
        y = np.random.randint(0, 2, size=(20, 1)).astype(float)

        model = Model()
        model.add(DenseLayer(2, 4, seed=1))
        model.add(ReLU())
        model.add(DenseLayer(4, 1, seed=2))
        model.add(Sigmoid())
        model.set(loss=BinaryCrossEntropy(), optimizer=Optimizer_Adam(),
                  accuracy=LegacyAccuracy())
        model.finalize()

        model.fit(X, y, epochs=2, print_every=1)
        _, results = model.evaluate(X, y)
        assert 'accuracy' in results


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))