import numpy as np
from activations import Softmax
from losses import CategoricalCrossEntropy
from softmax_loss import Activation_Softmax_Loss_CategoricalCrossentropy


class Model:

    def __init__(self):
        # Create a list of network objects
        self.layers = []
        # Catch-all object for the fast Softmax+CrossEntropy backward pass
        self.softmax_classifier_output = None
        self.metrics = {}

    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)

    # Set loss, optimizer and metrics
    def set(self, *, loss, optimizer, accuracy=None, metrics=None):
        """
        loss: a loss instance (CategoricalCrossEntropy, BinaryCrossEntropy, MeanSquaredError)
        optimizer: an optimizer instance
        accuracy: (legacy) single metric object with a .calculate(predictions, y) method.
                  Kept for backward compatibility - automatically folded into `metrics`.
        metrics: dict of {name: metric_object} or list of metric objects.
                 Each metric object must implement .calculate(predictions, y).
                 Examples: Accuracy(), Precision(), Recall(), F1Score(),
                           R2Score(), MAE(), RMSE()
        """
        self.loss = loss
        self.optimizer = optimizer

        self.metrics = {}

        # Backward-compatible single 'accuracy' metric
        if accuracy is not None:
            self.metrics['accuracy'] = accuracy

        # New flexible metrics interface
        if metrics is not None:
            if isinstance(metrics, dict):
                self.metrics.update(metrics)
            else:
                # list/tuple of metric objects -> derive names from class names
                for m in metrics:
                    name = type(m).__name__.lower()
                    self.metrics[name] = m

    # Finalize the model setup
    def finalize(self):
        # If the last layer is Softmax and the loss is Categorical Cross-Entropy,
        # we create the combined object for a much faster backward pass
        if self.loss is not None and isinstance(self.layers[-1], Softmax) and \
           isinstance(self.loss, CategoricalCrossEntropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()

    # Run a forward pass through all layers and return final output
    def predict(self, X):
        layer_input = X
        for layer in self.layers:
            layer.forward(layer_input)
            layer_input = layer.output
        return layer_input

    # Compute loss and all configured metrics for a given dataset
    # without performing any backward pass / parameter updates
    def evaluate(self, X, y):
        predictions = self.predict(X)
        loss_value = self.loss.calculate(predictions, y)

        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric.calculate(predictions, y)

        return loss_value, results

    # Train the model
    def fit(self, X, y, *, epochs=1, print_every=100, validation_data=None):
        """
        validation_data: optional (X_val, y_val) tuple. If provided, validation
        loss and metrics are computed (without affecting training) and printed
        alongside training stats.
        """

        # Main training loop
        for epoch in range(1, epochs + 1):

            # --- FORWARD PASS ---
            # The initial input is our training data
            layer_input = X

            # Forward pass through all layers in the list
            for layer in self.layers:
                layer.forward(layer_input)
                # The output of this layer becomes the input of the next layer
                layer_input = layer.output

            # Calculate loss from the output of the final layer
            data_loss = self.loss.calculate(layer_input, y)

            # --- METRICS ---
            metric_results = {}
            for name, metric in self.metrics.items():
                metric_results[name] = metric.calculate(layer_input, y)

            # --- BACKWARD PASS ---

            # Check if we are using the fast Softmax+CCE combination
            if self.softmax_classifier_output is not None:
                # Do the fast backward pass
                self.softmax_classifier_output.backward(layer_input, y)
                # The gradient to pass back comes from this fused object
                dinputs = self.softmax_classifier_output.dinputs

                # We safely ignore the standalone Softmax layer for the backward loop
                layers_to_backprop = self.layers[:-1]

            else:
                # ONLY if we aren't using the shortcut, calculate standalone loss gradient
                self.loss.backward(layer_input, y)
                dinputs = self.loss.dinputs

                # Backpropagate through all layers normally
                layers_to_backprop = self.layers

            # Loop backward through the remaining layers
            for layer in reversed(layers_to_backprop):
                layer.backward(dinputs)
                dinputs = layer.dinputs

            # --- OPTIMIZATION ---
            self.optimizer.pre_update_lr()

            # We only update parameters for layers that actually have weights (DenseLayers)
            for layer in self.layers:
                if hasattr(layer, 'weights'):
                    self.optimizer.update_params(layer)

            # Increment iteration counter
            self.optimizer.post_update_params()

            # --- VALIDATION (optional) ---
            # Run after backward/update so this forward pass doesn't
            # clobber the cached layer state used during backprop.
            val_loss = None
            val_metric_results = {}
            if validation_data is not None:
                X_val, y_val = validation_data
                val_loss, val_metric_results = self.evaluate(X_val, y_val)

            # Print status updates
            if not epoch % print_every:
                metric_str = ', '.join(
                    f'{name}: {value:.3f}' for name, value in metric_results.items()
                )
                line = f'epoch: {epoch}'
                if metric_str:
                    line += f', {metric_str}'
                line += f', loss: {data_loss:.3f}'
                line += f', lr: {self.optimizer.current_learning_rate:.6f}'

                if validation_data is not None:
                    val_metric_str = ', '.join(
                        f'val_{name}: {value:.3f}' for name, value in val_metric_results.items()
                    )
                    line += f', val_loss: {val_loss:.3f}'
                    if val_metric_str:
                        line += f', {val_metric_str}'

                print(line)