import numpy as np

# Common loss class
class Loss:

    """
    Calculate mean loss between actual value and predicted value
    """

    # Calculate the data and regularization losses given model output and ground truth values
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        return data_loss
    
class BinaryCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """
        Returns shape (n_samples,) when y_pred and y_true are both shape (n_samples,)

        """

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Binary cross-entropy formula
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

        return sample_losses
    
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        
        # Number of outputs in every sample
        outputs = len(dvalues[0])

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs

        # Normalize gradient across the batch
        self.dinputs = self.dinputs / samples
    
# Cross-entropy loss
class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """
        Returns shape (n_samples,) after extracting correct_confidences

        """
        
        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values
        # Only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = np.array([y_pred_clipped[i, y_true[i]] for i in range(samples)])
        
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # Losses
        negative_log_likelihood = -np.log(correct_confidences)

        return negative_log_likelihood
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Clip data to prevent division by 0
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate gradient using the clipped values
        self.dinputs = -y_true / clipped_dvalues

        # Normalize gradient across the batch
        self.dinputs = self.dinputs / samples
    
# Cross-entropy loss
class SparseCategoricalCrossEntropy(CategoricalCrossEntropy):
    """
    Accepts integer class labels (y_true shape (n,)).
    forward/backward inherited from CategoricalCrossEntropy, which
    already branches on len(y_true.shape) == 1 for the sparse case.
    """
    pass
        

class MeanSquaredError(Loss):

    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        
        # Number of outputs in every sample
        outputs = len(dvalues[0])

        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs

        # Normalize gradient across the batch
        self.dinputs = self.dinputs / samples