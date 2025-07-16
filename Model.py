
import numpy as np

def sigmoid(x):
    """Compute the sigmoid function element-wise on NumPy arrays or scalars."""
    return 1 / (1 + np.exp(-x))

class LogisticRegression():
    def __init__(self, lr=0.01, n_iters=1000 , threshold = 0.5):
        self.lr = lr
        self.thr = threshold
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None

    def _standardize(self, X):
        return (X - self.mean) / self.std

    def fit(self, X, y):
        # Store mean and std for use during prediction
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X = self._standardize(X)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self , X):
        X = (X - self.mean) / self.std  # Apply same scaling as during training
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        return [0 if y < self.thr else 1 for y in y_pred]
