
import numpy as np
from scipy.stats import mode
def sigmoid(x):
    """Compute the sigmoid function element-wise on NumPy arrays or scalars."""
    return 1 / (1 + np.exp(-x))
def _standardize(X , mean , std):
        return (X - mean) / std
class LogisticRegression():
    def __init__(self, lr=0.01, n_iters=1000 , threshold = 0.5):
        self.lr = lr
        self.thr = threshold
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None

    def fit(self, X, y):
        # Store mean and std for use during prediction
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X = _standardize(X , self.mean , self.std)
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
        X = _standardize(X , self.mean , self.std)  # Apply same scaling as during training
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        return [0 if y < self.thr else 1 for y in y_pred]


class K_nearest_neighbors:
    def __init__(self, nb_neighbors = 3):
        self.k =  nb_neighbors
        self.X = None
        self.Y = None
        self.mean = None
        self.std = None
    def fit(self , X , Y):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.X = _standardize(X , self.mean , self.std)
        self.Y = Y
    
    def euclidean_distance(self , newX):
        return np.sqrt(np.sum((self.X - newX) ** 2, axis=1))
    def predict(self , newX):
        newX = _standardize(newX , self.mean , self.std)
        distances = self.euclidean_distance(newX)
        top_k_indexes = np.argsort(distances)[:self.k]
        top_k_Y = self.Y[top_k_indexes]
        return mode(top_k_Y, keepdims=True).mode[0]
    