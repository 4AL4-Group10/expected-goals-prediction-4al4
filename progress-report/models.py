"""Logistic Regression following Tutorial 3."""
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=256, seed=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.X = None
        self.X_scaled = None
        self.y = None
        # Learned values are populated by fit once the feature count is known.
        self.mean = None
        self.std = None
        self.w = None
        self.b = None
        self.loss_history = []

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        
        # Estimate scaling from training data only; constant columns stay zero.
        self.mean = self.X.mean(axis=0)
        self.std = self.X.std(axis=0)
        self.std[self.std == 0] = 1.0
        self.X_scaled = (self.X - self.mean) / self.std

        self.w = np.zeros(self.X.shape[1])
        self.b = 0.0
        self.loss_history = []

        rng = np.random.default_rng(self.seed)
        for epoch in range(self.epochs):
            order = rng.permutation(len(self.X))
            for start in range(0, len(self.X), self.batch_size):
                batch = order[start:start + self.batch_size]
                errors = self.sigmoid(self.X_scaled[batch] @ self.w + self.b) - self.y[batch]
                dw = self.X_scaled[batch].T @ errors / len(batch)
                db = np.mean(errors)
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
            loss = self.cross_entropy(self.X, self.y)
            if not np.isfinite(loss):
                raise ValueError("Nonfinite loss; reduce the learning rate.")
            self.loss_history.append(loss)
        return self

    def sigmoid(self, z):
        return np.exp(-np.logaddexp(0, -np.asarray(z)))

    def cross_entropy(self, X, y):
        X_scaled = (np.asarray(X, dtype=float) - self.mean) / self.std
        z = X_scaled @ self.w + self.b
        return np.mean(np.logaddexp(0, z) - np.asarray(y) * z)

    def predict_probability(self, X):
        X_raw = np.asarray(X, dtype=float)
        X_scaled = (X_raw - self.mean) / self.std
        p = self.sigmoid(X_scaled @ self.w + self.b)
        return p

    def predict(self, X, threshold=0.5):
        return (self.predict_probability(X) >= threshold).astype(int)

class ConstantBaseline:
    def __init__(self, majority=False):
        self.majority = majority
        self.X = None
        self.y = None
        self.probability = None

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.probability = float(np.mean(self.y))
        if self.majority:
            self.probability = float(self.probability > 0.5)
        return self

    def predict_probability(self, X):
        p = np.full(len(X), self.probability)
        return p
    
    def cross_entropy(self, X, y):
        # Clip hard baseline probabilities so log(0) stays finite.
        p = np.clip(self.predict_probability(X), 1e-12, 1 - 1e-12)
        y = np.asarray(y)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def make_baselines():
    return {"majority": ConstantBaseline(majority=True), "constant_probability": ConstantBaseline()}
