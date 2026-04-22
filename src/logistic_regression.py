import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, fit_intercept=True):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.fit_intercept = fit_intercept
        self.weights = None
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        if self.fit_intercept:
            X = np.c_[np.ones(n_samples), X]
            n_features += 1
        self.weights = np.zeros(n_features)
        for _ in range(self.n_iter):
            linear = X.dot(self.weights)
            y_pred = self._sigmoid(linear)
            dw = (1/n_samples) * X.T.dot(y_pred - y)
            self.weights -= self.lr * dw
    
    def predict_proba(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        linear = X.dot(self.weights)
        return self._sigmoid(linear)
    
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

def logistic_regression_classify(X_train, y_train, X_test, lr=0.01, n_iter=1000):
    model = LogisticRegression(learning_rate=lr, n_iterations=n_iter)
    model.fit(X_train, y_train)
    return model.predict(X_test)