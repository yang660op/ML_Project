import numpy as np

class LinearRegression:
    def __init__(self, method='normal', batch_size=None, learning_rate=0.01, n_iterations=1000):
        """
        method: 'normal' 最小二乘求解, 'gd' 梯度下降
        batch_size: 仅gd有效，若为None则全批量，否则小批量
        """
        self.method = method
        self.batch_size = batch_size
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        X_with_bias = np.c_[np.ones(n_samples), X]  # 添加偏置列
        if self.method == 'normal':
            # 最小二乘解 (X^T X)^{-1} X^T y
            self.weights = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        elif self.method == 'gd':
            self.weights = np.zeros(n_features + 1)
            if self.batch_size is None:
                # 全批量梯度下降
                for _ in range(self.n_iter):
                    y_pred = X_with_bias @ self.weights
                    gradient = (2/n_samples) * X_with_bias.T @ (y_pred - y)
                    self.weights -= self.lr * gradient
            else:
                # 小批量梯度下降
                for _ in range(self.n_iter):
                    indices = np.random.permutation(n_samples)
                    X_shuffled = X_with_bias[indices]
                    y_shuffled = y[indices]
                    for i in range(0, n_samples, self.batch_size):
                        X_batch = X_shuffled[i:i+self.batch_size]
                        y_batch = y_shuffled[i:i+self.batch_size]
                        y_pred = X_batch @ self.weights
                        gradient = (2/len(y_batch)) * X_batch.T @ (y_pred - y_batch)
                        self.weights -= self.lr * gradient
        else:
            raise ValueError("method must be 'normal' or 'gd'")
    
    def predict(self, X):
        n_samples = X.shape[0]
        X_with_bias = np.c_[np.ones(n_samples), X]
        return X_with_bias @ self.weights

def linear_regression_predict(X_train, y_train, X_test, method='normal', batch_size=None, lr=0.01, n_iter=1000):
    """函数形式线性回归，返回预测值"""
    model = LinearRegression(method=method, batch_size=batch_size, learning_rate=lr, n_iterations=n_iter)
    model.fit(X_train, y_train)
    return model.predict(X_test)