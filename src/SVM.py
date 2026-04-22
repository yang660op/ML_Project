import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iter = n_iterations
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        # 将标签转为 +1 和 -1
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iter):
            for i, x_i in enumerate(X):
                condition = y_[i] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.w - np.dot(x_i, y_[i])
                    db = -y_[i]
                self.w -= self.lr * dw
                self.b -= self.lr * db
    
    def predict(self, X):
        linear = np.dot(X, self.w) + self.b
        return np.where(linear >= 0, 1, 0)

def svm_classify(X_train, y_train, X_test, lr=0.001, lambda_param=0.01, n_iter=1000):
    """函数形式SVM二分类，返回预测标签(0/1)"""
    model = SVM(learning_rate=lr, lambda_param=lambda_param, n_iterations=n_iter)
    model.fit(X_train, y_train)
    return model.predict(X_test)