import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        preds = [self._predict_one(x) for x in X]
        return np.array(preds)
    
    def _predict_one(self, x):
        # 计算欧氏距离
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]

def knn_classify(X_train, y_train, X_test, k=3):
    """函数形式的KNN，返回预测标签"""
    knn = KNN(k=k)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)