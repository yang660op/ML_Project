import numpy as np

class ANN:
    def __init__(self, layer_sizes, learning_rate=0.01, n_iterations=1000):
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = []
        self.biases = []
        self.losses = []
        
    def _init_weights(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.01
            b = np.zeros((1, self.layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _cross_entropy(self, y_pred, y_true):
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(y_pred)) / m
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights)):
            z = self.activations[-1].dot(self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            if i == len(self.weights) - 1:
                a = self._softmax(z)
            else:
                a = self._relu(z)
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, y_true):
        m = y_true.shape[0]
        deltas = [None] * len(self.weights)
        
        output_error = self.activations[-1] - y_true
        deltas[-1] = output_error
        
        for i in range(len(self.weights) - 2, -1, -1):
            deltas[i] = self.activations[i+1] * (1 - self.activations[i+1]) * deltas[i+1].dot(self.weights[i+1].T)
            deltas[i] = deltas[i] * self._relu_derivative(self.z_values[i])
        
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * self.activations[i].T.dot(deltas[i]) / m
            self.biases[i] -= self.lr * np.sum(deltas[i], axis=0, keepdims=True) / m
    
    def fit(self, X, y):
        self._init_weights()
        
        n_samples = X.shape[0]
        y_onehot = self._one_hot(y)
        
        for epoch in range(self.n_iter):
            output = self.forward(X)
            loss = self._cross_entropy(output, y_onehot)
            self.losses.append(loss)
            self.backward(y_onehot)
            
            if (epoch + 1) % 100 == 0:
                predictions = self.predict(X)
                accuracy = np.mean(predictions == y)
                print(f"Epoch {epoch+1}/{self.n_iter}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    def _one_hot(self, y):
        n_classes = np.max(y) + 1
        onehot = np.zeros((len(y), n_classes))
        onehot[np.arange(len(y)), y] = 1
        return onehot
    
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X):
        output = self.forward(X)
        return output


class SimpleANN:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01, n_iterations=1000):
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.losses = []
        self.accuracies = []
        
    def _init_weights(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * np.sqrt(2.0 / self.layer_sizes[i])
            b = np.zeros((1, self.layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _cross_entropy(self, y_pred, y_true):
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(y_pred)) / m
    
    def forward(self, X):
        self.activations = [X]
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            if i == len(self.weights) - 1:
                a = self._softmax(z)
            else:
                a = self._relu(z)
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, y_true):
        m = y_true.shape[0]
        n_layers = len(self.weights)
        
        delta = self.activations[-1] - y_true
        
        for i in range(n_layers - 1, -1, -1):
            dw = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._relu(self.activations[i])
                delta = delta * (self.activations[i] > 0).astype(float)
            
            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db
    
    def fit(self, X, y):
        self._init_weights()
        
        n_classes = len(np.unique(y))
        y_onehot = np.zeros((len(y), n_classes))
        y_onehot[np.arange(len(y)), y] = 1
        
        for epoch in range(self.n_iter):
            output = self.forward(X)
            loss = self._cross_entropy(output, y_onehot)
            self.losses.append(loss)
            
            predictions = np.argmax(output, axis=1)
            accuracy = np.mean(predictions == y)
            self.accuracies.append(accuracy)
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{self.n_iter}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
            self.backward(y_onehot)
    
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X):
        return self.forward(X)


def ann_classify(X_train, y_train, X_test, hidden_sizes=[64, 32], lr=0.01, n_iter=1000):
    input_size = X_train.shape[1]
    output_size = len(np.unique(y_train))
    
    model = SimpleANN(input_size, hidden_sizes, output_size, lr, n_iter)
    model.fit(X_train, y_train)
    
    return model.predict(X_test), model