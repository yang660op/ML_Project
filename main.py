import argparse
import os
import sys
import pickle
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error

# 确保可以导入 src 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocess import (
    load_titanic_data, load_house_data, load_mnist, load_cifar10,
    extract_hog_features, filter_two_classes
)
from src.KNN import knn_classify
from src.logistic_regression import logistic_regression_classify, LogisticRegression
from src.linear_regression import linear_regression_predict, LinearRegression
from src.SVM import svm_classify, SVM
from src.ANN import SimpleANN, ann_classify

# 创建保存目录
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def save_model(model_params, algo, data, suffix='pk'):
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"models/{algo}_{data}_{timestamp}.{suffix}"
    with open(filename, 'wb') as f:
        pickle.dump(model_params, f)
    print(f"Model saved to {filename}")
    return filename

def save_results(content, filename):
    with open(f"results/{filename}", 'w') as f:
        f.write(content)
    print(f"Result saved to results/{filename}")

def plot_loss(losses, title, filename):
    plt.figure()
    plt.plot(losses)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(f"results/{filename}")
    plt.close()
    print(f"Plot saved to results/{filename}")

def main():
    parser = argparse.ArgumentParser(description='Machine Learning Project')
    parser.add_argument('--algo', type=str, required=True,
                        choices=['knn', 'logistic', 'linear', 'svm', 'ann'],
                        help='Algorithm to use')
    parser.add_argument('--data', type=str, required=True,
                        choices=['titanic', 'house', 'mnist', 'cifar10'],
                        help='Dataset')
    parser.add_argument('--process', type=str, required=True,
                        choices=['train', 'test'],
                        help='Process type')
    parser.add_argument('--k', type=int, default=3, help='K for KNN')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--method', type=str, default='normal', choices=['normal','gd'],
                        help='Linear regression method')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for GD')
    parser.add_argument('--class_a', type=int, default=None, help='First class for binary classification')
    parser.add_argument('--class_b', type=int, default=None, help='Second class for binary classification')
    parser.add_argument('--hog', action='store_true', help='Use HOG features for image data')
    parser.add_argument('--hidden', type=str, default='64,32', help='Hidden layer sizes (comma-separated)')
    
    args = parser.parse_args()
    
    # 加载数据
    if args.data == 'titanic':
        X_train, y_train, X_test, y_test = load_titanic_data('data/titanic/train.csv', 'data/titanic/test.csv')
    elif args.data == 'house':
        X, y = load_house_data('data/house/house_data.csv')
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
    elif args.data == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist('data/mnist')
        if args.hog:
            X_train = extract_hog_features(X_train.reshape(-1,28,28))
            X_test = extract_hog_features(X_test.reshape(-1,28,28))
    elif args.data == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10('data/cifar10'
        )
        if args.hog:
            X_train_gray = X_train.reshape(-1,32,32,3).mean(axis=3)
            X_test_gray = X_test.reshape(-1,32,32,3).mean(axis=3)
            X_train = extract_hog_features(X_train_gray)
            X_test = extract_hog_features(X_test_gray)
    
    # 二分类算法处理多分类数据集
    if args.algo in ['logistic', 'svm', 'ann'] and args.data in ['mnist', 'cifar10']:
        if args.class_a is None or args.class_b is None:
            raise ValueError(f"For {args.algo} on {args.data}, you must specify --class_a and --class_b")
        X_train, y_train = filter_two_classes(X_train, y_train, args.class_a, args.class_b)
        X_test, y_test = filter_two_classes(X_test, y_test, args.class_a, args.class_b)
    
    # 训练或测试
    if args.process == 'train':
        if args.algo == 'knn':
            model_params = {'X_train': X_train, 'y_train': y_train, 'k': args.k}
            save_model(model_params, args.algo, args.data)
            y_pred = knn_classify(X_train, y_train, X_train, k=args.k)
            acc = accuracy_score(y_train, y_pred)
            print(f"Train accuracy: {acc:.4f}")
        elif args.algo == 'logistic':
            model = LogisticRegression(learning_rate=args.lr, n_iterations=args.epochs)
            model.fit(X_train, y_train)
            model_params = {'weights': model.weights, 'fit_intercept': model.fit_intercept}
            save_model(model_params, args.algo, args.data)
            y_pred = model.predict(X_train)
            acc = accuracy_score(y_train, y_pred)
            print(f"Train accuracy: {acc:.4f}")
        elif args.algo == 'linear':
            model = LinearRegression(method=args.method, batch_size=args.batch_size,
                                     learning_rate=args.lr, n_iterations=args.epochs)
            model.fit(X_train, y_train)
            model_params = {'weights': model.weights, 'method': args.method}
            save_model(model_params, args.algo, args.data)
            y_pred = model.predict(X_train)
            mse = mean_squared_error(y_train, y_pred)
            print(f"Train MSE: {mse:.4f}")
        elif args.algo == 'svm':
            model = SVM(learning_rate=args.lr, n_iterations=args.epochs)
            model.fit(X_train, y_train)
            model_params = {'w': model.w, 'b': model.b}
            save_model(model_params, args.algo, args.data)
            y_pred = model.predict(X_train)
            acc = accuracy_score(y_train, y_pred)
            print(f"Train accuracy: {acc:.4f}")
        elif args.algo == 'ann':
            hidden_sizes = [int(x) for x in args.hidden.split(',')]
            output_size = len(np.unique(y_train))
            model = SimpleANN(X_train.shape[1], hidden_sizes, output_size, 
                            learning_rate=args.lr, n_iterations=args.epochs)
            model.fit(X_train, y_train)
            model_params = {
                'weights': model.weights, 
                'biases': model.biases,
                'layer_sizes': [X_train.shape[1]] + hidden_sizes + [output_size]
            }
            save_model(model_params, args.algo, args.data)
            y_pred = model.predict(X_train)
            acc = accuracy_score(y_train, y_pred)
            print(f"Train accuracy: {acc:.4f}")
            plot_loss(model.losses, 'ANN Training Loss', f'ann_loss_{datetime.now().strftime("%Y%m%d")}.png')
            plot_loss(model.accuracies, 'ANN Training Accuracy', f'ann_accuracy_{datetime.now().strftime("%Y%m%d")}.png')
    elif args.process == 'test':
        import glob
        pattern = f"models/{args.algo}_{args.data}_*.pk"
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No saved model found for {args.algo}_{args.data}")
        model_path = max(files)
        with open(model_path, 'rb') as f:
            model_params = pickle.load(f)
        
        if args.algo == 'knn':
            X_train_saved = model_params['X_train']
            y_train_saved = model_params['y_train']
            k = model_params.get('k', 3)
            y_pred = knn_classify(X_train_saved, y_train_saved, X_test, k=k)
        elif args.algo == 'logistic':
            model = LogisticRegression()
            model.weights = model_params['weights']
            model.fit_intercept = model_params.get('fit_intercept', True)
            y_pred = model.predict(X_test)
        elif args.algo == 'linear':
            model = LinearRegression()
            model.weights = model_params['weights']
            y_pred = model.predict(X_test)
        elif args.algo == 'svm':
            model = SVM()
            model.w = model_params['w']
            model.b = model_params['b']
            y_pred = model.predict(X_test)
        elif args.algo == 'ann':
            model = SimpleANN(
                model_params['layer_sizes'][0],
                model_params['layer_sizes'][1:-1],
                model_params['layer_sizes'][-1]
            )
            model.weights = model_params['weights']
            model.biases = model_params['biases']
            y_pred = model.predict(X_test)
        
        if args.algo == 'linear':
            mse = mean_squared_error(y_test, y_pred)
            result_str = f"Test MSE: {mse:.4f}\n"
            print(result_str)
            save_results(result_str, f"{args.algo}_{args.data}_{datetime.now().strftime('%Y%m%d')}_mse.txt")
        else:
            acc = accuracy_score(y_test, y_pred)
            result_str = f"Test accuracy: {acc:.4f}\n"
            print(result_str)
            save_results(result_str, f"{args.algo}_{args.data}_{datetime.now().strftime('%Y%m%d')}_accuracy.txt")

if __name__ == '__main__':
    main()