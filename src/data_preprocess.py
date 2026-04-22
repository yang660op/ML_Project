import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from PIL import Image

# ==================== 泰坦尼克和房价数据加载 ====================
def load_titanic_data(train_path, test_path):
    """加载并预处理泰坦尼克数据集，返回 (X_train, y_train, X_test, y_test)"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # 统一列名为小写
    train.columns = train.columns.str.lower()
    test.columns = test.columns.str.lower()
    
    useful_cols = ['age', 'fare', 'sex', 'sibsp', 'parch', 'pclass', 'embarked']
    
    for df in [train, test]:
        df['age'] = df['age'].fillna(df['age'].median())
        df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
        df['fare'] = df['fare'].fillna(df['fare'].median())
    
    X_train = train[useful_cols].values
    target_col = '2urvived' if '2urvived' in train.columns else 'survived'
    y_train = train[target_col].values
    X_test = test[useful_cols].values
    y_test = test[target_col].values
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test

def load_house_data(path):
    """加载房价数据，返回 (X, y)"""
    data = pd.read_csv(path)
    X = data[['x1','x2','x3','x4']].values
    y = data['y'].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

# ==================== 通用图片文件夹加载函数 ====================
def load_image_folder(data_dir, image_size=(28,28), grayscale=True):
    """
    从文件夹结构加载图片数据集。
    假设结构：
        data_dir/
            train/
                0/
                    img1.png
                    ...
                1/
                ...
            test/
                0/
                ...
    返回 (X_train, y_train, X_test, y_test)
    """
    def load_images_from_folder(folder):
        images = []
        labels = []
        for class_name in os.listdir(folder):
            class_path = os.path.join(folder, class_name)
            if not os.path.isdir(class_path):
                continue
            label = int(class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = Image.open(img_path)
                    if grayscale:
                        img = img.convert('L')  # 转灰度
                    img = img.resize(image_size)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"跳过无法读取的图片 {img_path}: {e}")
        return np.array(images), np.array(labels)
    
    train_folder = os.path.join(data_dir, 'train')
    test_folder = os.path.join(data_dir, 'test')
    if not (os.path.exists(train_folder) and os.path.exists(test_folder)):
        raise FileNotFoundError(f"未找到 train/test 文件夹，请确保 {data_dir} 下包含 train 和 test 子文件夹，每个子文件夹内有以类别命名的子文件夹。")
    
    X_train, y_train = load_images_from_folder(train_folder)
    X_test, y_test = load_images_from_folder(test_folder)
    
    # 展平图像
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    return X_train, y_train, X_test, y_test

# ==================== MNIST 加载（自动识别格式） ====================
def load_mnist(data_dir='data/mnist'):
    """
    自动检测数据格式并加载 MNIST：
    1. 如果存在 IDX 格式文件，则使用 IDX 加载。
    2. 否则，尝试从 train/test 文件夹加载图片。
    """
    # 检查 IDX 文件是否存在
    idx_files = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
                 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']
    idx_paths = [os.path.join(data_dir, f) for f in idx_files]
    if all(os.path.exists(p) for p in idx_paths):
        return _load_mnist_idx(data_dir)
    
    # 否则尝试从图片文件夹加载
    print("未找到 IDX 文件，尝试从图片文件夹加载...")
    return load_image_folder(data_dir, image_size=(28,28), grayscale=True)

def _load_mnist_idx(data_dir):
    """内部函数：从 IDX 文件加载 MNIST"""
    def read_idx(filename):
        with open(filename, 'rb') as f:
            magic = np.frombuffer(f.read(4), dtype='>i4')[0]
            data_type = (magic >> 8) & 0xFF
            dims = magic & 0xFF
            shape = tuple(np.frombuffer(f.read(4 * dims), dtype='>i4'))
            data = np.frombuffer(f.read(), dtype='>u1' if data_type == 0x08 else '>i4')
            return data.reshape(shape)
    
    train_images = os.path.join(data_dir, 'train-images-idx3-ubyte')
    train_labels = os.path.join(data_dir, 'train-labels-idx1-ubyte')
    test_images = os.path.join(data_dir, 't10k-images-idx3-ubyte')
    test_labels = os.path.join(data_dir, 't10k-labels-idx1-ubyte')
    
    X_train = read_idx(train_images).astype(np.float32) / 255.0
    y_train = read_idx(train_labels)
    X_test = read_idx(test_images).astype(np.float32) / 255.0
    y_test = read_idx(test_labels)
    
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    return X_train, y_train, X_test, y_test

# ==================== CIFAR-10 加载（自动识别格式） ====================
def load_cifar10(data_dir='data/cifar10'):
    """
    自动检测数据格式并加载 CIFAR-10：
    1. 如果存在 pickle 文件（data_batch_1 等），则使用 pickle 加载。
    2. 否则，尝试从 train/test 文件夹加载图片（32x32 彩色）。
    """
    # 检查 pickle 文件是否存在
    pickle_files = [f'data_batch_{i}' for i in range(1,6)] + ['test_batch']
    pickle_paths = [os.path.join(data_dir, f) for f in pickle_files]
    if all(os.path.exists(p) for p in pickle_paths):
        return _load_cifar10_pickle(data_dir)
    
    # 否则尝试从图片文件夹加载
    print("未找到 pickle 文件，尝试从图片文件夹加载 CIFAR-10...")
    return load_image_folder(data_dir, image_size=(32,32), grayscale=False)

def _load_cifar10_pickle(data_dir):
    """内部函数：从 pickle 文件加载 CIFAR-10"""
    def unpickle(file):
        with open(file, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
        return d
    
    train_batches = [f'data_batch_{i}' for i in range(1, 6)]
    test_batch = 'test_batch'
    
    X_list, y_list = [], []
    for batch in train_batches:
        batch_dict = unpickle(os.path.join(data_dir, batch))
        X_list.append(batch_dict[b'data'])
        y_list.append(batch_dict[b'labels'])
    X_train = np.concatenate(X_list, axis=0).astype(np.float32) / 255.0
    y_train = np.concatenate(y_list, axis=0)
    
    test_dict = unpickle(os.path.join(data_dir, test_batch))
    X_test = test_dict[b'data'].astype(np.float32) / 255.0
    y_test = np.array(test_dict[b'labels'])
    
    return X_train, y_train, X_test, y_test

# ==================== 特征提取与工具函数 ====================
def extract_hog_features(images, resize=(28,28)):
    """从图像数组提取 HOG 特征"""
    if images.ndim == 2:
        side = int(np.sqrt(images.shape[1]))
        images = images.reshape(-1, side, side)
    features = []
    for img in images:
        img_uint8 = (img * 255).astype(np.uint8)
        feat = hog(img_uint8, pixels_per_cell=(4,4), cells_per_block=(2,2), visualize=False)
        features.append(feat)
    return np.array(features)

def filter_two_classes(X, y, class_a, class_b):
    """筛选两类并转为二分类标签"""
    mask = (y == class_a) | (y == class_b)
    X_filtered = X[mask]
    y_filtered = y[mask]
    y_binary = np.where(y_filtered == class_a, 0, 1)
    return X_filtered, y_binary