import numpy as np

X_path_train = './data/train/X_train.txt'
Y_path_train = './data/train/y_train.txt'
X_path_test =  './data/test/X_test.txt'
Y_path_test =  './data/test/y_test.txt'

def read_data(path):
    """
    Reads data from the given path.
    
    Args:
        path: str
        
    Returns:
        data: np.ndarray
    """
    with open(path, 'r') as f:
        data = np.array([np.array(row.replace('  ', ' ').strip().split(' '), dtype=np.float32) for row in f])
        
    return data

def read_labels(path):
    """
    Reads labels from the given path.
    
    Args:
        path: str
        
    Returns:
        np.ndarray
    """
    return np.genfromtxt(path).reshape((-1, 1))

def normalize_data(data):
    """
    Normalizes the input array to zero mean and unit variance across the 0th dimension.
    
    Args:
        data: np.ndarray
    
    Returns:
        normalized: np.ndarray, the argument normalized to zero mean and unit variance
        mean, std: floats, mean and std of the original data
    """
    mean = data.mean(0)
    std = data.std(0)
    
    normalized = (data - mean) / std
    
    return normalized, mean, std

def shuffle_data(X, Y):
    """
    Randomly shuffle the data and labels across the zeroth axis (that is, across samples).
    
    Args:
        X: np.ndarray, of shape (m, n) where m is the number of samples and n - features per sample
        Y: np.ndarray, of shape (m, 1)
    
    Returns:
        X, Y: shuffled data and labels
    """
    
    # Shuffle the data
    idx = np.random.permutation(np.arange(X.shape[0]))
    return X[idx, :], Y[idx, :]
    

def preprocess_labels(labels):
    """
    Fixes the labels to start at 0 and have dtype = np.int
    """
    return (labels - 1).astype(np.int)

def get_data(X_path, Y_path):
    """
    Extracts and normalizes the data and labels.
    """
    X_train, Y_train = read_data(X_path), read_labels(Y_path)
    X_norm, mean, std = normalize_data(X_train)
    Y_norm = preprocess_labels(Y_train)
    
    return X_norm, Y_norm, mean, std

def train_dev_split(X, Y, ratio=0.1):
    """
    Shuffles the data and splits it into a train set and a dev/validation set.
    """
    assert ratio < 1 and ratio >= 0
    X, Y = shuffle_data(X, Y)
    
    m_dev = int(ratio * X.shape[0])
    
    X_train, Y_train, X_dev, Y_dev = X[m_dev:,:], Y[m_dev:,:], X[:m_dev,:], Y[:m_dev,:]
    
    return X_train, Y_train, X_dev, Y_dev

def to_onehot(Y):
    """
    Converts the labes to onehot format
    """
    nb_classes = 6
    targets = Y.reshape(-1)
    one_hot_targets = np.eye(nb_classes)[targets]
    
    return one_hot_targets

def get_and_fix_data(X_path, Y_path, ratio=0.1):
    """
    Convenience function for extracting data and doing all the necessary preprocessing.
    """
    X_all, Y_all, _, _ = get_data(X_path, Y_path)
    Y_all = to_onehot(Y_all)

    X_train, Y_train, X_dev, Y_dev = train_dev_split(X_all, Y_all, ratio=ratio)
    
    return X_train, Y_train, X_dev, Y_dev
