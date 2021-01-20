import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_normalized_feature_label(dataset, features_col, labels_col):
    """ Normalizing data points."""
    # Considering columns from index 1 to 7 ('Relative_Date','Volume','Open','High','Low','Close','Adj Close')
    # Relative date has advantage over normal date which is not contiguous due to holidays.
    data = dataset.iloc[:,1:8].values
    X = data[:, features_col]
    y = data[:, labels_col]
    y = np.reshape(y, (y.shape[0], 1))
    X, y = min_max_scaling(X, y)
    return X, y

def min_max_scaling(X, y):
    """ MinMaxScaling of features to range of values between (0.0, 1.0)."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(X), scaler.fit_transform(y)

def train_test_split(X, y, test_size):
    """ Splitting feature X and labels y each into two contiguous array according to test_size specify.""" 
    total_len = len(X)
    split_index_from_end = -int(test_size * total_len)
    return X[:split_index_from_end], X[split_index_from_end:], y[:split_index_from_end],y[split_index_from_end:]

