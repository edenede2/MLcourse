from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
import multiprocessing
import pandas as pd


# Define the update_progress function at the module level
def update_progress(progress_queue, total):
    pbar = tqdm(total=total, desc="Fitting Trees", position=0, leave=True)
    while True:
        item = progress_queue.get()
        if item is None:
            break
        pbar.update(1)
    pbar.close()

def fit_single_tree(tree_index, X, y, max_depth, min_samples_split, progress_queue):
    idxs = np.random.choice(len(X), len(X), replace=True)
    tree = DecisionTreeRegressorOptimized(max_depth=max_depth, min_samples_split=min_samples_split)
    tree.fit(X[idxs], y[idxs])
    progress_queue.put(1)  # Put a progress update in the queue
    return tree

class DecisionTreeRegressorOptimized:
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)
    
    def predict(self, X):
        return np.array([self._predict(inputs, self.tree) for inputs in X])

    def _ssr(self, y):
        if len(y) == 0:
            return 0
        mean_y = np.mean(y)
        return np.sum((y - mean_y) ** 2)

    def _split(self, X, y, idx, thresh):
        left_mask = X[:, idx] <= thresh
        right_mask = X[:, idx] > thresh
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= self.min_samples_split:
            return None, None

        best_ssr = np.inf
        best_idx, best_thresh = None, None

        for idx in range(n):
            thresholds, values = zip(*sorted(zip(X[:, idx], y)))
            for i in range(1, m):
                y_left, y_right = values[:i], values[i:]
                ssr_left, ssr_right = self._ssr(y_left), self._ssr(y_right)
                ssr = ssr_left + ssr_right

                if thresholds[i] == thresholds[i - 1]:
                    continue
                if ssr < best_ssr:
                    best_ssr = ssr
                    best_idx = idx
                    best_thresh = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thresh

    def _grow_tree(self, X, y, depth=0):
        if len(y) == 0:
            return None
        predicted_value = np.mean(y)
        node = {'predicted_value': predicted_value}

        if depth < self.max_depth:
            idx, thresh = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] <= thresh
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node['feature_index'] = idx
                node['threshold'] = thresh
                node['left'] = self._grow_tree(X_left, y_left, depth + 1)
                node['right'] = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs, tree):
        if 'threshold' in tree:
            feature_index = tree['feature_index']
            if inputs[feature_index] <= tree['threshold']:
                return self._predict(inputs, tree['left'])
            else:
                return self._predict(inputs, tree['right'])
        else:
            return tree['predicted_value']
        


class RandomForestRegressorParallel:
    def __init__(self, n_trees=100, max_depth=5, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape
        self.n_features = self.n_features or n_features

        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()

        # Start the progress bar updater process
        progress_process = multiprocessing.Process(target=update_progress, args=(progress_queue, self.n_trees))
        progress_process.start()

        # Fit trees in parallel
        self.trees = Parallel(n_jobs=-1)(delayed(fit_single_tree)(i, X, y, self.max_depth, self.min_samples_split, progress_queue) for i in range(self.n_trees))
        
        # Ensure all progress updates are completed
        progress_queue.put(None)
        progress_process.join()

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)

if __name__ == "__main__":
    imputed_data = pd.read_csv('/Users/edeneldar/Downloads/imputed_data.csv')

    # Correct the values of the 'Condition' column to 'New' and 'Refurbished'
    imputed_data['Condition'] = imputed_data['Condition'].apply(lambda x: 'New' if x == 'New' or x == 'Open box' else 'Refurbished')

    # Convert categorical features to numerical values using one-hot encoding
    imputed_data = pd.get_dummies(imputed_data, columns=['Brand', 'Processor', 'GPU', 'GPU-Type', 'Resolution'])

    reg_data = pd.get_dummies(imputed_data, columns=['Condition'])

    ######### Split the data

    # Split the data
    train_data = imputed_data.iloc[0:2058]
    val_data = imputed_data.iloc[2058:2499]
    test_data = imputed_data.iloc[2499:2939]

    # Split the data for regression
    train_data_reg = reg_data.iloc[0:2058]
    val_data_reg = reg_data.iloc[2058:2499]
    test_data_reg = reg_data.iloc[2499:2939]


    # Extract features and target variables
    X_train_clas = train_data.drop(columns=['Condition'])
    X_train_reg = train_data_reg.drop(columns=['Price'])
    y_train_clas = train_data['Condition']
    y_train_reg = train_data_reg['Price']

    X_val_clas = val_data.drop(columns=['Condition'])
    X_val_reg = val_data_reg.drop(columns=['Price'])
    y_val_clas = val_data['Condition']
    y_val_reg = val_data_reg['Price']

    X_test_clas = test_data.drop(columns=['Condition'])
    X_test_reg = test_data_reg.drop(columns=['Price'])
    y_test_clas = test_data['Condition']
    y_test_reg = test_data_reg['Price']

    # Convert string labels to numerical indices
    class_map = {label: idx for idx, label in enumerate(np.unique(y_train_clas))}
    y_train_clas_numeric = np.array([class_map[label] for label in y_train_clas])
    y_val_clas_numeric = np.array([class_map[label] for label in y_val_clas])
    y_test_clas_numeric = np.array([class_map[label] for label in y_test_clas])

    import warnings

    warnings.filterwarnings('ignore')

    rf_regressor = RandomForestRegressorParallel(n_trees=100, max_depth=5)
    rf_regressor.fit(X_train_reg.values, y_train_reg.values)
    predictions = rf_regressor.predict(X_val_reg.values)
    mse = np.mean((predictions - y_val_reg.values) ** 2)
    print(f'Validation MSE: {mse}')