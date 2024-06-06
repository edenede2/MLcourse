import numpy as np
from tqdm import tqdm



# Decision Tree Classifier (with Gini impurity)
class DecisionTree:
    # Initialize the Decision Tree Classifier
    def __init__(self, min_samples_split=2, max_depth=5):
        # Minimum number of samples required to split an internal node
        self.min_samples_split = min_samples_split
        # Maximum depth of the tree
        self.max_depth = max_depth
        # The decision tree
        self.tree = None

    # Fit the Decision Tree Classifier
    def fit(self, X, y):
        # Convert string labels to numerical indices
        self.classes = np.unique(y)
        y = np.array([np.where(self.classes == label)[0][0] for label in y])
        # Grow the decision tree
        self.tree = self._grow_tree(X, y)
    
    # Predict the target variable for the input data
    def predict(self, X):
        predictions = [self._predict(inputs, self.tree) for inputs in X]
        # Make predictions for each input based on the decision tree that was grown
        return np.array([self.classes[pred] for pred in predictions])

    # Calculate the Gini impurity
    def _gini(self, y):
        # Number of samples
        m = len(y)
        # Return 1 - sum of the square of the proportion of samples in each class label to the total number of samples in the node
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    # Split the data into left and right subsets based on the feature and threshold
    def _split(self, X, y, idx, thresh):
        # Get the indices of the samples in the left subset
        left_mask = X[:, idx] <= thresh
        # Get the indices of the samples in the right subset
        right_mask = X[:, idx] > thresh
        # Return the left and right subsets of the data
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]


    # Find the best split for the data
    def _best_split(self, X, y):
        # Number of samples (m) and number of features (n)
        m, n = X.shape

        # If the number of samples is less than or equal to the minimum number of samples required to split, return None
        if m <= 1:
            return None, None

        # Initialize the Gini impurity of the best split to infinity and the best feature and threshold to None
        best_gini = 1.0
        best_idx, best_thresh = None, None

        unique_classes = np.unique(y)
        class_count = len(unique_classes)

        # For each feature
        for idx in range(n):
            # Get the thresholds and class labels
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            # Ensure numeric thresholds
            thresholds = np.array(thresholds, dtype=np.float64)

            num_left = [0] * class_count
            num_right = [np.sum(classes == c) for c in unique_classes]

            # For each sample
            for i in range(1, m):
                class_idx = np.where(unique_classes == classes[i - 1])[0][0]
                num_left[class_idx] += 1
                num_right[class_idx] -= 1

                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(class_count))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(class_count))
                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thresh = (thresholds[i] + thresholds[i - 1]) / 2
        
        return best_idx, best_thresh

    # Grow the decision tree
    def _grow_tree(self, X, y, depth=0):
        
        # Get the number of samples for each class label
        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
        # Get the class that occurs
        predicted_class = np.argmax(num_samples_per_class)
        # Create a node for the decision tree
        node = {'predicted_class': predicted_class}

        

        # If the depth of the tree is less than the maximum depth
        if depth < self.max_depth:
            # Get the best split
            idx, thresh = self._best_split(X, y)
            # If the best split is not None
            if idx is not None:
                # Get the left and right subsets of the data
                indices_left = X[:, idx] <= thresh
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                # Grow the left and right subtrees
                node['feature_index'] = idx
                node['threshold'] = thresh
                node['left'] = self._grow_tree(X_left, y_left, depth + 1)
                node['right'] = self._grow_tree(X_right, y_right, depth + 1)
        return node

    # Predict the target variable for the input data
    def _predict(self, inputs, tree):
        # If the tree is not a leaf node
        if 'threshold' in tree:
            # Get the feature index and threshold
            feature_index = tree['feature_index']
            # Traverse the left or right subtree
            if inputs[feature_index] <= tree['threshold']:
                return self._predict(inputs, tree['left'])
            else:
                return self._predict(inputs, tree['right'])
        # If the tree is a leaf node
        else:
            # Return the predicted class
            return tree['predicted_class']




