import numpy as np
from Models.ClassifierDT import DecisionTree

# Decision Tree Regressor (with SSR)
class DecisionTreeRegressor(DecisionTree):
    # Initialize the Decision Tree Regressor that inherits from the Decision Tree Classifier
    def _ssr(self, y):
        if len(y) == 0:
            return 0
        mean_y = np.mean(y)
        # Calculate the sum of the squared residuals and return it
        return np.sum((y - mean_y) ** 2)

    # Find the best split for the data based on the sum of the squared residuals
    def _best_split(self, X, y):

        # Number of samples (m) and number of features (n)
        m, n = X.shape

        # If the number of samples is less than or equal to the minimum number of samples required to split, return None
        if m <= 1:
            return None, None

        # Initialize the sum of the squared residuals of the best split to infinity and the best feature and threshold to None
        best_ssr = np.inf
        best_idx, best_thresh = None, None

        # For each feature
        for idx in range(n):
            # Get the thresholds and values
            thresholds, values = zip(*sorted(zip(X[:, idx], y)))
            # For each sample
            for i in range(1, m):
                # Get the left and right subsets of the target variable
                y_left, y_right = values[:i], values[i:]
                ssr_left, ssr_right = self._ssr(y_left), self._ssr(y_right)

                # Calculate the sum of the squared residuals of the left and right subsets
                ssr = ssr_left + ssr_right

                # Update the best split if the current split has a lower sum of the squared residuals
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if ssr < best_ssr:
                    best_ssr = ssr
                    best_idx = idx
                    best_thresh = (thresholds[i] + thresholds[i - 1]) / 2

        # Return the best feature and threshold
        return best_idx, best_thresh