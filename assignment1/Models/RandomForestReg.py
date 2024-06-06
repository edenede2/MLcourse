import numpy as np
from Models.ClassifierDT import DecisionTree
from Models.RandomForest import RandomForest

# Random Forest Regressor (with averaging)
class RandomForestRegressor(RandomForest):
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)