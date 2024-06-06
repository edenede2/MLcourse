import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Models.ClassifierDT import DecisionTree
from Models.RegressionDT import DecisionTreeRegressor

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('/Users/edeneldar/Library/Mobile Documents/com~apple~CloudDocs/ML learn.worktrees/origin/mainEden/assignment1/assignment-1-data.csv')


data = data[['Brand', 'Screen_Size', 'RAM', 'Processor', 'GPU', 'GPU_Type', 'Resolution', 'Condition', 'Price']]

# Convert categorical features to numerical values using one-hot encoding
data = pd.get_dummies(data, columns=['Brand', 'Processor', 'GPU', 'GPU_Type', 'Resolution'])

# Split the data
train_data = data.iloc[0:2058]
val_data = data.iloc[2058:2499]
test_data = data.iloc[2499:2939]

# Extract features and target variables
X_train_class = train_data.drop(columns=['Condition'])
X_train_reg = train_data.drop(columns=['Price'])
y_train_class = train_data['Condition']
y_train_reg = train_data['Price']

X_val_class = val_data.drop(columns=['Condition'])
X_val_reg = val_data.drop(columns=['Price'])
y_val_class = val_data['Condition']
y_val_reg = val_data['Price']

X_test_class = test_data.drop(columns=['Condition'])
X_test_reg = test_data.drop(columns=['Price'])
y_test_class = test_data['Condition']
y_test_reg = test_data['Price']

# Convert string labels to numerical indices
class_map = {label: idx for idx, label in enumerate(np.unique(y_train_class))}
y_train_class_numeric = np.array([class_map[label] for label in y_train_class])
y_val_class_numeric = np.array([class_map[label] for label in y_val_class])
y_test_class_numeric = np.array([class_map[label] for label in y_test_class])


