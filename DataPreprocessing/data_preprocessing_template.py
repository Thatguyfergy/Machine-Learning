# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv') # Importing Data.csv
X = dataset.iloc[:, :-1].values # Taking all the rows and columns except the last one
y = dataset.iloc[:, 3].values # Taking the rows of the third column

# Splitting dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # X_train is the training part of matrix of features, y_train is the training part of the dependent variables

"""
# Feature scaling to prevent domination in ie Euclidean algorithm
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # Scale X_train, fit and transform for training set
X_test = sc_X.transform(X_test) # Scale X_test, transform for test set only
"""
