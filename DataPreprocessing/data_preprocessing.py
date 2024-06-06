# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Library to manage datasets

# Importing the dataset
dataset = pd.read_csv('Data.csv') # Importing Data.csv
X = dataset.iloc[:, :-1].values # Taking all the rows and columns except the last one
y = dataset.iloc[:, 3].values # Taking the rows of the third column

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) # Object imputer with parameters
imputer = imputer.fit(X[:, 1:3]) # Fit the missing data in all rows, and columns 1 and 2
X[:, 1:3] = imputer.transform(X[:, 1:3]) # Replace missing data by the mean of the column

# Enconding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) # Fit label encoder to countries, encoded by replacing text with number

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y) # Encode Yes as 1, No as 0

# Splitting dataset into training and test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # X_train is the training part of matrix of features, y_train is the training part of the dependent variables

# Feature scaling to prevent domination in ie Euclidean algorithm
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # Scale X_train, fit and transform for training set
X_test = sc_X.transform(X_test) # Scale X_test, transform for test set only
