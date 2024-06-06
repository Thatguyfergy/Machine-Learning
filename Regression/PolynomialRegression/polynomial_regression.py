# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # Independent Variables
y = dataset.iloc[:, 2].values # Dependent Variable

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 3) # Transform original matrix of features X into new matrix of features X_poly
X_poly = polynomial_regression.fit_transform(X) # Contains original independent variable position levels and associated polynomial terms

linear_regression_2 = LinearRegression() # Created a new linear regression object that is fitted to new matrix X_poly
linear_regression_2.fit(X_poly, y) # Fitted also with original dependent variable vector y

# Visualizing the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, linear_regression.predict(X), color = 'blue')
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, linear_regression_2.predict(polynomial_regression.fit_transform(X)), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
