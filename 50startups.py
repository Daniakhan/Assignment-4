# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
 
dataset = pd.read_csv('50_Startups.csv')

#Analyzing Startups of New York
NY = dataset.loc[dataset.State=='New York', :]
Y1 = NY.iloc[:, -1].values
X1 = np.arange(17).reshape(-1, 1)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size = 0.25, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X1_train, Y1_train)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X1_poly = poly_reg.fit_transform(X1)
poly_reg.fit(X1_poly,Y1)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X1_poly, Y1)


# Visualising the Polynomial Regression results (for high resolution and smoother curve)
X1_grid = np.arange(min(X1), max(X1), 0.1)
X1_grid = X1_grid.reshape((len(X1_grid), 1))
plt.scatter(X1, Y1, color = 'red')
plt.plot(X1_grid, lin_reg_2.predict(poly_reg.fit_transform(X1_grid)), color = 'blue')
plt.title('New York Startups (Polynomial Regression)')
plt.xlabel('Startup')
plt.ylabel('Profit')
plt.show()

# predicting and Visualising the linear results
Y_pred = lin_reg_2.predict(poly_reg.fit_transform([[20]]))
print('Prediction of New York Startup:')
print(Y_pred)

# Analyzing Startups of California
C =dataset.loc[dataset.State=='California',:]
Y2 = C.iloc[:, -1].values
X2 = np.arange(17).reshape(-1, 1)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size = 0.25, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X2_train, Y2_train)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X2_poly = poly_reg.fit_transform(X2)
poly_reg.fit(X2_poly,Y2)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X2_poly, Y2)


# Visualising the Polynomial Regression results (for high resolution and smoother curve)
X2_grid = np.arange(min(X2), max(X2), 0.1)
X2_grid = X2_grid.reshape((len(X2_grid), 1))
plt.scatter(X2, Y2, color = 'red')
plt.plot(X1_grid, lin_reg_2.predict(poly_reg.fit_transform(X2_grid)), color = 'blue')
plt.title('California Startups (Polynomial Regression)')
plt.xlabel('Startup')
plt.ylabel('Profit')
plt.show()

# predicting and Visualising the linear results
Y_pred = lin_reg_2.predict(poly_reg.fit_transform([[20]]))
print('Prediction of California Startup:')
print(Y_pred)

# Analyzing Startups of Florida
F =dataset.loc[dataset.State=='Florida',:]
Y3 = F.iloc[:, -1].values
X3 = np.arange(16).reshape(-1, 1)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X3_train, X3_test, Y3_train, Y3_test = train_test_split(X3, Y3, test_size = 0.25, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X3_train, Y3_train)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X3_poly = poly_reg.fit_transform(X3)
poly_reg.fit(X3_poly,Y3)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X3_poly, Y3)

# Visualising the Polynomial Regression results (for high resolution and smoother curve)
X3_grid = np.arange(min(X3), max(X3), 0.1)
X3_grid = X3_grid.reshape((len(X3_grid), 1))
plt.scatter(X3, Y3, color = 'red')
plt.plot(X3_grid, lin_reg_2.predict(poly_reg.fit_transform(X3_grid)), color = 'blue')
plt.title('Florida Startups (Polynomial Regression)')
plt.xlabel('Startup')
plt.ylabel('Profit')
plt.show()

# predicting and Visualising the linear results
Y_pred = lin_reg_2.predict(poly_reg.fit_transform([[20]]))
print('Prediction of Florida Startup:') 
print(Y_pred)