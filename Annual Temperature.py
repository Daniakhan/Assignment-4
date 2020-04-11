# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer

#Importing the dataset
dataset = pd.read_csv('annual_temp.csv')
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

onehotencoder = ColumnTransformer([("Source", OneHotEncoder(), [0])], remainder = 'passthrough')
X = onehotencoder.fit_transform(X)
X = X[:,0:]

                    
X1 = dataset.loc[dataset.Source == 'GCAG', ['Year']]
Y1=dataset.loc[dataset.Source == 'GCAG', ['Mean']]


print(X1)
print(Y1)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X1,Y1)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X1_poly= poly_reg.fit_transform(X1)
poly_reg.fit(X1_poly, Y1)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X1_poly, Y1)

# Visualising the Polynomial Regression results
plt.scatter(X1, Y1, color = 'red')
plt.plot(X1, lin_reg_2.predict(poly_reg.fit_transform(X1)), color = 'blue')
plt.title('Annual Temperature Of GCAG (Polynomial Regression)')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.show()


X2 = dataset.loc[dataset.Source == 'GISTEMP', ['Year']]
Y2=dataset.loc[dataset.Source == 'GISTEMP', ['Mean']]


print(X2)
print(Y2)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X2,Y2)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X2_poly= poly_reg.fit_transform(X1)
poly_reg.fit(X2_poly, Y2)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X2_poly, Y2)

# Visualising the Polynomial Regression results
plt.scatter(X2, Y2, color = 'red')
plt.plot(X2, lin_reg_2.predict(poly_reg.fit_transform(X2)), color = 'blue')
plt.title('Annual Temperature Of GISTEMP (Polynomial Regression)')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.show()

