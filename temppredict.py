# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Rituraj Saha

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Chennai Dataset
dataset=pd.read_csv("Chennai.csv")
newdataset=dataset
x=newdataset.iloc[:,[0,1,2]].values
y=newdataset.iloc[:,3].values
newdataset.dropna(inplace=True) 

# Importing the Delhi Dataset
dataset=pd.read_csv("Delhi.csv")
newdataset=dataset
x=newdataset.iloc[:,[0,1,2]].values
y=newdataset.iloc[:,3].values
newdataset.dropna(inplace=True) 

# Importing the Mumbai Dataset
dataset=pd.read_csv("Mumbai.csv")
newdataset=dataset
x=newdataset.iloc[:,[0,1,2]].values
y=newdataset.iloc[:,3].values
newdataset.dropna(inplace=True) 

# Importing the Kolkata Dataset
dataset=pd.read_csv("Kolkata.csv")
newdataset=dataset
x=newdataset.iloc[:,[0,1,2]].values
y=newdataset.iloc[:,3].values
newdataset.dropna(inplace=True) 


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)


# Fitting Model to the Training set(Random Forest Regression)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=2000,random_state=0)
regressor.fit(x_train, y_train)

# Fitting Model to the Training set(SVR)
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x_train, y_train)

# Fitting Model to the Training set(Multiple Linear Regression)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Fitting Model to the Training set(Polynomial Regression)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(x_train)
poly_reg.fit(X_poly, y_train)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train)


# Predicting the Test set results
y_pred = regressor.predict(x_test)

# Analyzing the outut
from sklearn.metrics import r2_score  
print("r2_score: ",r2_score(y_test, y_pred)) 




