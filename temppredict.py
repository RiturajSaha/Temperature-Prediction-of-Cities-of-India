# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: R2J

"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Importing the Dataset
dataset=pd.read_csv("Temperature dataset\Chennai.csv")
dataset=pd.read_csv("Temperature dataset\Delhi.csv")
dataset=pd.read_csv("Temperature dataset\Mumbai.csv")
dataset=pd.read_csv("Temperature dataset\Kolkata.csv")


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


# Predicting the Test set results
y_pred = regressor.predict(x_test)


# Analyzing the outut
from sklearn.metrics import r2_score  
print("r2_score: ",r2_score(y_test, y_pred)) 
