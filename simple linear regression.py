# -*- coding: utf-8 -*-
"""
Created on Fri May 15 18:53:14 2020

@author: HP
"""
# importing the libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import os
print(os.getcwd())
os.chdir("C:\\Users\\HP\\Downloads")

# input dataset
dataset= pd.read_csv("VC_Startups.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values
# Encoding numerical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x=LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x), dtype=np.float)


# Avoiding the dummy variable trap 
x= x[:,1:]
# Splitting data into training set and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
# Fitting the multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train,y_train)
#Predicting the test set results
y_pred=regressor.predict(x_test)
# Building the optimal model using Backward elimination
import statsmodels.api as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
x_opt=x[:, [0, 1, 2, 3,4]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=x[:, [0, 1, 2, 4]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=x[:, [0, 2,4]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=x[:, [0, 2]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()




