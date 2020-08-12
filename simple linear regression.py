
# importing the libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import os
print(os.getcwd())
os.chdir("C:\\Users\\HP\\Downloads")#input csv file location

# input dataset
dataset= pd.read_csv("Salary_Data.csv")
print(dataset)
X= dataset.iloc[:,:-1].values
y= dataset.iloc[:,1].values


# Spliting the data into training set and test set
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=1/3, random_state=0 )

#fitting simple linear regression to training set

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

# Visualizing the training set results
 
plt.scatter(X_train,y_train, color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience ( Training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the test set results
plt.scatter(X_test,y_test,color='magenta')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title('Salary vs experience (Test set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
