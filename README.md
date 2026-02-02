# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load the dataset into a DataFrame and explore its contents to understand the data structure.

2.Separate the dataset into independent (X) and dependent (Y) variables, and split them into training and testing sets.

3.Create a linear regression model and fit it using the training data.

4.Predict the results for the testing set and plot the training and testing sets with fitted lines.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Jegan S K
RegisterNumber:  212225230117
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv("student_scores.csv")
x=df.iloc[:,0:1]
y=df.iloc[:,-1]

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
lr=LinearRegression()
lr.fit(X_train,Y_train)
lr.predict(X_test.iloc[0].values.reshape(1,1))

plt.scatter(df['Hours'],df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.plot(X_train,lr.predict(X_train),color='green')
lr.coef_
lr.intercept_

y_pred=lr.predict(X_test)
mse=mean_squared_error(Y_test,y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(Y_test,y_pred)
r2=r2_score(Y_test,y_pred)
print("MSE:",mse)
print("RMSE:",rmse)
print("MAE:",mae)
```

## Output:


<img width="1264" height="630" alt="image" src="https://github.com/user-attachments/assets/1dd559e7-07c5-4f25-ae7d-071de56784ee" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
