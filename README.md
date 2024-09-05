# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries in python required for finding Gradient Design.
2. Read the dataset file and check any null value using .isnull() method.
3. Declare the default variables with respective values for linear regression.
4. Calculate the loss using Mean Square Error.
5. Predict the value of y.
6. Plot the graph respect to hours and scores using .scatterplot() method for Linear Regression.
7. Plot the graph respect to loss and iterations using .plot() method for Gradient Descent.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Aaron I
RegisterNumber:  212223230002
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate = 0.01, num_iters = 1000):
    
    X = np.c_[np.ones(len(X1)), X1]
    
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors = (predictions - y).reshape(-1,1)
        theta = learning_rate * (1/len(X1) * X.T.dot(errors))
    
    return theta

data = pd.read_csv('/content/50_Startups.csv')
data.head()

X = (data.iloc[1: , :-2].values)
# print(X)

X1 = X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1: , -1].values).reshape(-1, 1)
# print(y)

X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)

print("X Scaled : \n",X1_Scaled)
print("Y Scaled : \n",Y1_Scaled)

theta = linear_regression(X1_Scaled, Y1_Scaled)

new_data = np.array([165349.2, 136897.8, 471784.1]).reshape(-1,1)
new_scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1, new_scaled), theta)
prediction = prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")

# print(linear_regression(X1, y))
```
## Output:

![Screenshot 2024-09-05 112124](https://github.com/user-attachments/assets/b0110d29-ed2b-459d-8ec4-6f0b426b0587)


![Screenshot 2024-09-05 112132](https://github.com/user-attachments/assets/6914878e-f707-4487-93c3-29d5b7f5f96e)


![Screenshot 2024-09-05 112021](https://github.com/user-attachments/assets/20175ffc-7d67-4a9d-9c17-eb966f452b43)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
