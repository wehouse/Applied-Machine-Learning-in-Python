# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
def part1_scatter():
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4)
    
    
# NOTE: Uncomment the function below to visualize the data, but be sure 
# to **re-comment it before submitting this assignment to the autograder**.   
# part1_scatter()


def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    linreg = LinearRegression().fit(X_train.reshape(-1,1), y_train.reshape(-1,1))
    ret_arr = []
    ret_arr.append(np.squeeze(linreg.predict(np.linspace(0,10,100).reshape(-1,1))))
    # Your code here
    poly3 = PolynomialFeatures(degree=3)
    X_poly3 = poly3.fit_transform(x.reshape(-1,1))
    X_train_F3, X_test, y_train_F3, y_test = train_test_split(X_poly3, y,random_state = 0)
    linreg3 = LinearRegression().fit(X_train_F3, y_train_F3)
    
    ret_arr.append(linreg3.predict(poly3.transform(np.linspace(0,10,100).reshape(-1,1))))
    
    poly6 = PolynomialFeatures(degree=6)
    X_poly6 = poly6.fit_transform(x.reshape(-1,1))
    X_train_F6, X_test, y_train_F6, y_test = train_test_split(X_poly6, y,random_state = 0)
    linreg6 = LinearRegression().fit(X_train_F6, y_train_F6)
    
    ret_arr.append(linreg6.predict(poly6.transform(np.linspace(0,10,100).reshape(-1,1))))
    
    poly9 = PolynomialFeatures(degree=9)
    X_poly9 = poly9.fit_transform(x.reshape(-1,1))
    X_train_F9, X_test, y_train_F9, y_test = train_test_split(X_poly9, y,random_state = 0)
    linreg9 = LinearRegression().fit(X_train_F9, y_train_F9)
    
    ret_arr.append(linreg9.predict(poly9.transform(np.linspace(0,10,100).reshape(-1,1))))

    return np.array(ret_arr)

answer_one().shape