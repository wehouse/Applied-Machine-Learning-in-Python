# -*- coding: utf-8 -*-

# =============================================================================
# ### Question 1
# Import the data from `fraud_data.csv`. What percentage of the observations in the dataset are instances of fraud?
# 
# *This function should return a float between 0 and 1.* 
# =============================================================================

# Use X_train, X_test, y_train, y_test for all of the following questions
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

def answer_one():
    
    # Your code here
    
    return len(df[df['Class'] == 1])/len(df)

answer_one()