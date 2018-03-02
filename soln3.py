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



# =============================================================================
# 
# Question 2
# Using X_train, X_test, y_train, and y_test (as defined above), train a dummy classifier that classifies everything as the majority class of the training data. What is the accuracy of this classifier? What is the recall?
# 
# This function should a return a tuple with two floats, i.e. (accuracy score, recall score).
# =============================================================================

def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score
    
    # Negative class (0) is most frequent
    dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
    return dummy_majority.score(X_test,y_test), recall_score(y_test, dummy_majority.predict(X_test))

# =============================================================================
# Question 3
# Using X_train, X_test, y_train, y_test (as defined above), train a SVC classifer using the default parameters. What is the accuracy, recall, and precision of this classifier?
# 
# This function should a return a tuple with three floats, i.e. (accuracy score, recall score, precision score).
# =============================================================================

def answer_three():
    from sklearn.metrics import recall_score, precision_score
    from sklearn.svm import SVC
    svm = SVC().fit(X_train, y_train)
    
    # Your code here
    svm_predicted = svm.predict(X_test)
    return svm.score(X_test, y_test), recall_score(y_test, svm_predicted), precision_score(y_test, svm_predicted)


# =============================================================================
# Question 4
# Using the SVC classifier with parameters {'C': 1e9, 'gamma': 1e-07}, what is the confusion matrix 
# when using a threshold of -220 on the decision function. Use X_test and y_test.
# 
# This function should return a confusion matrix, a 2x2 numpy array with 4 integers.
# =============================================================================

def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC

    svm_predicted = SVC(C=1e9,gamma=1e-07).fit(X_train, y_train).decision_function(X_test) > -220

    return confusion_matrix(y_test, svm_predicted)

# =============================================================================
# Question 5
# Train a logisitic regression classifier with default parameters using X_train and y_train.
# 
# For the logisitic regression classifier, create a precision recall curve and a roc curve using y_test 
# and the probability estimates for X_test (probability it is fraud).
# 
# Looking at the precision recall curve, what is the recall when the precision is 0.75?
# 
# Looking at the roc curve, what is the true positive rate when the false positive rate is 0.16?
# 
# This function should return a tuple with two floats, i.e. (recall, true positive rate).
# =============================================================================

def answer_five():
        
    return 0.83, 0.94