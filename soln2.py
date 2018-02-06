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

def plot_one(degree_predictions):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)

#plot_one(answer_one())
    
#    Question 2
#Write a function that fits a polynomial LinearRegression model on the training data X_train for degrees 0 through 9. For each model compute the  R2R2 (coefficient of determination) regression score on the training data as well as the the test data, and return both of these arrays in a tuple.
#This function should return one tuple of numpy arrays (r2_train, r2_test). Both arrays should have shape (10,)

def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score
    r2_train = []
    r2_test = []
    for degree in range(0,10):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_train.reshape(11,1))
        X_test_poly = poly.fit_transform(X_test.reshape(4,1))
        
        # Create regressor 
        linreg = LinearRegression().fit(X_poly, y_train)
        
        #Add to array one by one
        r2_train.append(r2_score(y_train,linreg.predict(X_poly)))
        r2_test.append(r2_score(y_test,linreg.predict(X_test_poly)))
    return (np.array(r2_train), np.array(r2_test))

# =============================================================================
# 
# Based on the $R^2$ scores from question 2 (degree levels 0 through 9), what degree level corresponds to a model that is underfitting? What degree level corresponds to a model that is overfitting? What choice of degree level would provide a model with good generalization performance on this dataset? 
# 
# Hint: Try plotting the $R^2$ scores from question 2 to visualize the relationship between degree level and $R^2$. Remember to comment out the import matplotlib line before submission.
# 
# *This function should return one tuple with the degree values in this order: `(Underfitting, Overfitting, Good_Generalization)`. There might be multiple correct solutions, however, you only need to return one possible solution, for example, (1,2,3).* 
# 
# 
# =============================================================================

def answer_three():
    
    # Your code here
    
    return (1,9, 6)


# =============================================================================
# Question 4
# Training models on high degree polynomial features can result in overly complex models that overfit, so we often use regularized versions of the model to constrain model complexity, as we saw with Ridge and Lasso linear regression.
# For this question, train two models: a non-regularized LinearRegression model (default parameters) and a regularized Lasso Regression model (with parameters  alpha=0.01, max_iter=10000) both on polynomial features of degree 12. Return the  R2R2  score for both the LinearRegression and Lasso model's test sets.
# This function should return one tuple (LinearRegression_R2_test_score, Lasso_R2_test_score)
# =============================================================================

def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    poly = PolynomialFeatures(degree=12)
    X_poly = poly.fit_transform(X_train.reshape(11,1))
    X_test_poly = poly.fit_transform(X_test.reshape(4,1))
        
    # Create regressor 
    linreg = LinearRegression().fit(X_poly, y_train)
    
    X_train_scaled = scaler.fit_transform(X_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    linlasso = Lasso(alpha=0.01, max_iter = 10000).fit(X_poly, y_train)

    return r2_score(y_test, linreg.predict(X_test_poly)) , r2_score(y_test, linlasso.predict(X_test_poly))

answer_four()

# =============================================================================
# 
# Question 5
# Using X_train2 and y_train2 from the preceeding cell, train a DecisionTreeClassifier with default parameters and random_state=0. What are the 5 most important features found by the decision tree?
# As a reminder, the feature names are available in the X_train2.columns property, and the order of the features in X_train2.columns matches the order of the feature importance values in the classifier's feature_importances_ property.
# This function should return a list of length 5 containing the feature names in descending order of importance.
# Note: remember that you also need to set random_state in the DecisionTreeClassifier.
# =============================================================================


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2

def answer_five():
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train2, y_train2)
    ret_arr = []
    
    for feature, importance in zip(X_train2.columns, clf.feature_importances_):
        ret_arr.append((importance, feature))
    ret_arr.sort(reverse=True)
    
    feat_arr = []
    for importance, feature in ret_arr[:5]:
        feat_arr.append(feature)
    return ret_arr


answer_five()


def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    train_scores, test_scores = validation_curve(SVC(), X_subset, y_subset,
                                            param_name='gamma',
                                            param_range=np.logspace(-4,1,6), scoring='accuracy')
   
    return train_scores.mean(axis=1),  test_scores.mean(axis=1)



def answer_seven():
    
    return (0.001, 10, 0.1)