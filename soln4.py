# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



def blight_model():
    df = pd.read_csv('train.csv', encoding = "ISO-8859-1")
    df.index = df.ticket_id
    
    feature_list = ['fine_amount', 'admin_fee', 'state_fee', 'late_fee']    
    df.compliance = df.compliance.fillna(value=-1)
    df = df[df.compliance != -1]
    X = df[feature_list]
    X.fillna(value = -1)
    y = df.compliance
    
    # split to test and training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    
    clf = RandomForestClassifier(n_estimators = 20, max_depth = 10).fit(X_train, y_train)
   
    df_test = pd.read_csv('test.csv', encoding = "ISO-8859-1")
    
    df_test.index = df_test.ticket_id
    
    X_predict = clf.predict_proba(df_test[feature_list])
    
    return pd.Series(data = X_predict[:,1], index = df_test.ticket_id, dtype='float32')

blight_model()