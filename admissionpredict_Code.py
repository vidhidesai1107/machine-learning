#-*- coding: utf-8 -*-
"""
Created on Sun Oct 31 23:01:08 2021

@author: Vidhi Desai 53004200014
"""
import pandas as pd
import numpy as np
import matplotlib as mat 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# replace below file with excel file which is provided
data_logr=pd.read_csv("C:/Users/admission_predict.xlsx") 
print(data_logr.head())
data_logr['Research']=data_logr["Research"].map({'YES':1,'NO':0})
print(data_logr)
featuresSet=['Serial No.', 'Total Marks', 'TOEFL Score', 'University Rating', 'CGPA', 'Chance of Admit']
logr_train=data_logr[featuresSet]
logr_test=data_logr['Research']
X_train,X_test,y_train,y_test=train_test_split(logr_train,logr_test,test_size=0.5 , random_state=1)
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
print("Precision: ",metrics.precision_score(y_test, y_pred))
print("Recall: ",metrics.recall_score(y_test, y_pred))
print("F1-Score: ",metrics.f1_score(y_test, y_pred))
