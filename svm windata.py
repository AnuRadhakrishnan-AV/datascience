# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:02:31 2022

@author: hp
"""

import pandas as pd
import numpy as nm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.svm import SVC#svm algorithm
import matplotlib.pyplot as plt
data=pd.read_csv("Wine.csv")
x=data.iloc[:,0:13].values
y=data.iloc[:,-1].values
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8,random_state=0)
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)
model=SVC(kernel='linear',random_state=0) #call svc function
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
cm=confusion_matrix(ytest,ypred)
cm
cr=classification_report(ytest,ypred)
cr
ac=accuracy_score(ytest,ypred)
ac
