# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:01:59 2022

@author: hp
"""

import pandas as pd
import numpy as np
data=pd.read_csv("Wine.csv")
data
x=data.iloc[:,0:13].values
x
y=data.iloc[:,-1].values
y
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(x)
X
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,y,train_size=0.8,random_state=0)
xtrain
xtest
ytrain
ytest
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
ypred
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm=confusion_matrix(ytest,ypred)
cm
cr=classification_report(ytest,ypred)
cr
ac=accuracy_score(ytest,ypred)
ac
model.score(xtest,ytest)
