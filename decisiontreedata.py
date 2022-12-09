# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 16:23:39 2022

@author: hp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
data=pd.read_csv("salaries (4).csv")
le=LabelEncoder()
data["company"]=le.fit_transform(data["company"])
data["job"]=le.fit_transform(data["job"])
data["degree"]=le.fit_transform(data["degree"])
x=data.iloc[:,0:3]
y=data.iloc[:,-1]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8,random_state=0)
model=DecisionTreeClassifier()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
cm=confusion_matrix(ytest,ypred)
cr=classification_report(ytset,ypred)
ac=accuracy_score(ytest,ypred)
ac


from sklearn.linear_model import 