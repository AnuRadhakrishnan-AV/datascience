# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:18:30 2022

@author: hp
"""

import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.svm import SVC
#from sklearn.linear_model import LinearRegression
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
data=pd.read_csv("Wine.csv")
x=data.iloc[:,0:13].values
y=data.iloc[:,-1].values
sc=StandardScaler()
X=sc.fit_transform(x)
xtrain,xtest,ytrain,ytest=train_test_split(X,y,train_size=0.8,random_state=0)
model=SVC(kernel='linear',random_state=0)
k_model
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
cm=confusion_matrix(ytest,ypred)
ac=accuracy_score(ytest,ypred)
cr=classification_report(ytest,ypred)
