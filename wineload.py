# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 14:57:26 2022

@author: hp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets  import load_wine #wine data load
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
wine=load_wine()
dir(wine)
wine.data
wine['frame']
wine['feature_names']
wine['target']
wine['DESCR']
pd.DataFrame(wine['data'],columns=wine['feature_names'])
x=wine.data
y=wine.target
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8,random_state=1)
model=GaussianNB()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
cm=confusion_matrix(ytest,ypred)
ac=accuracy_score(ytest,ypred)
model1=BernoulliNB()
model1.fit(xtrain,ytrain)
ypred=model1.predict(xtest)
cm=confusion_matrix(ytest,ypred)
ac=accuracy_score(ytest,ypred)
ac
model2=MultinomialNB()
model2.fit(xtrain,ytrain)
ypred=model2.predict(xtest)
cm=confusion_matrix(ytest,ypred)
ac=accuracy_score(ytest,ypred)
ac


