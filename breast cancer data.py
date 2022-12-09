# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 12:26:00 2022

@author: hp
"""

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
df=load_breast_cancer()
df
x=df.data
y=df.target
print(df.feature_names)
x.shape
y.shape
knn_model=KNeighborsClassifier(n_neighbors=4,metric='minkowski',p=2)#stab=ndard nucleadian matric 
knn_model.fit(x,y)
ypred=knn_model.predict(x)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm=confusion_matrix(y,ypred)
cm
cr=classification_report(y,ypred)
accuracy_score(y,ypred)

