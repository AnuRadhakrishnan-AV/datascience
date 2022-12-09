# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 22:26:06 2022

@author: hp
"""

import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
data=pd.read_csv("Wine.csv")
x=data.iloc[:,0:13].values
y=data.iloc[:,-1].values
sc=StandardScaler()
X=sc.fit_transform(x)
X
model=KNeighborsClassifier(n_neighbors=4,metric='minkowski',p=2)
model.fit(X,y)
xtrain,xtest,ytrain,ytest=train_test_split(X,y,train_size=0.8,random_state=0)
ypred=model.predict(xtest)
cm=confusion_matrix(ytest,ypred)
cm
cr=classification_report(ytest,ypred)
cr
ac=accuracy_score(ytest,ypred)
ac

from matplotlib.colors import ListedColormap  
x_set, y_set = xtrain, ytrain  
x1, x2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
plt.contourf(x1, x2, model.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha = 0.75, cmap = ListedColormap(('purple','green' )))  
plt.xlim(x1.min(), x1.max())  
plt.ylim(x2.min(), x2.max())  
for i, j in enumerate(nm.unique(y_set)):  
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c = ListedColormap(('purple', 'green'))(i), label = j)  
plt.title('knn(Training set)')  
plt.xlabel('Age')  
plt.ylabel('Estimated Salary')  
plt.legend()  
plt.show()  