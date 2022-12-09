# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 10:47:29 2022

@author: hp
"""

import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt
data=pd.read_csv("Social_Network_Ads.csv")
data
data.shape
x=data.iloc[:,2:4].values
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
#knn algorithm 
#from sklearn.neighbors import KNeighborsClassifier
#acc=[]
#for i in range(1,11):
    #model=KNeighborsClassifier(n_neighbors=i,metric='minkowski',p=2)
    #model.fit(xtrain,ytrain)
   # yp=model.predict(xtest)
   # yp
    #a=accuracy_score(ytest,yp)
   # a
   # acc.append(a)
#plt.plot(range(1,11),acc)

    
knn_model=KNeighborsClassifier(n_neighbors=4,metric='minkowski',p=2)#stab=ndard nucleadian matric 
knn_model.fit(xtrain,ytrain)
ypred=knn_model.predict(xtest)
ypred
ytest
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm=confusion_matrix(ytest,ypred)
cm
cr=classification_report(ytest,ypred)
cr
ac=accuracy_score(ytest,ypred)
ac
#training

from matplotlib.colors import ListedColormap  
x_set, y_set = xtrain, ytrain  
x1, x2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
plt.contourf(x1, x2, knn_model.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
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
#testing

from matplotlib.colors import ListedColormap  
x_set, y_set = xtest,ytest  
x1, x2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
plt.contourf(x1, x2, knn_model.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha = 0.75, cmap = ListedColormap(('purple','green' )))  
plt.xlim(x1.min(), x1.max())  
plt.ylim(x2.min(), x2.max())  
for i, j in enumerate(nm.unique(y_set)):  
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c = ListedColormap(('purple', 'green'))(i), label = j)  
plt.title('knn (Testing set)')  
plt.xlabel('Age')  
plt.ylabel('Estimated Salary')  
plt.legend()  
plt.show()  

