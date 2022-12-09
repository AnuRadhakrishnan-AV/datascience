# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
data=pd.read_csv("diabetes.csv")

x=data.iloc[:,0:8].values
y=data.iloc[:,-1].values

xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8,random_state=0)
xtrain
xtest
ytrain
ytest
model=LogisticRegression()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
ypred
ytest
cm=confusion_matrix(ytest,ypred)
cm
model.score(xtest,ytest)
cr=classification_report(ytest,ypred)
cr
ac=accuracy_score(ytest,ypred)
ac
from matplotlib.colors import ListedColormap  
x_set, y_set = xtrain, ytrain  
x1, x2 = np.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
mtp.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha = 0.75, cmap = ListedColormap(('purple','green' )))  
pt.xlim(x1.min(), x1.max())  
plt.ylim(x2.min(), x2.max())  
for i, j in enumerate(np.unique(y_set)):  
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c = ListedColormap(('purple', 'green'))(i), label = j)  
plt.title('Logistic Regression (Training set)')  
plt.xlabel('Age')  
plt.ylabel('Estimated Salary')  
plt.legend()  
plt.show()  