# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 10:35:59 2022

@author: hp
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.tree import export_text
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB 

data=pd.read_csv("Social_Network_Ads.csv")
x=data.iloc[:,2:4].values
y=data.iloc[:,-1].values
sc=StandardScaler()
X=sc.fit_transform(x)
xtrain,xtest,ytrain,ytest=train_test_split(X,y,train_size=0.8,random_state=0)
#decision tree algorithm
model=DecisionTreeClassifier(criterion="entropy",random_state=0)
model.fit(xtrain,ytrain)
tree=export_text(model,feature_names=['age','salary'])
ypred=model.predict(xtest)
cm=confusion_matrix(ytest,ypred)
ac=accuracy_score(ytest,ypred)
cr=classification_report(ytest,ypred)
#logistic regression
lg_model=LogisticRegression()
lg_model.fit(xtrain,ytrain)
y_pred=lg_model.predict(xtest)
cm1=confusion_matrix(ytest,y_pred)
ac1=accuracy_score(ytest,y_pred)
ac1
#knn algorithm
knn=KNeighborsClassifier(n_neighbors=4,metric='minkowski',p=2)
knn.fit(xtrain,ytrain)
ypred1=knn.predict(xtest)
cm2=confusion_matrix(ytest,ypred1)
ac2=accuracy_score(ytest,ypred1)
ac2
s_model=SVC(kernel='linear',random_state=0)
s_model.fit(xtrain,ytrain)
ypred2=s_model.predict(xtest)
n_model=GaussianNB()
n_model.fit(xtrain,ytrain)
ypred3=n_model.predict(xtest)


#roc,auc curvec
#from sklearn.metrics import roc_curve,auc,roc_auc_score
#fpr,tpr,thresh=roc_curve(ytest,ypred)


#a=auc(fpr,tpr)
#plt.plot(fpr,tpr,color="green",label=("AUC value: %0.2f"%(a)))
#plt.plot([0,1],[0,1],"--",color="red")
#plt.xlabel("False positive rate")
#plt.ylabel("True positive rate")
#plt.title("ROC-AUC CURVE")
#plt.legend(loc="best")
#plt.show()

from sklearn.metrics import roc_auc_score,roc_curve,auc
fpr,tpr,thresh=roc_curve(ytest,ypred)
a = auc(fpr,tpr)


fpr1,tpr1,thresh = roc_curve(ytest,y_pred)
b = auc(fpr1,tpr1)

fpr2,tpr2,thresh=roc_curve(ytest,ypred1)
c = auc(fpr2,tpr2)

fpr3,tpr3,thresh=roc_curve(ytest,ypred2)
d = auc(fpr3,tpr3)

fpr4,tpr4,thresh=roc_curve(ytest,ypred3)
e = auc(fpr4,tpr4)

plt.plot(fpr,tpr,color="green",label=("AUC value of Decision tree: %0.2f"%(a)))
plt.plot(fpr1,tpr1,color="blue",label=("AUC value of logistic Regression: %0.2f"%(b)))
plt.plot(fpr1,tpr2,color="red",label=("AUC value of logistic Regression: %0.2f"%(c)))
plt.plot(fpr1,tpr3,color="yellow",label=("AUC value of logistic Regression: %0.2f"%(d)))
plt.plot(fpr1,tpr4,color="purple",label=("AUC value of logistic Regression: %0.2f"%(e)))
plt.plot([0,1],[0,1],"--",color="red")
plt.xlabel("False positive rate")
plt.ylabel("True Positive rate")
plt.title("ROC-AUC Curve")
plt.legend(loc="best")
plt.show()



