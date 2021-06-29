# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 12:02:29 2021

@author: Ramesh
"""

import pandas as pd
names=[]
for i in 'abcdefghijklmno':
    names.append(i)


data=pd.read_csv(r'C:\Users\Ramesh\Desktop\ML\income.csv',
                 names=names)


###replace with particular value
data['b']=data['b'].replace(to_replace=' ?',
                            value=' Private')

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
##
#data['b']=le.fit_transform(data['b'])
#data['d']=le.fit_transform(data['d'])
#data['f']=le.fit_transform(data['f'])
data['g']=data['g'].replace(to_replace=' ?',
                            value=' Prof-specialty')
#data['g']=le.fit_transform(data['g'])
#data['h']=le.fit_transform(data['h'])
#data['i']=le.fit_transform(data['i'])
#data['j']=le.fit_transform(data['j'])

data['n']=data['n'].replace(to_replace=' ?',
                            value=' United-States')
#data['n']=le.fit_transform(data['n'])

for i in 'bdfghijn':    
    data[i]=le.fit_transform(data[i])
    
 #standerdtion 
#standerd scarler  
'''
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
model=ss.fit(data.iloc[:,:-1])
data.iloc[:,:-1]=model.transform(data.iloc[:,:-1])
'''
'''
#min max scalar
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)

'''

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=87)


#logestic Regression
from sklearn.linear_model import LogisticRegression
modelLR=LogisticRegression()
modelLR.fit(xtrain,ytrain)
ypredLR=modelLR.predict(xtest)



from sklearn.naive_bayes import GaussianNB
modelNB=GaussianNB()
modelNB.fit(xtrain,ytrain)

ypredNB=modelNB.predict(xtest)


#KNN
from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier()
modelknn=KNeighborsClassifier(n_neighbors=29)
modelknn.fit(xtrain,ytrain)

ypredknn=modelknn.predict(xtest)
from sklearn.metrics import accuracy_score
accuracy_score(ytest,ypredknn)*100

#Decision Tree
from sklearn.tree import  DecisionTreeClassifier
#modeldt=DecisionTreeClassifier(criterion='entropy')
modeldt=DecisionTreeClassifier(criterion='gini')
modeldt.fit(xtrain,ytrain)
ypreddt=modeldt.predict(xtest)
accuracy_score(ytest,ypreddt)*100


print("Navie bayes",accuracy_score(ytest,ypredNB)*100)
print("KNN",accuracy_score(ytest,ypredknn)*100)
print("decision tree",accuracy_score(ytest,ypreddt)*100 )
print("Logestic Regression",accuracy_score(ytest,ypredLR)*100)



#confusion matrix
from sklearn.metrics import confusion_matrix
print("confusion_matrix of Logestic",confusion_matrix(ytest,ypredLR))
      