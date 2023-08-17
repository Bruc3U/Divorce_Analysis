# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:01:08 2022

@author: Yanis Escartin
"""
#importing the necessary libraries

import os
import pandas as pd 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence



import seaborn as sns

from dmba import stepwise_selection
from dmba import AIC_score
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression  
import dmba
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#collecting and cleaning the data

os.chdir(r'Y:\Document\Yanis\Professionel\Portfolio\divorce')
os.getcwd()

dc = pd.read_csv('divorce.csv',sep=';')
dc.info()
dc.head(10)
dc.tail(10)
dc.isna().sum()

dc.duplicated().sum()

#data seems clean, no NaaN values or duplicates 
#training 2 logistics  model to measure accuracy

X2 = dc.iloc[:, 0:54]

X2.info()

X2

y2 = dc.iloc[:,54]


y2.info()

y2

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.7, random_state = 1, shuffle=True)

X2_train.shape
X2_test.shape
y2_train.shape
y2_test.shape

model = LogisticRegression()
model.fit(X2_train, y2_train)

print("Test Accuracy: {:.2f}%".format(model.score(X2_test, y2_test) * 100))

#Elbow Method for clustering > 2

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X2)
    wcss.append(kmeans.inertia_)

wcss

plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show()

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,12)).fit(X2)
visualizer.show()



plt.figure(figsize=(16, 10))
plt.scatter(X_train_reduced.loc[y2_train == 0, 'PC1'], X_train_reduced.loc[y2_train == 0, 'PC2'], label="Married", color='blue')
plt.scatter(X_train_reduced.loc[y2_train == 1, 'PC1'], X_train_reduced.loc[y2_train == 1, 'PC2'], label="Divorced", color='orange')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Train Set")
plt.legend()
plt.show()


#PCA to reduce number of X
#2 as advised by the shoulder methods see below
n_components = 2

pca = PCA(n_components=n_components)
pca.fit(X2_train)

explained_variance = pca.explained_variance_ratio_

explained_variance 

plt.figure(figsize=(16, 10))
sns.barplot(x=pca.explained_variance_ratio_, y=["PC" + str(i) for i in range(1, n_components + 1)], orient='h', palette='husl')
plt.xlim(0., 1.)
plt.xlabel("Proportion of Variance in Original Data")
plt.title("Principal Component Variance")
plt.show()

X_train_reduced = pd.DataFrame(pca.transform(X2_train), index=X2_train.index, columns=["PC" + str(i) for i in range(1, n_components + 1)])
X_test_reduced = pd.DataFrame(pca.transform(X2_test), index=X2_test.index, columns=["PC" + str(i) for i in range(1, n_components + 1)])


#Prediction test and confusion matrix 

Xtest = X_test_reduced.copy()
ytest = y2_test.copy()
   
log_reg = sm.Logit(ytest, Xtest).fit()

log_reg.summary()

test = X_test_reduced.copy()
test['married_y'] = y2_test

 predict = log_reg.predict(test[['PC1','PC2']])
 
 predict
 
 test['true_m']= predict
 
 test.loc[test['true_m'] >= 0.6, ['prediction']] = 1
 test.loc[test['true_m'] < 0.6, ['prediction']] = 0
 
 test.drop('true_m',axis=1, inplace=True)

 test
 
 cm_t_test = confusion_matrix(test.married_y, test.prediction) 
 print ("Confusion Matrix : \n", cm_t_test) 
   
 print('Testing accuracy Model 1 = ', accuracy_score(test.married_y, test.prediction))


#random forest classifier

clf1=RandomForestClassifier(n_estimators=100)    
fit1=clf1.fit(X2_train, y2_train)
preds1 = clf1.predict(X2_test)

print(confusion_matrix(y2_test,preds1))
print(classification_report(y2_test,preds1))
print(accuracy_score(y2_test, preds1))

ran = pd.DataFrame(clf1.feature_importances_, index=X2_train.columns, columns=['importance'])

ran.sort_values('importance').plot(kind='barh', figsize=(15, 15))
plt.title('Divorce Predictors (uestions)')
