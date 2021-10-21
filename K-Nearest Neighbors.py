# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 12:21:40 2021
# K-Nearest Neighbors
Estimated time needed: **25** minutes
## Objectives
After completing this lab you will be able to:
*   Use K Nearest neighbors to classify data
In this Lab you will load a customer dataset, fit the data, and use K-Nearest Neighbors to predict a data point. But what is K-Nearest Neighbors?
K-Nearest Neighbors is a supervised learning algorithm. 
Where the data is 'trained' with data points corresponding to their classification. 

To predict the class of a given data point, it takes into account the classes of the
'K' nearest data points and chooses the class in which the majority of the 'K' nearest
data points belong to as the predicted class.
"""
import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing



# About the dataset

# Imagine a telecommunications provider has segmented its customer base by service usage patterns, categorizing the customers into four groups.
# If demographic data can be used to predict group membership, the company can customize offers for individual prospective customers. 
# It is a classification problem. That is, given the dataset, with predefined labels, we need to build a model to be used to predict class of a new 
# or unknown case.

# The example focuses on using demographic data, such as region, age, and marital, to predict usage patterns.
# The target field, called custcat, has four possible values that correspond to the four customer groups, as follows: 
# 1- Basic Service 2- E-Service 3- Plus Service 4- Total Service
# Our objective is to build a classifier, to predict the class of unknown cases.
# We will use a specific type of classification called K nearest neighbour.



# downlaoding data
url ='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv'
r = requests.get(url)
open('teleCust1000t.csv', 'wb').write(r.content)

## Reading the data in
df = pd.read_csv('teleCust1000t.csv')

#%%  Data Visualization and Analysis
# Let’s see how many of each class is in our data set
print(df['custcat'].value_counts())
# 281 Plus Service, 266 Basic-service, 236 Total Service, and 217 E-Service customers

#%% explore data using visualization techniques
df.hist(column= 'age', bins=50)

#%% Feature set
# Let's define feature sets, X:
# print(df.columns)

# To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array:
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
# print(X[0:5])

# What are our labels?
y = df['custcat'].values
#%% Normalize Data
# Data Standardization gives the data zero mean and unit variance, it is good practice, 
# especially for algorithms such as KNN which is based on the distance of data points:
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
 
#%% Train Test Split¶

# Out of Sample Accuracy is the percentage of correct predictions that the model makes on data that that the model has NOT been trained on.
# Doing a train and test on the same dataset will most likely have low out-of-sample accuracy, due to the likelihood of our model overfitting.

# It is important that our models have a high, out-of-sample accuracy, because the purpose of any model, of course, 
# is to make correct predictions on unknown data. So how can we improve out-of-sample accuracy? 
# One way is to use an evaluation approach called Train/Test Split. Train/Test Split involves splitting
#  the dataset into training and testing sets respectively, which are mutually exclusive. After which,
#  you train with the training set and test with the testing set.
# This will provide a more accurate evaluation on out-of-sample accuracy because the testing dataset is not part 
# of the dataset that has been used to train the model. It is more realistic for real world problems.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#%% Classification
# K nearest neighbor (KNN)
# Import library
# Classifier implementing the k-nearest neighbors vote.
from sklearn.neighbors import KNeighborsClassifier

# Training

# Let's start the algorithm with k=4 for now:
k = 4

#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

# Predicting
# We can use the model to make predictions on the test set:
yhat = neigh.predict(X_test)
# yhat[0:5]

# Accuracy evaluation
# In multilabel classification, accuracy classification score is a function that computes subset accuracy.
# This function is equal to the jaccard_score function.
# Essentially, it calculates how closely the actual labels and predicted labels are matched in the test set.
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

#%% What about other K?

# K in KNN, is the number of nearest neighbors to examine. 
# It is supposed to be specified by the user. So, how can we choose right value for K? 
# The general solution is to reserve a part of your data for testing the accuracy of the model. 
# Then choose k =1, use the training part for modeling, and calculate the accuracy of prediction using all samples in your test set. 
# Repeat this process, increasing the k, and see which k is the best for your model.

# We can calculate the accuracy of KNN for different values of k.

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

# Plot the model accuracy for a different number of neighbors.
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show() 