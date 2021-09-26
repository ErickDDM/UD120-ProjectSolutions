#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
clf = SVC(kernel='linear')

# Only use 1% of the original training data
features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]

# Fit and Time
t0 = time()
clf.fit(features_train, labels_train)
print(f'Train time: {time() - t0:.2f}')

# Predict and time
t0 = time()
preds = clf.predict(features_test)
print(f'Predict time: {time() - t0:.2f}')
print(f'Accuracy: {accuracy_score(preds, labels_test):.2f}')

#########################################################