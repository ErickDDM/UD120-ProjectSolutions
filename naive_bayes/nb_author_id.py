#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels

####################### WATCH OUT WINDOWS USERS !! ######################
# Pickle's load operation fails because the pkl files end with 'CRLF' (which
# is nice for UNIX based systems like linux or MacOS) but not for Windows)
# In order to be able to use this files the CRLF endings have to be changed
# to LF. There are many ways to to this. The pkl files in this repo have already
# been converted for you.
features_train, features_test, labels_train, labels_test = preprocess()


##############################################################
clf = GaussianNB()
clf.fit(features_train, labels_train)

# Time training step
t0 = time()
clf.fit(features_train, labels_train)
print(f"Training Time: {round(time()-t0, 3)} s")

# Time prediction step
t0 = time()
labels_pred = clf.predict(features_test)
print(f"Predicting Time: {round(time()-t0, 3)} s")

# Print accuracy
print(f"Accuracy: {accuracy_score(labels_test, labels_pred):.2f}")
##############################################################