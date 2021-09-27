#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################
# We try adaboost with a simple manual implementation of 'grid search' (we haven't
# seen this technique in the course yet, so that's why we make it this way)
num_estimators = [20, 25, 30, 35, 40]
learning_rates = [0.3, 0.4, 0.5, 0.6, 0.7]

best_accuracy = 0
best_num_estimators = None
best_learning_rate = None

for n_estimator in num_estimators:
    for learning_rate in learning_rates:
        print(f'Training using {n_estimator} estimators and learning rate of {learning_rate:.2f}')
        clf = AdaBoostClassifier(n_estimators=n_estimator, learning_rate=learning_rate)
        clf.fit(features_train, labels_train)
        preds = clf.predict(features_test)
        accuracy = accuracy_score(labels_test, preds)
        print(f"Accuracy: {accuracy:.3f}\n")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_num_estimators = n_estimator
            best_learning_rate = learning_rate

print(f'Best model accuracy: {best_accuracy}')
print(f'Best parameters: n_estimators={best_num_estimators}, learning_rate={best_learning_rate}.')








try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
