#!/usr/bin/python3

"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""    


import sys
import joblib
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.linear_model import LinearRegression
dictionary = joblib.load( open("../final_project/final_project_dataset_modified.pkl", "rb") )


### list the features you want to look at--first item in the 
### list will be the "target" feature
features_list = ["bonus", "salary"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True, sort_keys = '../tools/python2_lesson06_keys.pkl')
target, features = targetFeatureSplit( data )

### training-testing split needed in regression, just like classification
from sklearn.model_selection import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"



# Fit regression
reg = LinearRegression()
reg.fit(feature_train, target_train)

# Print slope and bias
print(f"Slope: {reg.coef_[0]}")
print(f"Intercept: {reg.intercept_}")

# Print training score
print(f"Train R-squared: {reg.score(feature_train, target_train):.3f}")


# Print test score
print(f"Test R-squared: {reg.score(feature_test, target_test):.3f}")


### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")




### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass

# Oulier experiment: train on test and test on train (there is an outlier in the original training set)
reg.fit(feature_test, target_test)
plt.plot(feature_train, reg.predict(feature_train), color="b")
print(f"New line slope: {reg.coef_[0]}")

plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()