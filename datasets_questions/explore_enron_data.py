#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib
import pandas as pd

####################### WATCH OUT WINDOWS USERS !! ######################
# Pickle's load operation fails because the pkl files end with 'CRLF' (which
# is nice for UNIX based systems like linux or MacOS, but not for Windows)
# In order to be able to use these .pkl files the CRLF endings have to be changed
# to LF. There are many ways to to this. The pkl files in this repo have already
# been converted for you using VS code.
enron_data = joblib.load(open("../final_project/final_project_dataset.pkl", "rb"))

print(f"Number of people in dataset: {len(enron_data)}")

# Pick first observation to see how many features we have
print(f"Number of features: {len(enron_data['METTS MARK'])}")

# Count number of POI's
poi_list = [person for person, values in enron_data.items() if values['poi']==1]
print(f'Number of POIs: {len(poi_list)}')

# Stock value of James Prentice
print(f"Stock value of James Prentice: {enron_data['PRENTICE JAMES']['total_stock_value']}")

# Emails from Wesley Colwell to POIs:
print(f"Emails from Wesiley Colwell to POIs: {enron_data['COLWELL WESLEY']['from_this_person_to_poi']}")

# Stock Options Values for Jeffrey K Skilling
print(f"Stock Options Values for Jeffrey K Skilling: {enron_data['SKILLING JEFFREY K']['exercised_stock_options']}")

# Total many amounts that the CEO, Chairman and CFO took home:
print(f"Jeffrey Skilling (CEO) took {enron_data['SKILLING JEFFREY K']['total_payments']} dollars home.") 
print(f"Kenneth Lay (Chairman) took {enron_data['LAY KENNETH L']['total_payments']} dollars home.") 
print(f"Andrew Fastow (CFO) took {enron_data['FASTOW ANDREW S']['total_payments']} dollars home.") 

# Number of persons with a non null salary
n = 0
for features in enron_data.values():
    if features['salary'] != 'NaN':
        n += 1
print(f"Number of persons with non null salary: {n}")

# Number of persons with non null email
n = 0
for features in enron_data.values():
    if features['email_address'] != 'NaN':
        n += 1
print(f"Number of persons with non null email: {n}")