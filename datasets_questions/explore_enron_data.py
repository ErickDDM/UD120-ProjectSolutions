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

####################### WATCH OUT WINDOWS USERS !! ######################
# Pickle's load operation fails because the pkl files end with 'CRLF' (which
# is nice for UNIX based systems like linux or MacOS, but not for Windows)
# In order to be able to use these .pkl files the CRLF endings have to be changed
# to LF. There are many ways to to this. The pkl files in this repo have already
# been converted for you using VS code.
enron_data = joblib.load(open("../final_project/final_project_dataset.pkl", "rb"))

