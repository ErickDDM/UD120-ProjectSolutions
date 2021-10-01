#!/usr/bin/python
import numpy as np

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    # Find errors (important to take absolute values)
    errors = np.abs(net_worths-predictions)
    
    # Find index order that would sort the errors in ascending order
    sorted_indices = np.argsort(errors, axis=0)

    # Drop the top 10% of the indexes
    filtered_indices = sorted_indices[:int(0.9*len(sorted_indices))]

    # Create filtered data object
    cleaned_data = [ (float(ages[i]), float(net_worths[i]), float(predictions[i])) for i in filtered_indices]

    return cleaned_data

