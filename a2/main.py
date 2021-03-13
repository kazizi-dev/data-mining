import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt
import math
import warnings

def preprocessing(df):
    """
        @Param: Dataframe df which holds the dataset
        @Return: None
    """

    # drop unnecessary cols/attributes
    attributes = df.columns.tolist()
    if 'Date' in attributes:
        df.drop('Date', axis=1, inplace=True)
    if 'Time' in attributes:
        df.drop('Time', axis=1, inplace=True)

    # impute missing values for attributes with type string
    attributes = df.columns.tolist()
    attributes.remove('Sub_metering_3')
    for col in attributes:
        copy = df[df[col] != '?']
        copy = copy[[col]].apply(pd.to_numeric)
        copy[col] = copy[col].replace('?', copy[col].mean())
        df[col] = copy[col]

    # impute missing values for attributes including np.nan
    for col in df.columns.tolist():
        copy = df[df[col] != np.nan]
        copy = copy[[col]].apply(pd.to_numeric)
        copy[col] = copy[col].replace(np.nan, copy[col].mean())
        df[col] = copy[col]

    # contribute string to numeric, and normalize the dataset
    for attribute in df.columns.tolist():
        df[[attribute]] = df[[attribute]].apply(pd.to_numeric)
        df[attribute] = ((df[attribute]-df[attribute].mean())/df[attribute].std(ddof=0))

    return 0


def get_min_pts(attributes):
    """
        @Param: List of attributes
        @Return: Integer value representing the MinPts
    """
    
    # length of attributes is the number of dimensions in the dataset
    k = (2*len(attributes))-1
    return k+1


def get_distance(a, b):
    """
        @Param: a is vector 1, and b is vector 2
        @Return: Epsilon value representing eps
    """
    return np.sqrt(np.sum(np.square(np.array(a) - np.array(b))))


