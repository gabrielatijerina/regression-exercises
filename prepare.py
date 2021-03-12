import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

########################### Clean Telco Data ############################

def clean_telco(df):
    '''
    clean_telco will take one argument df, a pandas dataframe and will:
    fill in missing values
    replace missing values from total_charges and convert to float
    
    return: a single pandas dataframe with the above operations performed
    '''
    
    #fill missing numbers
    df = df.fillna(0)
    
    #replace total_charges missing values and convert to float
    df.total_charges = df.total_charges.str.replace(' ', '0').astype(float)
    

    return df




########################### Split Function ############################


def split(df, stratify_by=None):
    """
    Crude train, validate, test split
    To stratify, send in a column name
    """
    
    if stratify_by == None:
        train, test = train_test_split(df, test_size=.2, random_state=123)
        train, validate = train_test_split(train, test_size=.3, random_state=123)
    else:
        train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[stratify_by])
        train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train[stratify_by])
    
    return train, validate, test