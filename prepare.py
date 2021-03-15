import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

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

########################### Scaling Functions ############################

#MinMaxScaler()
def min_max_scaler(train, validate, test):
    '''
    Takes in train, validate and test dfs with numeric values only
    Exludes string objects
    makes, uses, and transforms the data
    Returns (scaler, train_scaled, validate_scaled, test_scaled)
    '''

    #exclude objects
    train = train.select_dtypes(exclude=['object'])
    validate = validate.select_dtypes(exclude=['object'])
    test = test.select_dtypes(exclude=['object'])

    #make
    scaler = sklearn.preprocessing.MinMaxScaler()

    #fit
    scaler.fit(train)

    #use
    train_scaled = scaler.transform(train)
    validate_scaled = scaler.transform(validate)
    test_scaled = scaler.transform(test)

    # turn the numpy arrays into dataframes
    train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
    validate_scaled = pd.DataFrame(validate_scaled, columns=train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=train.columns)


    return scaler, train_scaled, validate_scaled, test_scaled 




#StandardScaler()
def standard_scaler(train, validate, test):
    '''
    Takes in train, validate and test dfs with numeric values only
    Exludes string objects
    makes, uses, and transforms the data
    Returns (scaler, train_scaled, validate_scaled, test_scaled)
    '''

    #exclude objects
    train = train.select_dtypes(exclude=['object'])
    validate = validate.select_dtypes(exclude=['object'])
    test = test.select_dtypes(exclude=['object'])

    #make
    scaler = sklearn.preprocessing.StandardScaler()

    #fit
    scaler.fit(train)

    #use
    train_scaled = scaler.transform(train)
    validate_scaled = scaler.transform(validate)
    test_scaled = scaler.transform(test)

    # turn the numpy arrays into dataframes
    train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
    validate_scaled = pd.DataFrame(validate_scaled, columns=train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=train.columns)


    return scaler, train_scaled, validate_scaled, test_scaled 




#RobustScaler()
def robust_scaler(train, validate, test):
    '''
    Takes in train, validate and test dfs with numeric values only
    Exludes string objects
    makes, uses, and transforms the data
    Returns (scaler, train_scaled, validate_scaled, test_scaled)
    '''

    #exclude objects
    train = train.select_dtypes(exclude=['object'])
    validate = validate.select_dtypes(exclude=['object'])
    test = test.select_dtypes(exclude=['object'])

    #make
    scaler = sklearn.preprocessing.RobustScaler()

    #fit
    scaler.fit(train)

    #use
    train_scaled = scaler.transform(train)
    validate_scaled = scaler.transform(validate)
    test_scaled = scaler.transform(test)

    # turn the numpy arrays into dataframes
    train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
    validate_scaled = pd.DataFrame(validate_scaled, columns=train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=train.columns)


    return scaler, train_scaled, validate_scaled, test_scaled 





########################### Visualize Scaling Function ############################

def visualize_scaled_date(scaler, scaler_name, feature):
    scaled = scaler.fit_transform(train[[feature]])
    fig = plt.figure(figsize = (12,6))

    gs = plt.GridSpec(2,2)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])

    ax1.scatter(train[[feature]], scaled)
    ax1.set(xlabel = feature, ylabel = 'Scaled_' + feature, title = scaler_name)

    ax2.hist(train[[feature]])
    ax2.set(title = 'Original')

    ax3.hist(scaled)
    ax3.set(title = 'Scaled')
    plt.tight_layout()