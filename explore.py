import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import env

from sklearn.model_selection import train_test_split

import sklearn.preprocessing

import acquire
import prepare
from wrangle import wrangle_telco 

def plot_variable_pairs(df):
    sns.pairplot(df, kind="reg", plot_kws={'line_kws':{'color':'orange'}}) 


def months_to_years(df):
    df['tenure_years'] = (df.tenure//12)
    return df

def plot_categorical_and_continuous_vars(df, categorical_var, continuous_var):
    sns.barplot(data=df, y=continuous_var, x=categorical_var)
    plt.show()
    sns.swarmplot(data=df, y=continuous_var, x=categorical_var)
    plt.show()
    sns.boxplot(data=df, y=continuous_var, x=categorical_var)