import pandas as pd
import numpy as numpy
from env import host, user, password 
from sklearn.model_selection import train_test_split


#defines function to create a sql url using personal credentials
def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'




def wrangle_telco():
    '''
    This function reads in the telco_churn data from the Codeup db,
    joins contract-, internet-service-, and payment-types tables, 

    fills in missing values
    replaces missing values from total_charges and convert to float
    
    and returns a pandas DataFrame
    '''

    #create SQL query
    sql_query = '''
                SELECT customer_id, monthly_charges, total_charges, tenure
                FROM customers
                JOIN contract_types USING (contract_type_id)
                WHERE contract_type = "Two year" ;
                '''
    
    #read in dataframe from Codeup db
    df = pd.read_sql(sql_query, get_connection('telco_churn'))

    #fill missing numbers
    df = df.fillna(0)
    
    #replace total_charges missing values and convert to float
    df.total_charges = df.total_charges.str.replace(' ', '0').astype(float)

    return df