# import modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(fname):
    """ 
    The task of this function is to receive a filename and use it to fetch data from a file, which will then be stored in a Pandas Dataframe. 
	The Dataframe is returned by the function.
    """
    df = pd.read_csv(fname)
    return df 

def clean_data(df, flag):
    """ 
    This function should take a Pandas Dataframe and either remove or replace all NaN
	values. Flag will take either the value “replace” or “remove.” This function should also
	remove any columns of the data that are not numerical features. This function will return
	a cleaned Dataframe.
    """
	# if flag is 'remove'
    if flag == 'remove':
        # drop Timestap column
        cleaned_df = df.drop('Timestamp', axis=1)
  
  		# replace infinity values with NaN
        cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)
		
		# drop rows with NaN values
        cleaned_df.dropna(inplace=True)
		
		# return cleaned_df
        return cleaned_df

    elif flag == 'replace':
        # drop Timestap column
        cleaned_df = df.drop('Timestamp', axis=1)

		# replace NaN values in 'Flow Byts/s' column with 0
        cleaned_df = df["Flow Byts/s"].fillna(0)

		# replace NaN values in 'Flow Pkts/s' column with 1000000
        cleaned_df = df["Flow Pkts/s"].fillna(1000000)
   
		# Return the cleaned DataFrame
        return cleaned_df

def split_data(df, label):
    """ 
    This function should take a Pandas Dataframe and split the Dataframe into training and
	testing data. Label will be a string with one of the keys from the data (first row). This
	function should split the data into 80% for training and 20% for testing. I used the first 
	80% for training and the remaining for testing. This function will return four Dataframes: 
	X_train, y_train, X_test, and y_test.
    """
    
    # Split the DataFrame into features and target variable
    X = df.drop('Label', axis=1).astype(np.float64) # select all columns minus the target column
    y = df['Label'].values # select target column
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, y_train, X_test, y_test
