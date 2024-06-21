# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:14:50 2024

@author: Yassine Zaim
"""
from typing import List
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Preprocessing:
    
    @staticmethod
    def to_date(df:pd.DataFrame, date_cols: List) -> pd.DataFrame:
        ''' This function will convert the date columns to date.
        
        Parameters:
            df: DataFrame, represent the dataframe in which we want to convert
                the columns to date.
            date_cols: List, represent the list of date columns.     
            
        Returns:
            df: pd.DataFrame, which represent the input dataframe in which
                the date columns are converted.
        '''
        
        for col in date_cols:
            if df[col].dtypes != 'period[M]':
                df[col] = pd.to_datetime(df[col]).dt.to_period('M')
                # Convert to Month YYYY
                #df[col] = pd.to_datetime(df[col]).dt.strftime('%B %Y')
        return df
    
    @staticmethod
    def encode_categ_var(df: pd.DataFrame, 
                         categorical_vars: List) -> pd.DataFrame:
        ''' This function will encode the list of the given categorical 
        variables.
        
        Parameters:
            df: DataFrame, represent the dataframe in which we want to encode
                variables.
            categorical_vars: List, represent the list of categorical variables.     
            
        Returns:
            df: pd.DataFrame, which represent the input dataframe in which
                the categorical vriables are encoded.
        '''
        
        label_encoder = LabelEncoder()
        for var in categorical_vars: 
            # df[var+"_to_convert"] = df[var]

            # Encode categorical variable
            df[var+'_Encoded'] = label_encoder.fit_transform(df[var])                        
            
        return df
    
    @staticmethod
    def create_dummies(df: pd.DataFrame, 
                       categorical_vars: List) -> pd.DataFrame:
        ''' This function will create the dummies variables for the list of 
        categorical variables.
        
        Parameters:
            df: DataFrame, represent the dataframe in which we want to create
                the dummies variables.
            categorical_vars: List, represent the list of categorical variables.     
            
        Returns:
            df: pd.DataFrame, which represent the input dataframe in which
                the dummies vriables are created.
        '''
        
        for var in categorical_vars: 
            df[var+"_to_convert"] = df[var]
            df = pd.get_dummies(df, columns=[var+"_to_convert"], prefix='Cat',
                                drop_first=False, dtype=int)
            
            
        return df
    
    @staticmethod
    def select_x_y_vars(df: pd.DataFrame, inputs: List,
                        target: str) -> (pd.DataFrame, pd.Series):
        ''' This function will select the input and output variables.
        
        Parameters:
            df: DataFrame, represent the dataframe from which we want to select
                the input and output variables.
            inputs: List, represent the list of input variables.
            target: string, represent the name of the output/target variable.
            
            
        Returns:
            X: pd.DataFrame, which represent the input dataframe.
            y: pd.Series, which represent the series of output/target.
        '''
        X = df[inputs]
        y = df[target]
        return X, y
    
    @staticmethod
    def split_data(X: pd.DataFrame, y: pd.Series, train_size: float = 0.8):
        ''' This function will select the input and output variables.
        
        Parameters:
            df: DataFrame, represent the dataframe from which we want to select
                the input and output variables.
            inputs: List, represent the list of input variables.
            target: string, represent the name of the output/target variable.
            
            
        Returns:
            X: pd.DataFrame, which represent the input dataframe.
            y: pd.Series, which represent the series of output/target.
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size = train_size,
                                                            random_state=40)
        return X_train, X_test, y_train, y_test
    
    
    @staticmethod
    def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                   minmax_or_std_sc: bool = True):
        ''' This function will scale the training and test data using either 
            standard scaler or min_max scaler. If minmax_or_std_sc equale True the
            MinMax scaler will be used, else the standard scaler will be used 
            instead.
        
        Parameters:
            X_train: pd.DataFrame, represent the independant variables of the 
                    training set.
            X_test: pd.DataFrame, represent the independant variables of the 
                    test set.
            minmax_or_std_sc: bool, default value is true. indicate if we use 
                    the MinMax scaling or Standard Scaling.
            
        Returns:
            X_train_sc: pd.DataFrame, represent the scaled training set.
            X_test_sc: pd.DataFrame, represent the scaled test set.
            scaler: the scaler used to scale the data
        '''
        if minmax_or_std_sc:
            scaler = MinMaxScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)
        else:
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)
            
        return X_train_sc, X_test_sc, scaler