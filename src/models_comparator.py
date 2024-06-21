# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 09:49:38 2024

@author: Yassine Zaim
"""

import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


class CompareRegressionModel:
    
    def __init__(self, X_train, y_train, X_test, y_test, random_state = 42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = {
            'Decision Tree': DecisionTreeRegressor(max_depth = 10, random_state=random_state),
            'Random Forest': RandomForestRegressor(n_estimators = 150, max_depth = 10, random_state=random_state),
            'SVM': SVR(kernel = 'poly'),
            'Linear Regression': LinearRegression(),
            'ANN': MLPRegressor(hidden_layer_sizes=(60,), learning_rate='constant', learning_rate_init=0.01,
                                random_state=random_state, max_iter=10000),
            'XGBoost': XGBRegressor(random_state=random_state),
            'ANN (TensorFlow)': self.build_tensorflow_ann()
        }
        self.results = pd.DataFrame(columns = ['Model', 'MAE Train', 'RMSE Train',
                                               'MAE Test', 'RMSE Test'])


    def build_tensorflow_ann(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.X_train.shape[1], activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model


    def train_and_evaluate(self):
        ''' This function will train and test the differents models and save the results'''
        
        for name, model in self.models.items():
            if name == 'ANN (TensorFlow)':
                model.fit(self.X_train, self.y_train, epochs=100, batch_size=20, verbose=0)
                y_pred_tr = model.predict(self.X_train).flatten()
                y_pred_tst = model.predict(self.X_test).flatten()
            else:
                # Train the model
                model.fit(self.X_train, self.y_train)
                # Predict the target for train and test sets
                y_pred_tr = model.predict(self.X_train)
                y_pred_tst = model.predict(self.X_test)
            # Evaluate the model for train and test sets
            mae_tr = mean_absolute_error(self.y_train, y_pred_tr)
            rmse_tr = root_mean_squared_error(self.y_train, y_pred_tr)
            mae_tst = mean_absolute_error(self.y_test, y_pred_tst)
            rmse_tst = root_mean_squared_error(self.y_test, y_pred_tst)
            self.results = pd.concat([self.results, pd.DataFrame({'Model': [name], 
                                                                 'MAE Train': [mae_tr],
                                                                 'RMSE Train': [rmse_tr],
                                                                 'MAE Test': [mae_tst],
                                                                 'RMSE Test': [rmse_tst]})], ignore_index = True)

    def get_best_model(self):
        ''' This function will allow us to find the best model based on MAE '''
        self.train_and_evaluate()
        # Find the model with the minimum MAE on the test set
        best_model_idx = self.results['MAE Test'].idxmin()
        best_model_name = self.results.loc[best_model_idx, 'Model']
        best_model = self.models[best_model_name]
        return best_model, best_model_name, self.results.loc[best_model_idx]


if __name__ == "__main__":

    # Load dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train models
    compare_model = CompareRegressionModel(X_train, y_train, X_test, y_test)
    best_model, best_model_name, best_model_results = compare_model.get_best_model()

    # Print the best model and its results
    print(f"Best Model: {best_model_name}")
    print(f"Results:\n{best_model_results}")
