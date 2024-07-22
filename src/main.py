import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import Preprocessing as prep
# Enable the interactive mode
#%matplotlib qt
from models_comparator import CompareRegressionModel
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# =============================================================================
# Read the differents dataset
# =============================================================================

# Data punctualite per train type
df_punc_per_train_type = pd.read_csv("../data/data_punctualite_per_train_type.csv", sep = ";")

# Data delays allocation per month
df_delays_alloc_per_month = pd.read_csv("../data/delays_allocation_per_month.csv", sep = ";")

# Data incidents with impact on punctuality.csv
df_incident_impact_punc = pd.read_csv("../data/incidents_with_impact_on_punctuality.csv", sep = ";")

# Data nationale punctuality based on canceled train
df_nat_punc_based_cancel = pd.read_csv("../data/nationale_punctuality_based_on_canceled_train.csv", sep = ";")

# Data national punctuality per month
df_nat_punc_per_month = pd.read_csv("../data/national_punctuality_per_month.csv", sep = ";")

# Data punctuality d-1
df_punc_d_1 = pd.read_csv("../data/punctuality_d-1.csv", sep = ";")

# Data punctuality per train type and per moment
df_punc_per_train_type_moment = pd.read_csv("../data/punctuality_per_train_type_and_per_moment.csv", sep = ";")

# Data punctualite TGV
df_punc_tgv = pd.read_csv("../data/punctuality_tgv.csv", sep = ";")

# Data punctuality train arrived_bxl per moment
df_punc_train_bxl = pd.read_csv("../data/punctuality_train_arrived_bxl_per_moment.csv", sep = ";")

# Data punctuality train arrived bxl per moment and long line
df_punc_train_bxl_per_moment_longline = pd.read_csv("../data/punctuality_train_arrived_bxl_per_moment_and_long_line.csv", sep = ";")

# Data punctuality train ICE
df_punc_train_ice = pd.read_csv("../data/punctuality_train_ICE.csv", sep = ";")

# Data distance station to station
df_dist_btw_stat = pd.read_csv("../data/subdataset/distance_station_to_station.csv", sep = ";")

# Data Monthly correspondence 
df_monthly_corresp = pd.read_csv("../data/subdataset/monthly_correspondence_data.csv", sep = ";")

# Data punctuality on arrival in bxl at times and long line
df_punc_arriv_bxl_per_times_longline = pd.read_csv("../data/subdataset/punctuality_on_arrival_in_bxl_at_times_and_long_line.csv", sep = ";")


# =============================================================================
#  Plot the frequence of the different incidents whith impact on punctuality
# =============================================================================
# Calculate the frequency of each modality
incident_counts = df_incident_impact_punc["Description de l'incident.2"].value_counts()
sns.barplot(x=incident_counts.values, y=incident_counts.index, palette='viridis')
plt.title('Incidents causing punctiality problem')
plt.xlabel('Frequency')
plt.ylabel('Incident')
plt.show()

# =============================================================================
#  Plot the sum of minutes delay for the different incident category
# =============================================================================
delay_sums_by_incident = df_incident_impact_punc.groupby("Description de l'incident.2")["Nombre de minutes de retard"].sum().reset_index()
sns.barplot(x="Nombre de minutes de retard", y="Description de l'incident.2", data=delay_sums_by_incident, palette='viridis')
plt.title('Sum of minutes delay for Different incidents')
plt.xlabel('Incident')
plt.ylabel('Sum of minutes delay')
plt.show()

# =============================================================================
# 
# =============================================================================
df_weather_pb = df_incident_impact_punc[df_incident_impact_punc["Description de l'incident.2"] =='Exceptional weather conditions']
df_weather_pb.reset_index(inplace = True, drop = True)


# =============================================================================
#  Preprocessing of the punctuality data per train type and moment
# =============================================================================
# Convert the first column to date
date_cols = [df_punc_per_train_type_moment.columns[0]]
df_punc_per_train_type_moment = prep.to_date(df_punc_per_train_type_moment, date_cols)

# Create dummies variables for train type and the moments
cat_vars = [df_punc_per_train_type_moment.columns[1], df_punc_per_train_type_moment.columns[4]]
df_punc_per_train_type_moment_dummies = prep.create_dummies(df_punc_per_train_type_moment, cat_vars)

# Select only the part of the dataframe which we will use to predict the delay
data_training_tr_type_moment = df_punc_per_train_type_moment_dummies.iloc[:, 5:]
# Correlation 
data_training_tr_type_momentt_corr = data_training_tr_type_moment.corr()

# Get the inputs and target
inputs = list(data_training_tr_type_moment.columns)
target = inputs.pop(3)
# Delete the train_nbr as it is correlated to train_with_less_6min_delay
inputs.pop(1) 
X_tr_type, y_tr_type = prep.select_x_y_vars(data_training_tr_type_moment, 
                            inputs, target)

# =============================================================================
#  Preprocessing of the incident impact on punctuality data 
# =============================================================================
# Convert the first column to date
date_cols = [df_incident_impact_punc.columns[0]] 
df_incident_impact_punc = prep.to_date(df_incident_impact_punc, date_cols)

# Create the dummies variablles for the place and the kind of incident
cat_vars = [df_incident_impact_punc.columns[5], df_incident_impact_punc.columns[8]]
df_incident_impact_punc_encoded = prep.encode_categ_var(df_incident_impact_punc, cat_vars)

# Select only the part of the dataframe which we will use to predict the delay
data_training_incident_impact_punc = df_incident_impact_punc_encoded.iloc[:, -4:]
# Correlation 
data_training_incident_impact_punc_corr = data_training_incident_impact_punc.corr()

# Get the inputs and target
inputs = data_training_incident_impact_punc.columns[1:]
target = data_training_incident_impact_punc.columns[0]
X_incid, y_incid = prep.select_x_y_vars(data_training_incident_impact_punc, 
                            inputs, target)

# =============================================================================
#  Train models to predict the delay in term of train type and moment
# =============================================================================
# Split the data
X_train_tr_type, X_test_tr_type, y_train_tr_type, y_test_tr_type = prep.split_data(X_tr_type,
                                                                                   y_tr_type)

# Scale the data
X_train_tr_type_sc, X_test_tr_type_sc, scaler_type = prep.scale_data(X_train_tr_type, 
                                                                     X_test_tr_type,
                                                                     minmax_or_std_sc= True)

# Train the different models and get the best one
CompareModelType = CompareRegressionModel(X_train_tr_type_sc, y_train_tr_type, 
                                            X_test_tr_type_sc, y_test_tr_type)
best_model, best_model_name, best_model_results = CompareModelType.get_best_model()
df_errors_tr_type = CompareModelType.results

# =============================================================================
#  Train models to predict the delay in term of incident and place
# =============================================================================
# Split the data
X_train_incid, X_test_incid, y_train_incid, y_test_incid = prep.split_data(X_incid, y_incid)


# Train the different models and get the best one
CompareModelIncid = CompareRegressionModel(X_train_incid, y_train_incid, X_test_incid, y_test_incid)
best_model, best_model_name, best_model_results = CompareModelIncid.get_best_model()
df_errors_incid = CompareModelIncid.results


# =============================================================================
#  GridsearchCV for Random forest model
# =============================================================================
run_grid_rf = False
if run_grid_rf:
    
    # Grid search For the incident data
    rf_model = RandomForestRegressor(random_state=42)
    
    parameters = [{"n_estimators":[10,50,100,150,200, 250, 500, 1000], 'max_depth': [5, 10, 15, 20, 30, 50, None]}]
    grid_bag_incid = GridSearchCV(estimator=rf_model, param_grid=parameters, cv=3, scoring='neg_mean_absolute_error')
    
    grid_incid = grid_bag_incid.fit(X_train_incid, y_train_incid)
    
    # Get the best score
    print(f"the best score for incident data is {grid_incid.best_score_}")
    
    # Get the best params
    print(f"the best parameters for the incident data are {grid_incid.best_params_}")
    
    
    
    # Grid search For the train type data
    rf_model_tr_type = RandomForestRegressor(random_state=42)
    
    parameters = [{"n_estimators":[10,50,100,150,200, 250, 500, 1000], 'max_depth': [5, 10, 15, 20, 30, 50, None]}]
    grid_bag_tr_type = GridSearchCV(estimator=rf_model_tr_type, param_grid=parameters, cv=3, scoring='neg_mean_absolute_error')
    
    grid_tr_type = grid_bag_tr_type.fit(X_train_tr_type_sc, y_train_tr_type)
    
    # Get the best score
    print(f"the best score for train type data is {grid_tr_type.best_score_}")
    
    # Get the best params
    print(f"the best parameters for the train type data are {grid_tr_type.best_params_}")


# =============================================================================
#  GridsearchCV for XGBoost model
# =============================================================================
run_grid_xgb = False
if run_grid_xgb:
    
    param_grid = {"n_estimators":[10,50,100,150,200, 250, 500, 1000], 
                  'max_depth': [5, 10, 15, 20, 30, 50, None],
                  'learning_rate': [0.01, 0.05, 0.1]}
    
    # Model XGBoost for incident data
    xgb_model_incid = XGBRegressor(objective='reg:absoluteerror', random_state=42)
    
    grid_search_incid = GridSearchCV(estimator=xgb_model_incid, param_grid=param_grid, 
                               scoring='neg_mean_absolute_error', cv=3, verbose=1)
    
    # Fit the grid search to the data
    grid_xgb_incid = grid_search_incid.fit(X_train_incid, y_train_incid)
    
    # Get the best score
    print(f"the best score for incident data is {grid_xgb_incid.best_score_}")
    
    # Get the best params
    print(f"the best parameters for the incident data are {grid_xgb_incid.best_params_}")
    
    
    
    # Model XGBoost for train type data
    xgb_model_tr_type = XGBRegressor(objective='reg:absoluteerror', random_state=42)
    
    grid_search_tr_type = GridSearchCV(estimator=xgb_model_tr_type, param_grid=param_grid, 
                               scoring='neg_mean_absolute_error', cv=3, verbose=1)
    
    # Fit the grid search to the data
    grid_xgb_tr_type = grid_search_tr_type.fit(X_train_tr_type_sc, y_train_tr_type)
    
    # Get the best score
    print(f"the best score for train type data is {grid_xgb_tr_type.best_score_}")
    
    # Get the best params
    print(f"the best parameters for the train type data are {grid_xgb_tr_type.best_params_}")
