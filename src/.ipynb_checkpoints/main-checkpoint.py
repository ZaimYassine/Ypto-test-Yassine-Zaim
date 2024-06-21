import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import preprocessing as prep
# Enable the interactive mode
#%matplotlib qt

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
# Show the delay for the different categories of train
# =============================================================================
# # Example time series data
# data = {
#     'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
#     'value': pd.Series(range(100)) + pd.Series(range(100)).apply(lambda x: x % 10)
# }

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Set the date column as the index
# df.set_index('date', inplace=True)

# # Create the plot
# plt.figure(figsize=(10, 6))
# plt.plot(df.index, df['value'], label='Value')
# plt.xlabel('Date')
# plt.ylabel('Value')
# plt.title('Interactive Time Series Plot')
# plt.legend()
# plt.grid(True)
# plt.show()

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

Prep = prep.Preprocessing()

# Convert the first column to date
date_cols = [df_punc_per_train_type_moment.columns[0]]
df_punc_per_train_type_moment = Prep.to_date(df_punc_per_train_type_moment, date_cols)

# Create dummies variables for train type and the moments
cat_vars = [df_punc_per_train_type_moment.columns[1], df_punc_per_train_type_moment.columns[4]]
df_punc_per_train_type_moment_dummies = Prep.create_dummies(df_punc_per_train_type_moment, cat_vars)

# Select only the part of the dataframe which we will use to predict the delay
data_punc_per_train_type_moment = df_punc_per_train_type_moment_dummies.iloc[:, 5:]
# Correlation 
data_punc_per_train_type_moment_corr = data_punc_per_train_type_moment.corr()

# =============================================================================
#  Preprocessing of the incident impact on punctuality data 
# =============================================================================
Prep_incid = prep.Preprocessing()

# Convert the first column to date
date_cols = [df_incident_impact_punc.columns[0]] 
df_incident_impact_punc = Prep.to_date(df_incident_impact_punc, date_cols)

# Create the dummies variablles for the place and the kind of incident
cat_vars = [df_incident_impact_punc.columns[5], df_incident_impact_punc.columns[8]]
df_incident_impact_punc_encoded = Prep_incid.encode_categ_var(df_incident_impact_punc, cat_vars)

# Select only the part of the dataframe which we will use to predict the delay
data_incident_impact_punc = df_incident_impact_punc_encoded.iloc[:, -4:]
# Correlation 
data_incident_impact_pun_corr = data_incident_impact_punc.corr()


# =============================================================================
#  Train models to predict the delay in term of train type and moment
# =============================================================================
# Decision tree model
# 1 Split data
# Train and test
# Evaluate


# =============================================================================
#  Train models to predict the delay in term of incident and place
# =============================================================================
# Decision tree model