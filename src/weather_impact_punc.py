# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 19:49:53 2024

@author: Yassine Zaim
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import Preprocessing as prep
from datetime import timedelta
import numpy as np
import ast

# =============================================================================
# Read the differents dataset
# =============================================================================

# Data punctuality d-1
df_punc_d_1 = pd.read_csv("../data/punctuality_d-1.csv", sep = ";")

# Data distance station to station
df_dist_btw_stat = pd.read_csv("../data/subdataset/distance_station_to_station.csv", sep = ";")

# Data Monthly correspondence 
df_monthly_corresp = pd.read_csv("../data/subdataset/monthly_correspondence_data.csv", sep = ";")

# Data punctuality on arrival in bxl at times and long line
df_meteo_be = pd.read_csv("../data/meteo_be/aws_10min.csv")


# =============================================================================
#  Preprocessing of the punctuality & meteo data 
# =============================================================================
# Convert the timestamp to datetime and add 2 hours to UTC to habe the Belgium time
df_meteo_be['timestamp'] = pd.to_datetime(df_meteo_be['timestamp'])
df_meteo_be['timestamp'] = df_meteo_be['timestamp'] + timedelta(hours=2)


# Convert the date column to datetime
df_punc_d_1['Date de départ'] = pd.to_datetime(df_punc_d_1['Date de départ'])

# Convert the time column to a time format
df_punc_d_1["Heure réelle d'arrivée"] = pd.to_datetime(df_punc_d_1["Heure réelle d'arrivée"]).dt.time
df_punc_d_1["Heure prévue d'arrivée"] = pd.to_datetime(df_punc_d_1["Heure prévue d'arrivée"]).dt.time
df_punc_d_1["Heure réelle de départ"] = pd.to_datetime(df_punc_d_1["Heure réelle de départ"]).dt.time
df_punc_d_1["Heure prévue de départ"] = pd.to_datetime(df_punc_d_1["Heure prévue de départ"]).dt.time

# Fillna by using the Heure prévue d'arrivée or Heure réelle or prévue de départ
df_punc_d_1["Heure réelle d'arrivée"] = (df_punc_d_1["Heure réelle d'arrivée"]
                              .fillna(df_punc_d_1["Heure prévue d'arrivée"])
                              .fillna(df_punc_d_1["Heure réelle de départ"])
                              .fillna(df_punc_d_1["Heure prévue de départ"]))

# Combine date and time into a new datetime column
df_punc_d_1['timestamp'] = df_punc_d_1.apply(lambda row: pd.Timestamp.combine(row['Date de départ'], 
                                                                              row["Heure réelle d'arrivée"]),
                                                                             axis=1)
# Insert the new datetime column at the first index
df_punc_d_1.insert(0, 'timestamp', df_punc_d_1.pop('timestamp'))
df_punc_d_1['timestamp'] = pd.to_datetime(df_punc_d_1['timestamp'])


# =============================================================================
#  Convert the geolocalisation to coordinate tuple
# =============================================================================
# Convert the geolocalisation coordinate from string to tupe in meteo data
df_meteo_be['the_geom'] = df_meteo_be['the_geom'].apply(lambda row: row.replace('POINT ', ''))
df_meteo_be['the_geom'] = df_meteo_be['the_geom'].apply(lambda row: row.replace(' ', ','))
df_meteo_be['the_geom'] =  df_meteo_be['the_geom'].apply(lambda row: ast.literal_eval(row))

# Convert the geolocalisation coordinate from string to tupe and permute x & y in dist_btw_stat data
df_dist_btw_stat['geo_point_2d'] =  df_dist_btw_stat['geo_point_2d'].apply(lambda row: (round(ast.literal_eval(row)[1], 3), round(ast.literal_eval(row)[0], 3)))

# =============================================================================
#  Find the same geolocalisation to merge the both dataframe
# =============================================================================
# Find the points in the both list
list_pts_meteo = list(df_meteo_be['the_geom'])
list_pts_dist = list(df_dist_btw_stat['geo_point_2d'])
ls_commun_pts = [elm for elm in list_pts_dist if elm in list_pts_meteo]
print(f'The list of commun geolocalisation points in both data have the lenght {len(ls_commun_pts)}')



'''
There is a problem of merging the weather data with punctuality because of
geolocalisation points. I will not continu this script.
'''
# # =============================================================================
# #  Merge the delay data with belgium weather data
# # =============================================================================
# # Interpolate the timestamp and the meteo data to have the same timestamp as df_punc_d_1
# df_punc_d_1_withou_dup = df_punc_d_1.drop_duplicates(subset='timestamp')
# unique_timestamps = df_punc_d_1_withou_dup['timestamp']

# # Set the timestamp as the index
# df_punc_d_1.set_index('timestamp', inplace=True)
# df_meteo_be.set_index('timestamp', inplace=True)

# Ensure no duplicates in meteo DataFrame before reindexing
#df_meteo_be = df_meteo_be[~df_meteo_be.index.duplicated(keep='first')]

# # Reindex the meteo DataFrame using the punctuality DataFrame's index
# df_meteo_reindexed = df_meteo_be.reindex(unique_timestamps)

# # Interpolate the missing values
# df_meteo_interpolated = df_meteo_reindexed.interpolate(method='time')

# # Combine the DataFrames (optional)
# df_combined = df_punctuality.join(df_meteo_interpolated) 
# df_merged = pd.merge(df_punc_d_1, df_meteo_be, on='timestamp')
