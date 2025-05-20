import numpy as np
import pandas as pd
import os
import joblib
import json
import datetime
from scipy import stats
from haversine import haversine
from sklearn.cluster import MiniBatchKMeans
from math import radians, cos, sin, asin, sqrt

def haversine_array(lat1, lng1, lat2, lng2):
    '''
    Measures the great-circle distance between pickup and dropoff points. 
        This is the shortest distance over the Earth's surface (in kilometers).
    '''
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    '''
    An approximation of the Manhattan distance by summing the north-south and east-west components using haversine
    '''
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b


def bearing_array(lat1, lng1, lat2, lng2):
    '''
    Indicates the compass direction (in degrees) from the pickup point to the dropoff point.
    '''
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

def delete_infrequent_categories(df, categorical_features, threshold=5):
    """
    Deletes rows containing infrequent categories in the specified categorical features.
    """
    for feature in categorical_features:

        category_counts = df[feature].value_counts()
        
        infrequent_categories = category_counts[category_counts < threshold].index
        
        df = df[~df[feature].isin(infrequent_categories)]

    return df

def Coordinate_deltas_and_midpoints(df):
        
        ablg = df['pickup_longitude'] + df['dropoff_longitude']
        ablt = df['pickup_latitude'] + df['dropoff_latitude']

        #The absolute differences between pickup and dropoff coordinates. These capture the magnitude of displacement.
        df['delta_longitude'] = np.abs(df['pickup_longitude'] - df['dropoff_longitude'])
        df['delta_latitude'] = np.abs(df['pickup_latitude'] - df['dropoff_latitude'])

        #Represent the center point of each trip. Useful for spatial clustering or heatmaps.
        df['midpoint_latitude'] = np.abs(ablt / 2)
        df['midpoint_longitude'] = np.abs(ablg / 2)

        #The straight-line (as-the-crow-flies) distance in degrees.
        df['euclidean_distance'] = np.sqrt(df['delta_longitude'] ** 2 + df['delta_latitude'] ** 2)
        return df

def City_center_distance(df):
        #Computed the Haversine distance between the trip’s midpoint and the city center (defined as Times Square: 40.758, -73.985). 
        city_center = (40.758, -73.985)
        df['distance_to_center'] = df.apply(
            lambda row: haversine((row['midpoint_latitude'], row['midpoint_longitude']), city_center),
            axis=1
        )
        return df

def Create_location_clusters(df, train=1, kmeans=None):
    '''
      To capture regional patterns in trip behavior, we applied clustering to the geographic coordinates of pickup and dropoff points using **MiniBatch K-Means**.
    '''
    if train == 1:
        coords = np.vstack((df[['pickup_latitude', 'pickup_longitude']].values,
                            df[['dropoff_latitude', 'dropoff_longitude']].values))
        simple_ind = np.random.permutation(len(coords))[:5000]
        kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000, random_state=42).fit(coords[simple_ind])
    
    # Use the trained k-means model to assign cluster labels for pickup and dropoff locations,
    # whether we're in training or testing phase.
    df.loc[:, 'pickup_cluster'] = kmeans.predict(df[['pickup_latitude', 'pickup_longitude']].values)
    df.loc[:, 'dropoff_cluster'] = kmeans.predict(df[['dropoff_latitude', 'dropoff_longitude']].values)

    # Additional features
    df['pickup_cluster_hour'] = df['pickup_cluster'].astype(str) + "_" + df['hour'].astype(str)
    df['dropoff_cluster_hour'] = df['dropoff_cluster'].astype(str) + "_" + df['hour'].astype(str)
    df['trip_type'] = (df['pickup_cluster'] == df['dropoff_cluster']).map({True: 'same', False: 'different'})
    df['pickup_dropoff_pair'] = df['pickup_cluster'].astype(str) + "_" + df['dropoff_cluster'].astype(str)

    if train == 1:
        return df, kmeans
    else:
        return df

def Create_average_durations_by_clusters(df, train=1, avg_duration=None):
    if train==1:
        avg_duration=[]
        pickup_avg_duration = df.groupby('pickup_cluster')['trip_duration'].mean().to_dict()
        dropoff_avg_duration = df.groupby('dropoff_cluster')['trip_duration'].mean().to_dict()
        global_avg = df['trip_duration'].mean()
        avg_duration = [pickup_avg_duration, dropoff_avg_duration, global_avg]


    

    df['avg_duration_by_pickup_cluster'] = df['pickup_cluster'].map(avg_duration[0])
    df['avg_duration_by_pickup_cluster'] = df['avg_duration_by_pickup_cluster'].fillna(avg_duration[2])

    df['avg_duration_by_dropoff_cluster'] = df['dropoff_cluster'].map(avg_duration[1])
    df['avg_duration_by_dropoff_cluster'] = df['avg_duration_by_dropoff_cluster'].fillna(avg_duration[2])

    if train == 1:
        return df, avg_duration
    else:
        return df

def Create_cluster_density_features(df,train=1,counts=None):
    if train == 1:
        counts=[]
        pickup_counts = df['pickup_cluster'].value_counts().to_dict()
        dropoff_counts = df['dropoff_cluster'].value_counts().to_dict()
        counts=[pickup_counts,dropoff_counts]

    
    df['pickup_density'] = df['pickup_cluster'].map(counts[0])
    df['pickup_density'] = df['pickup_density'].fillna(0)

    df['dropoff_density'] = df['dropoff_cluster'].map(counts[1])
    df['dropoff_density'] = df['dropoff_density'].fillna(0)

    if train == 1:
        return df, counts
    else:
        return df

def Create_time_features(df,train=1,high_traffic_hours=None):
    if train == 1:
        high_traffic_hours = [h for h, count in df['hour'].value_counts().items() if count > 45000]
    
    # Weekend indicator
    df['is_weekend'] = df['dayofweek'].apply(lambda x: int(x in [5, 6]))

    # Day part categorization
    df['day_part'] = df['hour'].apply(lambda hour:
                                    'night' if hour < 6 else
                                    'morning' if hour < 12 else
                                    'afternoon' if hour < 18 else
                                    'evening')
    

    # Traffic indicators
    df['high_traffic_hour'] = df['hour'].isin(high_traffic_hours).astype(int)

    # Estimated congestion
    df['estimated_congestion'] = df['hour'].apply(lambda hour:
                                                "very_low" if hour < 6 else "high")
    
    # Cyclical time encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    if train == 1:
        return df, high_traffic_hours
    else:
        return df


def Create_direction_bucket_feature(df):
        '''
        Created a categorical feature `direction_bucket` by discretizing the trip’s directional bearing into compass buckets:
               - 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W'
        '''
        df['direction_bucket'] = df['direction'].apply(lambda angle:
                                                       'N' if (angle < 45 or angle >= 315) else
                                                       'NE' if angle < 90 else
                                                       'E' if angle < 135 else
                                                       'SE' if angle < 180 else
                                                       'S' if angle < 225 else
                                                       'SW' if angle < 270 else
                                                       'W')
        
        return df


def Create_interaction_features(df):
        
        df['distance_hour_interaction'] = df['distance_haversine'] * df['hour']
        df['distance_x_traffic'] = df['distance_haversine'] * df['high_traffic_hour']
        df['weekend_x_hour'] = df['is_weekend'] * df['hour']
        df['passengers_x_distance'] = df['passenger_count'] * df['distance_haversine']
        df['is_group_ride'] = df['passenger_count'].apply(lambda x: int(x > 2))

        # Distance category
        df['trip_distance_category'] = df['distance_haversine'].apply(lambda d:
                                                                      0 if d < 2 else
                                                                      1 if d <= 5 else
                                                                      2)
        
        return df

if __name__ == "__main__":
     # NOTE: if did not unzip the data you can replace the .csv with .zip should be work 
    df_train = pd.read_csv('Data\Trip_train.csv') # 'split/train.zip'
    df_val = pd.read_csv('Data\Trip_val.csv')     # 'split/val.zip'  

    df_train['trip_duration'] = np.log1p(df_train['trip_duration'])
    df_val['trip_duration']  = np.log1p(df_val['trip_duration'])
    
    df_train["pickup_datetime"] = pd.to_datetime(df_train["pickup_datetime"]) 
    df_val["pickup_datetime"] = pd.to_datetime(df_val["pickup_datetime"])
   
    # From our EDA, can use these distance features.
    df_train.loc[:, 'distance_haversine'] = haversine_array(
            df_train['pickup_latitude'].values, df_train['pickup_longitude'].values,
            df_train['dropoff_latitude'].values, df_train['dropoff_longitude'].values
        )
    df_val.loc[:, 'distance_haversine'] = haversine_array(
            df_val['pickup_latitude'].values, df_val['pickup_longitude'].values,
            df_val['dropoff_latitude'].values, df_val['dropoff_longitude'].values
        )
    
    df_train.loc[:, 'direction'] = bearing_array(
            df_train['pickup_latitude'].values, df_train['pickup_longitude'].values,
            df_train['dropoff_latitude'].values, df_train['dropoff_longitude'].values
        )
    df_val.loc[:, 'direction'] = bearing_array(
            df_val['pickup_latitude'].values, df_val['pickup_longitude'].values,
            df_val['dropoff_latitude'].values, df_val['dropoff_longitude'].values
        )
    
    df_train.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(
            df_train['pickup_latitude'].values, df_train['pickup_longitude'].values,
            df_train['dropoff_latitude'].values, df_train['dropoff_longitude'].values
        )
    df_val.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(
            df_val['pickup_latitude'].values, df_val['pickup_longitude'].values,
            df_val['dropoff_latitude'].values, df_val['dropoff_longitude'].values
        )
    
    # From our EDA, we can use these features.
    bins = [0, 2, 5, 8, 11, 12]  # 0, 2, 5, 8, 11, 12 represent the starting and ending months of each season
    labels = ['0', '1', '2', '3', '4'] # Labels for each season ['Winter', 'Spring', 'Summer', 'Autumn', 'Winter']
    df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'])
    df_train['dayofweek'] = df_train.pickup_datetime.dt.dayofweek
    df_train['month'] = df_train.pickup_datetime.dt.month
    df_train['hour'] = df_train.pickup_datetime.dt.hour
    df_train['dayofmonth'] = df_train.pickup_datetime.dt.day
    df_train['season'] = pd.cut(df_train["month"] , bins=bins, labels=labels, right=False,ordered=False)

    df_val['pickup_datetime'] = pd.to_datetime(df_val['pickup_datetime'])
    df_val['dayofweek'] = df_val.pickup_datetime.dt.dayofweek
    df_val['month'] = df_val.pickup_datetime.dt.month
    df_val['hour'] = df_val.pickup_datetime.dt.hour
    df_val['dayofmonth'] = df_val.pickup_datetime.dt.day
    df_val['season'] = pd.cut(df_val["month"] , bins=bins, labels=labels, right=False,ordered=False)

    df_train.drop(columns=['id', 'pickup_datetime'], inplace=True)
    df_val.drop(columns=['id', 'pickup_datetime'], inplace=True)

    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    
    df_train=Coordinate_deltas_and_midpoints(df_train)
    df_val=Coordinate_deltas_and_midpoints(df_val)

    df_train=City_center_distance(df_train)
    df_val=City_center_distance(df_val)
    
    # 1. Location Clustering
    df_train, kmeans_model = Create_location_clusters(df_train, train=1)
    df_val = Create_location_clusters(df_val, train=0, kmeans=kmeans_model)
    
    # 2. Average Durations by Clusters
    df_train, avg_duration_info = Create_average_durations_by_clusters(df_train, train=1)
    df_val = Create_average_durations_by_clusters(df_val, train=0, avg_duration=avg_duration_info)

    # 3. Cluster Density Features
    df_train, density_info = Create_cluster_density_features(df_train, train=1)
    df_val = Create_cluster_density_features(df_val, train=0, counts=density_info)

    # 4. Time Features
    df_train, high_traffic_hours = Create_time_features(df_train, train=1)
    df_val = Create_time_features(df_val, train=0, high_traffic_hours=high_traffic_hours)

    df_train=Create_direction_bucket_feature(df_train)
    df_val=Create_direction_bucket_feature(df_val)

    df_train=Create_interaction_features(df_train)
    df_val=Create_interaction_features(df_val)

    directory = f'Process_data/trained_assets'

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save this models and load them to test data
    joblib.dump(kmeans_model, os.path.join(directory, 'kmeans_model.pkl'))
    joblib.dump(avg_duration_info, os.path.join(directory, 'avg_duration_info.pkl'))
    joblib.dump(density_info, os.path.join(directory, 'density_info.pkl'))
    joblib.dump(high_traffic_hours, os.path.join(directory, 'high_traffic_hours.pkl'))
    

    # From our EDA, we need to use remove outliers from passenger_count and seems distance_km is necessary also.
    categorical_features = ['passenger_count',  "hour", "month","season",'store_and_fwd_flag']

    threshold=40
    df_train = delete_infrequent_categories(df_train, categorical_features, threshold=threshold)

    
    directory = f'Process_data'

    if not os.path.exists(directory):
        os.makedirs(directory)

    df_train.to_csv(f"{directory}/train.csv", index=False)
    df_val.to_csv(f"{directory}/val.csv", index=False)

    # Save data_overview
    data_overview = {
        
        'description': "Extract New Features For Data and removed the infrequent item for categorical data .",
        'feature_names': df_train.columns.tolist(),
        'num_rows_train': len(df_train),
        'num_rows_val': len(df_val),
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(f"{directory}/data_overview.json", 'w') as f:
        json.dump(data_overview, f, indent=4)

    print("Data and data_overview have been saved successfully.")


       

    

