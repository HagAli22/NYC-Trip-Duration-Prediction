import numpy as np
import pandas as pd
from Prepare_Data import haversine_array , bearing_array ,  dummy_manhattan_distance,Coordinate_deltas_and_midpoints,City_center_distance,\
    Create_location_clusters,Create_average_durations_by_clusters,Create_cluster_density_features,Create_time_features,Create_direction_bucket_feature,Create_interaction_features
from train import log_transform, with_suffix, FunctionTransformer, predict_eval, target
from Help import load_model 
import joblib
import os


def prepare_data(df_test:pd.DataFrame):
    
    df_test['trip_duration'] = np.log1p(df_test['trip_duration']) 
    df_test["pickup_datetime"] = pd.to_datetime(df_test["pickup_datetime"]) 
      
    # From our EDA, can use these distance features.
    df_test.loc[:, 'distance_haversine'] = haversine_array(
            df_test['pickup_latitude'].values, df_test['pickup_longitude'].values,
            df_test['dropoff_latitude'].values, df_test['dropoff_longitude'].values
        )
    df_test.loc[:, 'direction'] = bearing_array(
            df_test['pickup_latitude'].values, df_test['pickup_longitude'].values,
            df_test['dropoff_latitude'].values, df_test['dropoff_longitude'].values
        )
    df_test.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(
            df_test['pickup_latitude'].values, df_test['pickup_longitude'].values,
            df_test['dropoff_latitude'].values, df_test['dropoff_longitude'].values
        )

    bins = [0, 2, 5, 8, 11, 12]  # 0, 2, 5, 8, 11, 12 represent the starting and ending months of each season
    labels = ['0', '1', '2', '3', '4'] # Labels for each season ['Winter', 'Spring', 'Summer', 'Autumn', 'Winter'] 

    df_test["hour"] = df_test["pickup_datetime"].dt.hour
    df_test["day"]  = df_test["pickup_datetime"].dt.day
    df_test["dayofweek"] = df_test["pickup_datetime"].dt.dayofweek
    df_test["month"]  = df_test["pickup_datetime"].dt.month
    df_test['dayofmonth'] = df_test.pickup_datetime.dt.day
    df_test['season'] = pd.cut(df_test["month"] , bins=bins, labels=labels, right=False,ordered=False) 

    df_test.drop(columns=['id', 'pickup_datetime'], inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    df_test=Coordinate_deltas_and_midpoints(df_test)
    df_test=City_center_distance(df_test)

    directory = 'Process_data/trained_assets'
    kmeans_model = joblib.load(os.path.join(directory, 'kmeans_model.pkl'))
    avg_duration_info = joblib.load(os.path.join(directory, 'avg_duration_info.pkl'))
    density_info = joblib.load(os.path.join(directory, 'density_info.pkl'))
    high_traffic_hours = joblib.load(os.path.join(directory, 'high_traffic_hours.pkl'))

    df_test = Create_location_clusters(df_test, train=0, kmeans=kmeans_model)
    df_test = Create_average_durations_by_clusters(df_test, train=0, avg_duration=avg_duration_info)
    df_test = Create_cluster_density_features(df_test, train=0, counts=density_info)
    df_test = Create_time_features(df_test, train=0, high_traffic_hours=high_traffic_hours)
    df_test=Create_direction_bucket_feature(df_test)
    df_test=Create_interaction_features(df_test)
    

    return df_test 

if __name__ == "__main__":

    train_path = "Data/Trip_train.csv" # "split/train.csv"
    val_path   = "Data/Trip_val.csv"   # "split/val.csv"
    test_path  = "Data/test (1).csv"  # "split/test.csv"
    model_path = "Ridge_2025_05_20_11_42_With_Train_R2_0.71_And_Test_0.71.pkl"

    LogFeatures = FunctionTransformer(log_transform, feature_names_out=with_suffix)
    modeling_pipeline = load_model(model_path)

    data_preprocessor = modeling_pipeline['data_preprocessor']
    training_features = modeling_pipeline['selected_feature_names']
    model = modeling_pipeline['model']
    
    # Apply model to try train dataset

    # df_train = pd.read_csv(train_path)
    # df_train = prepare_data(df_train)
    # df_train_processed = data_preprocessor.transform(df_train[training_features])
    # rmse, r2, _  = predict_eval(model, df_train_processed, df_train[target] ,'train')
    
    # # Apply model to try val dataset
    # df_val = pd.read_csv(val_path)
    # df_val = prepare_data(df_val)
    # df_vail_processed = data_preprocessor.transform(df_val[training_features])
    # rmse, r2, _  = predict_eval(model, df_vail_processed, df_val[target] ,'vail')

    # Apply model to try test dataset

    df_test = pd.read_csv(test_path)
    df_test = prepare_data(df_test)
    df_test_processed = data_preprocessor.transform(df_test[training_features])
    rmse, r2, _  = predict_eval(model, df_test_processed, df_test[target] ,'test')