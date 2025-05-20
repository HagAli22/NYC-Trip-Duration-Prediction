import joblib
import datetime
import numpy as np
import pandas as pd
from Help import update_baseline_metadata
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LassoCV
from sklearn.metrics import r2_score, root_mean_squared_error


# Global variables
seed = 31  # random seed
degree =4   # degree of polynomial features 
do_feature_selection = False # do feature selection using Lasso
np.random.seed(seed)

target = 'trip_duration'


# Feature of target feature
numeric_features = ["pickup_latitude","dropoff_longitude", "dropoff_latitude",'hour_cos','hour_sin',
                     'distance_dummy_manhattan', 'distance_haversine',
                    'delta_longitude','delta_latitude', 'direction'
                   ]

categorical_features = ['trip_distance_category', 'avg_duration_by_dropoff_cluster', 'avg_duration_by_pickup_cluster',
                        'direction_bucket', 'season', 'trip_type', 'store_and_fwd_flag', 'vendor_id', 'passenger_count',
                        'hour','dayofmonth', 'dayofweek', 'month'
                        ]



def predict_eval(model, data_preprocessed, target, name) -> str:
    y_train_pred = model.predict(data_preprocessed)
    rmse = root_mean_squared_error(target, y_train_pred)
    r2 = r2_score(target, y_train_pred)
    print(f"{name} RMSE = {rmse:.2f} - R2 = {r2:.2f}")
    return rmse, r2, f"{name} RMSE = {rmse:.2f} - R2 = {r2:.2f}"

def data_preprocessing_pipeline(categorical_features=[], numeric_features=[]):
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('ohe', OneHotEncoder(handle_unknown="infrequent_if_exist"))
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    data_preprocessor = Pipeline(steps=[
        ('preprocessor', column_transformer)
    ])

    return data_preprocessor

# Apply log1p transformation after ensuring values are non-negative (clip negatives to 0)
def log_transform(x): 
    return np.log1p(np.maximum(x, 0))


def with_suffix(_, names: list[str]): 
    return [name + '__log' for name in names]


def pipeline(train, test, do_feature_selection=True):

    #apply log_transform and renames features with '__log' suffix
    LogFeatures = FunctionTransformer(log_transform, feature_names_out=with_suffix)
    train_features = numeric_features + categorical_features

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()), 
        ('poly', PolynomialFeatures(degree=degree)),
        ('log', LogFeatures),
    ])

    categorical_transformer = Pipeline(steps=[
        ('ohe', OneHotEncoder(handle_unknown="infrequent_if_exist"))
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    data_preprocessor = Pipeline(steps=[
        ('preprocessor', column_transformer)
    ])

   
    train_preprocessed = data_preprocessor.fit_transform(train[train_features])
    test_preprocessed = data_preprocessor.transform(test[train_features])

    if do_feature_selection:
        print("Doing Features Selection : ")
        lasso_cv = LassoCV(cv=4, max_iter=1000, random_state=seed)
        lasso_cv.fit(train_preprocessed, train[target])
        selected_feature_indices = [i for i, coef in enumerate(lasso_cv.coef_) if coef != 0]
        
        all_feature_names = data_preprocessor.named_steps['preprocessor'].get_feature_names_out()
        selected_feature_names = all_feature_names[selected_feature_indices]

        print("LassoCV selected features: ", selected_feature_names)

        train_preprocessed_lasso = train_preprocessed[:, selected_feature_indices]
        test_preprocessed_lasso = test_preprocessed[:, selected_feature_indices]
    else:
        train_preprocessed_lasso = train_preprocessed
        test_preprocessed_lasso = test_preprocessed
        selected_feature_names = train_features

    
    ridge = Ridge(alpha=1, random_state=seed)
    ridge.fit(train_preprocessed_lasso, train[target])

    train_rmse, train_r2, _ = predict_eval(ridge, train_preprocessed_lasso, train[target], "train")
    test_rmse, test_r2, _ = predict_eval(ridge, test_preprocessed_lasso, test[target], "val")

    return ridge, selected_feature_names, data_preprocessor, train_rmse, train_r2, test_rmse, test_r2


if __name__ == "__main__":
    data_version = 0
    # NOTE: if did not unzip the data you can replace the .csv with .zip should be work
    data_path =  f"Process_data"
    train_path = f"{data_path}/train.csv" # f"{data_path}/train.zip"
    val_path =   f"{data_path}/val.csv"   # f"{data_path}/val.zip"


    df_train = pd.read_csv(train_path)
    df_val =   pd.read_csv(val_path)
   

    model, selected_feature_names, data_preprocessor, train_rmse, train_r2, test_rmse, test_r2 = pipeline(df_train, df_val, do_feature_selection)

    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    filename = f'Ridge_{now}_With_Train_R2_{train_r2:.2f}_And_Test_{test_r2:.2f}.pkl'

    if do_feature_selection:
        selected_feature_names = selected_feature_names.tolist()
    
    model_data = {
        'model':model,
        'data_path': data_path,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'selected_feature_names': selected_feature_names,
        'data_preprocessor': data_preprocessor,
        'data_version': data_version,
        'random_seed': seed
    }
    
    joblib.dump(model_data, filename)
    print(f"Model saved as {filename}")
    update_baseline_metadata(model_data)