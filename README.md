# [NYC-Trip-Duration-Prediction]() 
## Project Overview
This project focuses on building a machine learning pipeline to predict the total duration of taxi rides in New York City, based on the publicly available dataset from the NYC Taxi Duration Prediction competition on Kaggle and includes pickup/dropoff timestamps, coordinates, passenger count, and more.

## Usage
```shell
cd  NYC-Trip-Duration-Prediction
python3 test.py
```
## Repo structure and File descriptions
```
NYC-Trip-Duration-Prediction/
├── README.md
├── requirements.txt
├── (EDA) NYC-Trip_Duration.ipynb
├── (Report) NYC-Trip_Duration.pdf
├── test.py
├── train.py
├── Help.py
├── Prepare_Data.py
├── history.json
├── Best_model_data_overview.json
├── Process_data/   
│   └── trained_assets
│       ├── avg_duration_info.pkl
│       ├── density_info.pkl
│       ├── high_traffic_hours.pkl
│       ├── kmeans_model.pkl
│   ├── train.csv.zip
│   ├── val.csv.zip
│   └── data_overview.json
└── Data
    ├── Trip_train.csv.zip
    ├── test (1).csv.zip
    └── Trip_val.csv.zip
```
- `README.md`: Contains information about the project, how to set it up, and any other relevant details.
- `requirements.txt`: Lists all the dependencies required for the project to run.
- `(EDA) NYC-Trip_Duration.ipynb`: Jupyter notebook containing exploratory data analysis on New York City taxi trip duration.
- `test.py`: Python script for loading test data.
- `train.py`: Python script for training the model (main script).
- `Help.py`: Python script containing helper functions used in other scripts.
- `Prepare_Data.py`: Python script for data preparation (Data Versioning).
- `history.json`: JSON file containing the training history of trained models.
- `Best_model_data_overview.json`: JSON file containing overview about the baseline model.
- `(Report) NYC-Trip_Duration.pdf`: Report summarizes the project.
- `Process_data/trained_assets`: Stores pre-trained assets used for feature engineering on new datasets.
  - `avg_duration_info.pkl`: Average trip durations per pickup and dropoff cluster, plus global average.
  - `density_info.pkl`: Trip counts per pickup and dropoff cluster (used for density features).
  - `high_traffic_hours.pkl`: List of high traffic hours based on trip frequency during training.  
  - `kmeans_model.pkl`: Trained KMeans model for location clustering of pickup and dropoff points.
- `processed_data`: containing processed data.
  - `train.csv.zip`: Training data in zip format.
  - `val.csv.zip`: Validation data in zip format.
  - `data_overview.json`: Overview related to this processed data.
- `Data`: Directory containing data for training/testing.
  - `Trip_train.csv.zip`: Training data in zip format.
  - `test (1).csv.zip`: Test data in zip format.
  - `Trip_val.csv.zip`: Validation data in zip format.
- - `Ridge_2025_05_20_11_42_With_Train_R2_0.71_And_Test_0.71.pkl`: The Best model stored as pkl format.
## Notes

All the data files are compressed due to GitHub's limitations on pushing larger data files. Please be careful and use appropriate software to decompress and unzip the data files as needed.

## Data exploration

### Target Variable: Trip Duration
- The `trip_duration` variable exhibits a highly skewed distribution with a long tail of very large values.
- Most trips are between 150 seconds and 1000 seconds (about 2.5 to 16.7 minutes)
- Log transformation applied to visualize better and help with modeling large values

### Feature Engineering
1. Discrete Numerical Features:
   - Vendor ID and passenger count analyzed
   - No significant difference in trip duration among vendors
   - Trips with 7-8 passengers tend to have shorter durations, possibly due to trip purpose

2. Geospatial Distance and Direction:
   - The `Haversine-distance` Measures the great-circle distance between pickup and dropoff points.

   - The `distance_dummy_manhattan` An approximation of the Manhattan distance by summing the north-south and east-west components using haversine

   - Bearing `direction` Indicates the compass direction (in degrees) from the pickup point to the dropoff point. 

   - (`delta_latitude`, `delta_longitude`) The absolute differences between pickup and dropoff coordinates. These capture the magnitude of displacement.

   - (`midpoint_latitude`, `midpoint_longitude`) Represent the center point of each trip. Useful for spatial clustering or heatmaps.

   - The `euclidean_distance` The straight-line (as-the-crow-flies) distance in degrees.

   - These features provide the model with rich spatial information, which is essential for estimating trip duration more accurately.

   - Most trips range from less than 1 km to 25 km

   - Speed of trips calculated using distance and duration

3. Spatial Clustering of Pickup and Dropoff Locations:
   - **`pickup_cluster`**: The cluster label of the pickup location.
   - **`dropoff_cluster`**: The cluster label of the dropoff location.
   - These features introduce regional context and allow the model to learn patterns related to specific zones or travel routes, improving its ability to generalize     over space.

4. Other Features:
   - `high_traffic_hour` to indicate whether a trip occurred during hours with high trip volume.
   - `trip_type` to indicate whether the pickup and dropoff locations belong to the same cluster.
   - `pickup_cluster_hour` by combining the pickup location cluster and the hour of the day.
   - `distance_to_center` helps capture whether a trip occurred in central urban areas, which typically have different traffic dynamics than peripheral zones.
   - `day_part` to represent the general time of day based on the hour.
   - `distance_hour_interaction` by multiplying the Haversine distance of each trip by the hour of day.
   - And Others
5. Temporal Analysis:
   - Longer trip durations observed during summer months
   - Weekend trips generally longer than weekdays
   - Shorter durations during morning and evening rush hours

### Correlation Analysis
- Strong positive correlation between trip duration and distance
- Negative correlation between trip duration and speed

## Modeling

### Data Pipeline
1. Feature splitting into categorical and numerical
2. One-hot encoding for categorical features
3. Standard scaling for numerical features
4. Polynomial Features (degree=6)
5. Log transformation applied

### Results
- RMSE: 0.4326 (Validation)
- R²: 0.7075 (Validation)

### Lessons and Future Work
- Feature selection improves model performance
- Outlier removal for intra-trip duration doesn't improve performance
- Estimating speed as a separate feature didn't significantly improve results
- Consider exploring more complex algorithms like XGBoost and ensemble methods for better performance


