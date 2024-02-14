from datetime import datetime, timedelta
import os
os.chdir('..')
import hopsworks
from hsfs.feature_store import FeatureStore as fs
import pandas as pd
import numpy as np
import hsfs
import sys
import src.data
sys.path.append('/Users/yashwantsaikoneru/taxi_demand_predictor')

import src.config as config

#from src.config import FEATURE_VIEW_METADATA

def get_hopsworks_project() -> hopsworks.project.Project:

    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )

def get_feature_store()->fs:

    project=get_hopsworks_project()
    return project.get_feature_store()

def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """"""
    # past_rides_columns = [c for c in features.columns if c.startswith('rides_')]
    predictions = model.predict(features)
    results = pd.DataFrame()
    results['pickup_location_id'] = features['pickup_location_id'].values
    results['predicted_demand'] = predictions.round(0)
    
    return results


def load_batch_of_features_from_store(
    current_date: pd.Timestamp,    
) -> pd.DataFrame:
    """Fetches the batch of features used by the ML system at `current_date`

    Args:
        current_date (datetime): datetime of the prediction for which we want
        to get the batch of features

    Returns:
        pd.DataFrame: 4 columns:
            - `pickup_hour`
            - `rides`
            - `pickup_location_id`
            - `pickpu_ts`
    """
    n_features = config.N_FEATURES
    project=get_hopsworks_project()
    fs=project.get_feature_store()
    feature_view = fs.get_feature_view('time_series_hourly_feature_view', version=1)

    # fetch data from the feature store
    fetch_data_from = current_date - timedelta(days=28)
    fetch_data_to = current_date - timedelta(hours=1)

    # add plus minus margin to make sure we do not drop any observation
    ts_data = feature_view.get_batch_data(
        start_time=fetch_data_from - timedelta(days=1),
        end_time=fetch_data_to + timedelta(days=1)
    )
    
    # filter data to the time period we are interested in
    pickup_ts_from = int(fetch_data_from.timestamp() * 1000)
    pickup_ts_to = int(fetch_data_to.timestamp() * 1000)
    ts_data = ts_data[ts_data.pickup_ts.between(pickup_ts_from, pickup_ts_to)]

    # sort data by location and time
    ts_data.sort_values(by=['pickup_location_id', 'pickup_hour'], inplace=True)
    
    # validate we are not missing data in the feature store
    selected_location_id = 1

# Filter the data for the selected pickup location ID
    selected_location_data = ts_data[ts_data['pickup_location_id'] == selected_location_id]

# Find the earliest (minimum) and latest (maximum) 'pickup_hour' for the selected location
    earliest_date = selected_location_data['pickup_hour'].min()
    latest_date = selected_location_data['pickup_hour'].max()

# Print the earliest and latest dates
    print(f"Earliest date for location ID {selected_location_id}: {earliest_date}")
    print(f"Latest date for location ID {selected_location_id}: {latest_date}")
    location_ids = ts_data['pickup_location_id'].unique()
    assert len(ts_data) == config.N_FEATURES * len(location_ids), \
        "Time-series data is not complete. Make sure your feature pipeline is up and runnning."

    # transpose time-series data as a feature vector, for each `pickup_location_id`
    x = np.ndarray(shape=(len(location_ids), n_features), dtype=np.float32)
    for i, location_id in enumerate(location_ids):
        ts_data_i = ts_data.loc[ts_data.pickup_location_id == location_id, :]
        ts_data_i = ts_data_i.sort_values(by=['pickup_hour'])
        x[i, :] = ts_data_i['rides'].values

    # numpy arrays to Pandas dataframes
    features = pd.DataFrame(
        x,
        columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(n_features))]
    )
    features['pickup_hour'] = current_date
    features['pickup_location_id'] = location_ids
    features.sort_values(by=['pickup_location_id'], inplace=True)

    return features
    

def load_model_from_registry():
    
    import joblib
    from pathlib import Path

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    model = model_registry.get_model(
        name=config.MODEL_NAME,
        version=config.MODEL_VERSION,
    )  
    
    model_dir = model.download()
    model = joblib.load(Path(model_dir)  / 'model.pkl')
       
    return model

def load_predictions_from_store(
    from_pickup_hour: datetime,
    to_pickup_hour: datetime
    ) -> pd.DataFrame:
    """
    Connects to the feature store and retrieves model predictions for all
    `pickup_location_id`s and for the time period from `from_pickup_hour`
    to `to_pickup_hour`

    Args:
        from_pickup_hour (datetime): min datetime (rounded hour) for which we want to get
        predictions

        to_pickup_hour (datetime): max datetime (rounded hour) for which we want to get
        predictions

    Returns:
        pd.DataFrame: 3 columns:
            - `pickup_location_id`
            - `predicted_demand`
            - `pickup_hour`
    """
    
    
    
    # get pointer to the feature view
    project=get_hopsworks_project()
    store=project.get_feature_store()
    feature_group=store.get_feature_group(
        name=config.FEATURE_GROUP_MODEL_PREDICTIONS,version=1
    )
    results=store.get_feature_view(config.FEATURE_VIEW_PREDICTIONS_NAME,version=1)
    
    # get data from the feature view
    print(f'Fetching predictions for `pickup_hours` between {from_pickup_hour}  and {to_pickup_hour}')
    predictions = results.get_batch_data(
        
    )
    
    # make sure datetimes are UTC aware
    predictions['pickup_hour'] = pd.to_datetime(predictions['pickup_hour'], utc=None)
    from_pickup_hour = pd.to_datetime(from_pickup_hour, utc=None)
    to_pickup_hour = pd.to_datetime(to_pickup_hour, utc=None)
    
    # make sure we keep only the range we want
    

    # Specify the datetime you're interested in
    desired_datetime = pd.to_datetime(datetime.now(), utc=True).floor('H')

    # Filter the DataFrame for rows matching the desired datetime
    filtered_df = predictions[predictions['pickup_hour'] == desired_datetime]    
    # sort by `pick_up_hour` and `pickup_location_id`
    predictions.sort_values(by=['pickup_hour', 'pickup_location_id'], inplace=True)
    print(f'From:{from_pickup_hour} To:{to_pickup_hour}')
    predictions=predictions
    return predictions

