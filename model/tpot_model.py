import numpy as np
import pandas as pd
import joblib as jl
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
import os

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('model/training_data.csv', sep=',')

# Clearing erroneous temp values
tpot_data = tpot_data[tpot_data.temp > .6]
print(tpot_data.temp.describe())

features = tpot_data.drop('traffic_volume', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['traffic_volume'], random_state=None)

exported_pipeline = make_pipeline(
    MinMaxScaler(),
    KNeighborsRegressor(n_neighbors=69, p=2, weights="distance")
)

exported_pipeline.fit(training_features, training_target)
jl.dump(exported_pipeline, 'model.pkl')