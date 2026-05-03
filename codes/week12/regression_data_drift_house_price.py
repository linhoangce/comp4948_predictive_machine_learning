import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from evidently import Report, Dataset, DataDefinition, Regression
from evidently.presets import DataDriftPreset, RegressionPreset

reference = pd.read_csv("data/housing_reference.csv")
current = pd.read_csv("data/housing_current.csv")



