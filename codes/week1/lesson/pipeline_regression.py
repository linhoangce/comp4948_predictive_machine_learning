import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "sqft": [800, 900, 1000, 1100, np.nan, 1300, 1400],
    "age": [20, 15, np.nan, 5, 30, 10, 8],
    "city": ["A", "B", "A", "C", "B", "A", "C"],
    "property_type": ["Condo","Townhouse","Detached","Detached",np.nan,"Condo","Detached"],
    "price": [200000, 220000, 250000, 270000, 230000, 300000, 320000]
}
df = pd.DataFrame(data)

numeric_features = ['sqft', 'age']
categorical_features = ['city', 'property_type']
target = 'price'

X = df[numeric_features + categorical_features]
y = df[target]

numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=3)),
    ('scaler', StandardScaler())
])

categorical_transformer = OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False,
    drop='first'
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

cv = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_validate(
    pipeline,
    X, y,
    cv=cv,
    scoring={
        'rmse': 'neg_root_mean_squared_error',
        'r2': 'r2'
    },
    return_train_score=False
)

rmse_per_fold = -scores['test_rmse']
r2_per_fold = scores['test_r2']

print(f'RMSE per fold: {rmse_per_fold}')
print(f'RMSE mean: {np.mean(rmse_per_fold)}')
print(f'RMSE std: {rmse_per_fold.std()}')
print(f'R2 per fold: {r2_per_fold}')
print(f'R2 mean: {r2_per_fold.mean()}')

pipeline.fit(X, y)

joblib.dump(pipeline, 'regression_pipeline.joblib')
print('\nSaved regression_pipeline.joblib')

preprocessor = pipeline.named_steps['preprocessor']

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

feature_names = preprocessor.get_feature_names_out()

print('*** Final feature list:')
for name in feature_names:
    print(name)

X_prepared = preprocessor.transform(X.head(3))
X_prepared_df = pd.DataFrame(
    X_prepared, columns=feature_names
)
print(X_prepared_df)

loaded_pipeline = joblib.load('regression_pipeline.joblib')

new_data = pd.DataFrame({
    'sqft': [1200],
    'age': [12],
    'city': ['B'],
    'property_type': ['Condo']
})

predicted_price = loaded_pipeline.predict(new_data)
print(f'Predicted price: {predicted_price}')