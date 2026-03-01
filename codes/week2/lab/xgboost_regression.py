import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error

X, y = fetch_california_housing(return_X_y=True)

print(f'y max: {y.max()}')
print(f'y min: {y.min()}')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = xgb.XGBRegressor(
    n_estimators=1200,
    objective='reg:squarederror',
    eval_metric='rmse',
    subsample=0.6,
    reg_lambda=5,
    min_child_weight=10,
    max_depth=4,
    learning_rate=1,
    colsample_bytree=1,
    gamma=0,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f'Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}')

base_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    n_jobs=-1
)

cv = KFold(n_splits=5, shuffle=True)

param_grid = {
    'n_estimator': [400, 800, 1200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.5],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 1, 5],
    'reg_lambda': [1, 5, 10]
}

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=cv,
    n_jobs=-1,
    verbose=2
)

search.fit(X_train, y_train)

print(f'Best RMSE: {search.best_score_}')
print(f'Best params: {search.best_params_}')

best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
print(f'Random Search Best RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}')