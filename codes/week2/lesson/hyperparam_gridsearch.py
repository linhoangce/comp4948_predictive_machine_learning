from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv(r"C:\Users\linho\Desktop\CST\term4\pa\data\temperatures.csv")

df = pd.get_dummies(df)

X = df.drop(columns=['actual'], axis=1)
y = df['actual']

features = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(f'MAE: {mean_absolute_error(y_test, y_pred):.4f}')

mape = 100 * (abs(y_pred - y_test) / y_test)
accuracy = 100 - np.mean(mape)

print(f'Accuracy: {accuracy}')

print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}')

importances = rf.feature_importances_

feature_imp = pd.DataFrame([{'importance': importances[i],
                             'feature': features[i]}
                            for i in range(len(importances))])
feature_imp = feature_imp.sort_values(by=['importance'], ascending=False)
print(feature_imp)

random_grid = {
    'bootstrap': [True],
    'max_depth': [4, 6, None],
    'max_features': [None],
    'min_samples_leaf': [15],
    'min_samples_split': [15],
    'n_estimators': [400, 800, 1600]
}

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator=rf,
                               param_distributions=random_grid,
                               n_iter=100,
                               cv=10,
                               verbose=2, n_jobs=-1)
rf_random.fit(X_train, y_train)

print('\nBest Parameters')
print(rf_random.best_params_)

rf_grid = RandomForestRegressor(n_estimators=800,
                                min_samples_split=15,
                                min_samples_leaf=15,
                                max_features=None,
                                max_depth=4,
                                bootstrap=True)
rf_grid.fit(X_train, y_train)
y_pred = rf_grid.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f'MAE: {mae:.4f}')

mape = 100 * (abs(y_pred - y_test) / y_test)
acc = 100 - np.mean(mape)

print(f'Accuracy: {acc:.4f}')

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse:.4f}')