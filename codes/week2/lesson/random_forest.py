from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

path = r"C:\Users\linho\Desktop\CST\term4\pa\data\temperatures.csv"
df = pd.read_csv(path)
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=['actual'], axis=1)
y = df['actual']

feature_list = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f'MAE: {mae:.4f}')

mape = 100 * (abs(y_pred - y_test) / y_test)
acc = 100 - np.mean(mape)

print(f'Accuracy: {acc:.4f}')

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse:.4f}')

def show_feature_importances(importances, features):
    dict = [{'importance': importances[i],
            'feature': features[i]} for i in range(len(importances))]
    df_importance = pd.DataFrame(dict)
    df_importance = df_importance.sort_values(by=['importance'], ascending=False)
    print(df_importance)

importances = list(rf.feature_importances_)
show_feature_importances(importances, feature_list)

# model uses only important features
rf_most_important = RandomForestRegressor(n_estimators=1000, random_state=42)

important_indices = [feature_list.index('temp_1'), feature_list.index('average')]
train_important = X_train.iloc[:, important_indices]
test_important = X_test.iloc[:, important_indices]

rf_most_important.fit(train_important, y_train)

y_pred = rf_most_important.predict(test_important)

print(f'MAE: {mean_absolute_error(y_test, y_pred):.4f}')
mape = np.mean(100 * (abs(y_pred - y_test) / y_test))
accuracy = 100 - mape
print(f'Accuracy: {accuracy:.4f}')

# Grid Search best params
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