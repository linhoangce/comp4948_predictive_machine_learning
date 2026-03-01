import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

path = r"C:\Users\linho\Desktop\CST\term4\pa\data\petrol_consumption.csv"

df = pd.read_csv(path)

X = df.drop('Petrol_Consumption', axis=1)
y = df['Petrol_Consumption']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
print(results)

print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}')
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}')