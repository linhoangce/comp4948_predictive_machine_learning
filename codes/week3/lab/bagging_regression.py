import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse


def main():
    PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\winequality.csv"
    N_RUNS = 200

    df = pd.read_csv(PATH)

    X = df.drop('quality', axis=1)
    y = df['quality']

    rmse_ensemble, rmse_single = [], []

    for _ in range(N_RUNS):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        ensemble_model = BaggingRegressor(
            estimator=KNeighborsRegressor(n_neighbors=5),
            max_features=4,
            max_samples=0.5,
            n_estimators=100
        ).fit(X_train, y_train)

        rmse_ensemble.append(evaluate(ensemble_model, X_test_scaled, y_test))

        # single model
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X_train, y_train)
        rmse_single.append(evaluate(model, X_test_scaled, y_test))

    print('='*50)
    print('RMSE across many random splits')
    print('='*50)

    print('\n*** Bagged Model')
    print(f'Mean RMSE: {np.mean(rmse_ensemble):.4f}')
    print(f'STD RMSE: {np.std(rmse_ensemble):.4f}')

    print('\n*** Single Model')
    print(f'Mean RMSE: {np.mean(rmse_single):.4f}')
    print(f'STD RMSE: {np.std(rmse_single):.4f}')

    print(f'\nStd ratio (single/bagged): {np.std(rmse_single) / np.std(rmse_ensemble):.4f}')


if __name__ == '__main__':
    main()