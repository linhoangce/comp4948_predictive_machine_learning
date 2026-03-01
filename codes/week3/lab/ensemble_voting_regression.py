import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor


def main():
    N_RUNS = 20
    TEST_SIZE = 0.25

    data = fetch_california_housing()
    X, y = data.data, data.target

    ridge = Ridge(alpha=1.0)
    knn = KNeighborsRegressor(n_neighbors=15)
    rf = RandomForestRegressor(n_estimators=300)
    grad_boost = GradientBoostingRegressor(n_estimators=300)
    vote = VotingRegressor(estimators=[('ridge', ridge),
                                       ('knn', knn),
                                       ('rf', rf),
                                       ('gb', grad_boost)],
                           weights=[0.2, 0.2, 0.2, 0.4])
    models = {
        'Ridge': ridge,
        'KNN': knn,
        'RandomForest': rf,
        'GradientBoosting': grad_boost,
        'VotingRegressor': vote
    }
    rmse_store = {name:[] for name in models}

    for run in range(N_RUNS):
        print(f'Run: {run}')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            rmse_store[name].append(rmse)

    print('\nRMSE')
    for name, rmses in rmse_store.items():
        print(f'{name:15s} mean RMSE = {np.mean(rmses):.4f} | std = {np.std(rmses):.4f}')


if __name__ == '__main__':
    main()