import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

def evaluate(model, X_test, y_test, title, n_estimators, max_features):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    stats = {'type': title,
             'rmse': rmse,
             'estimators': n_estimators,
             'features': max_features}
    return rmse, stats

def main():
    PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\petrol_consumption.csv"
    ESTIMATORS_LIST = [800, 1000, 1200, 1400]
    MAX_FEATURES = [3, 4]

    df = pd.read_csv(PATH)
    X = df.drop('Petrol_Consumption', axis=1)
    y = df['Petrol_Consumption']

    single_rmse = []
    stats_ens = []
    stats_knn = []

    for estimators in ESTIMATORS_LIST:
        for max_features in MAX_FEATURES:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            ensemble_model = BaggingRegressor(
                estimator=KNeighborsRegressor(),
                max_features=max_features,
                max_samples=0.5,
                n_estimators=estimators
            )
            ensemble_model.fit(X_train_scaled, y_train.values.ravel())
            _, stats_en = evaluate(ensemble_model, X_test_scaled, y_test, 'Ensemble',
                     estimators, max_features)
            stats_ens.append(stats_en)

            # single model
            knn = KNeighborsRegressor(n_neighbors=5)
            knn.fit(X_train_scaled, y_train)
            rmse, stats = evaluate(knn, X_test_scaled, y_test, 'Single',
                            None, None)
            single_rmse.append(rmse)
            stats_knn.append(stats)

    df_stats_ens = pd.DataFrame(stats_ens)
    df_stats_knn = pd.DataFrame(stats_knn)
    df_stats_ens = df_stats_ens.sort_values(by=['type', 'rmse'])
    df_stats_knn = df_stats_knn.sort_values(by=['type', 'rmse'])

    print(df_stats_ens)
    print()
    print(df_stats_knn)
    print(f'Avg Single RMSE: {np.mean(single_rmse):.4f}')


if __name__ == '__main__':
    main()
