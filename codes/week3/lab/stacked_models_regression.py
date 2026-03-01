import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def get_model_list():
    models = list()
    models.append(ElasticNet())
    models.append(SVR(gamma='scale'))
    models.append(DecisionTreeRegressor())
    models.append(AdaBoostRegressor())
    models.append(RandomForestRegressor(n_estimators=200))
    models.append(ExtraTreesRegressor(n_estimators=200))
    return models

def evaluate(model, y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'{model.__class__.__name__} RMSE: {rmse:.4f}')

def fit_base_model(X_train, y_train, X_test, models):
    df_pred = pd.DataFrame()
    for i in range(len(models)):
        models[i].fit(X_train, y_train)
        y_pred = models[i].predict(X_test)
        df_pred[f'{models[i].__class__.__name__} Pred'] = y_pred
    return df_pred, models

def fit_stacked_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def main():
        PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\USA_Housing.csv"
        WINE = r"C:\Users\linho\Desktop\CST\term4\pa\data\winequality.csv"

        df = pd.read_csv(WINE)
        X = df.drop('quality', axis=1)
        y = df['quality']

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        unfit_models = get_model_list()

        # fit base and stacked model
        df_pred, models = fit_base_model(X_train_scaled, y_train,
                                          X_val_scaled, unfit_models)
        stacked_model = fit_stacked_model(df_pred.values, y_val)

        print('Input Prediction DataFrame')
        print(df_pred)

        print('\nEvalute Base Model')
        df_val_pred = pd.DataFrame()

        for i in range(len(models)):
            y_pred = models[i].predict(X_test_scaled)
            df_val_pred[i] = y_pred
            evaluate(models[i], y_test, y_pred)

        stacked_pred = stacked_model.predict(df_val_pred)
        print('\n Evaluate Stacked Model')
        evaluate(stacked_model, y_test, stacked_pred)


if __name__ == '__main__':
    main()