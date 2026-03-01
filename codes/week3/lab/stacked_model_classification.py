import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def get_model_list():
    models = list()
    models.append(LogisticRegression())
    models.append(DecisionTreeClassifier())
    models.append(AdaBoostClassifier())
    models.append(KNeighborsClassifier(n_neighbors=5))
    models.append(RandomForestClassifier(n_estimators=10))
    return models

def evaluate(model, y_test, y_pred):
    print('\n***' + model.__class__.__name__)
    report = classification_report(y_test, y_pred)
    print(report)

def fit_base_model(models, X_train, y_train, X_test):
    models_fitted = [model.fit(X_train, y_train) for model in models]
    df_pred = pd.DataFrame({model.__class__.__name__: model.predict(X_test)
                            for model in models_fitted})
    return df_pred, models_fitted

def fit_stacked_model(df_pred, y):
    model = LogisticRegression()
    model.fit(df_pred, y)
    return model

def main():
    PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\Social_Network_Ads.csv"
    IRIS = r"C:\Users\linho\Desktop\CST\term4\pa\data\iris_numeric.csv"

    # df = pd.read_csv(PATH)
    # df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
    # X = df.drop(columns=['User ID', 'Purchased'], axis=1)
    # y = df['Purchased']

    df = pd.read_csv(IRIS)
    X = df.drop('iris_type', axis=1)
    y = df['iris_type']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    model_list = get_model_list()

    df_pred, models = fit_base_model(model_list, X_train_scaled, y_train, X_val_scaled)
    stacked_model = fit_stacked_model(df_pred, y_val)

    print('Pred Input DataFrame')
    print(df_pred)

    print('\n*** Evaluate Base Models')
    df_test_pred = pd.DataFrame()
    for i in range(len(models)):
        y_pred = models[i].predict(X_test_scaled)
        df_test_pred[models[i].__class__.__name__] = y_pred
        evaluate(models[i], y_test, y_pred)

    y_pred_stacked = stacked_model.predict(df_test_pred)
    print('*** Evalue Stacked Model')
    evaluate(stacked_model, y_test, y_pred_stacked)


if __name__ == '__main__':
    main()