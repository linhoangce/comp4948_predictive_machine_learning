import shap
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from SHAP_regression import show_shap_plot

def main():
    PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\winequality.csv"

    df = pd.read_csv(PATH)
    X = df.drop(columns=['quality'], axis=1)
    y = df['quality']

    scaler = StandardScaler()
    model = SGDRegressor()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    explainer = shap.Explainer(model, X_train_scaled)
    shap_values = explainer(X_test_scaled)
    shap_values.feature_names = list(X.columns)

    show_shap_plot('bar', shap_values)
    show_shap_plot('beeswarm', shap_values)
    show_shap_plot('waterfall', shap_values[0])


if __name__ == '__main__':
    main()