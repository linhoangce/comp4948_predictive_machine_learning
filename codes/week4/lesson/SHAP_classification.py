import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

from SHAP_regression import show_shap_plot

def main():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = SGDClassifier(loss='log_loss',
                        max_iter=2000,
                        tol=1e-3,
                        random_state=42)
    clf.fit(X_train_scaled, y_train)

    explainer = shap.LinearExplainer(clf, X_train_scaled)
    shap_values = explainer(X_test_scaled)
    shap_values.feature_names = list(X.columns)

    show_shap_plot('bar', shap_values, figsize=(10,6), left=0.4, max_display=20)
    show_shap_plot('beeswarm', shap_values, figsize=(12, 7), left=0.4, max_display=20)
    show_shap_plot('waterfall', shap_values[0], figsize=(14, 6), left=0.45, max_display=14)


if __name__ == '__main__':
    main()