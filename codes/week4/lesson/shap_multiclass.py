import shap
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from SHAP_regression import show_shap_plot


def main():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    rf = RandomForestClassifier(n_estimators=100, max_depth=2)
    rf.fit(X_train, y_train)

    explainer = shap.Explainer(rf, X_train, model_output='probability')
    shap_values = explainer(X_test, check_additivity=False)
    shap_values.feature_names = list(X.columns)

    class_names = iris.target_names
    k = 0
    show_shap_plot('bar', shap_values[..., k],
                   figsize=(10, 6), left=0.4, max_display=20)
    show_shap_plot('beeswarm', shap_values[..., k],
                   figsize=(12, 7), left=0.4, max_display=20)
    show_shap_plot('waterfall', shap_values[0, ..., k],
                   figsize=(14, 6), left=0.45, max_display=14)


if __name__ == '__main__':
    main()