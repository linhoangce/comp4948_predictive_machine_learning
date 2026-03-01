import shap
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def show_shap_plot(kind, shap_values, *, figsize=(12, 7), left=0.35, max_display=15):
    if kind == 'bar':
        shap.plots.bar(shap_values, max_display=max_display, show=False)
    elif kind == 'beeswarm':
        shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    elif kind == 'waterfall':
        shap.plots.waterfall(shap_values, max_display=max_display, show=False)
    else:
        raise ValueError('kind be "bar", "beeswarm", or "waterfall"')

    fig = plt.gcf()
    fig.set_size_inches(*figsize)
    plt.subplots_adjust(left=left)
    plt.tight_layout()
    plt.show()

def main():
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9
    )
    model.fit(X_train, y_train)

    # SHAP explainer
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    show_shap_plot('bar', shap_values, figsize=(10, 6), left=0.4, max_display=20)
    show_shap_plot('beeswarm', shap_values, figsize=(12, 7), left=0.4, max_display=20)
    show_shap_plot('waterfall', shap_values[0], figsize=(14, 6), left=0.45, max_display=14)


if __name__ == '__main__':
    main()