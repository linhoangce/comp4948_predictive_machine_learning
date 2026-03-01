import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import mean_squared_error


def show_model_coefs(model, columns):
    coef_df = pd.Series(
        model.coef_,
        index=columns
    ).sort_values()
    print(coef_df)
    print(f'\nTotal remaining coef: {coef_df.sum()}')

def fit_and_evaluate(model, title, X_train, y_train, X_test, y_test, columns):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'\nRMSE ({title}): {rmse}')
    show_model_coefs(model, columns)
    return model

def main():
    PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\winequality.csv"

    df = pd.read_csv(PATH)
    X = df.drop(columns=['quality'], axis=1)
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    lasso_0 = Lasso(alpha=0, max_iter=5000)
    fit_and_evaluate(lasso_0, 'Lasso Alpha 0', X_train_scaled, y_train,
                     X_test_scaled, y_test, X.columns)

    lasso_04 = Lasso(alpha=0.06, max_iter=5000)
    fit_and_evaluate(lasso_04, 'Lasso Alpha 06', X_train_scaled, y_train,
                     X_test_scaled, y_test, X.columns)

    X_train_scaled = sm.add_constant(X_train_scaled)
    X_test_scaled = sm.add_constant(X_test_scaled)

    ols_model = sm.OLS(y_train, X_train_scaled).fit()
    print(ols_model.summary())

    y_pred_ols = ols_model.predict(X_test_scaled)
    rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred_ols))
    print(f'OLS RMSE: {rmse_ols}')

    # compare correlations between variables
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    top_pairs = upper.stack().sort_values(ascending=False).head(10)
    print('\nTop conrrelated feature pairs (|r|):')
    print(top_pairs)


if __name__ == '__main__':
    main()