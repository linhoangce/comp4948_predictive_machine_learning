import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor


def show_model_coefficients(model, columns):
    coef_df = pd.Series(
        model.coef_,
        index=columns
    ).sort_values()
    print(coef_df)
    print(f'\nTotal remaining coefficients: {coef_df.sum()}')
    return coef_df

def show_rmse(model, title, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'Test RMSE ({title}): {rmse:.4f}')
    return rmse

def fit_and_evaluate_ridge(model, title, X_train, X_test, y_train, y_test, columns):
    model.fit(X_train, y_train)
    show_rmse(model, title, X_test, y_test)
    coefs = show_model_coefficients(model, columns)
    return model, coefs

def get_data():
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    return (X, y, X_train, X_test, X_train_scaled, X_test_scaled,
            y_train, y_test)

def format_training_data_for_OLS(features, X_train, indices):
    X_train_df = pd.DataFrame(
        X_train, columns=features, index=indices
    )
    X_train_df = sm.add_constant(X_train_df)
    return X_train_df

def compare_coefficients(coefs_l2_0, coefs_l2_240):
    df_coefs = pd.concat([coefs_l2_0, coefs_l2_240], axis=1)
    df_coefs.columns = ['L2=0', 'L2=240']
    df_coefs['% reduction'] = abs(df_coefs['L2=0'] / df_coefs['L2=240'])
    df_coefs = df_coefs.sort_values('% reduction', ascending=False)
    print(df_coefs)

def show_VIF(X):
    vif_df = pd.DataFrame()
    vif_df['Feature'] = X.columns
    vif_df['VIF'] = [variance_inflation_factor(X.values, i)
                     for i in range(X.shape[1])]
    print('\nVariance Inflation Factors')
    vif_df = vif_df.sort_values(by='VIF', ascending=False)
    print(vif_df)

def show_mean_coef_std(coef_list, title):
    features = len(coef_list[0])
    feature_vars = []

    for idx in range(features):
        coef_vals = []
        for row in coef_list:
            coef_vals.append(row[idx])
        feature_vars.append(np.std(coef_vals))
    mean_coef_variance = np.mean(feature_vars)
    print(f'{title}: {mean_coef_variance}')

def check_variance_across_splits(X, y, n_splits=50, test_size=0.2, alpha=(0, 240)):
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size)

    rmse_ols, rmse_r0, rmse_r240 = [], [], []
    coef_ols, coef_r0, coef_r240 = [], [], []

    for train_idx, test_idx in ss.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_df = format_training_data_for_OLS(X.columns,
                                                  X_train_scaled,
                                                  X_train.index)
        X_test_df = format_training_data_for_OLS(X.columns,
                                                 X_test_scaled,
                                                 X_test.index)

        ols = sm.OLS(y_train, X_train_df).fit()
        y_pred_ols = ols.predict(X_test_df)
        rmse_ols.append(np.sqrt(mean_squared_error(y_test, y_pred_ols)))
        coef_ols.append(ols.params.drop('const').values)

        # Ridge alpha=0
        ridge_0 = Ridge(alpha=0, max_iter=5000)
        ridge_0.fit(X_train_scaled, y_train)
        y_pred_r0 = ridge_0.predict(X_test_scaled)
        rmse_r0.append(np.sqrt(mean_squared_error(y_test, y_pred_r0)))
        coef_r0.append(ridge_0.coef_)

        # Ridge alpha=240
        ridge_240 = Ridge(alpha=240, max_iter=5000)
        ridge_240.fit(X_train_scaled, y_train)
        y_pred_r240 = ridge_240.predict(X_test_scaled)
        rmse_r240.append(np.sqrt(mean_squared_error(y_test, y_pred_r240)))
        coef_r240.append(ridge_240.coef_)

    print('='*50)
    print('*** RSMSE STD')
    print(f'OLS: {np.std(rmse_ols)}')
    print(f'Ridge a=0: {np.std(rmse_r0)}')
    print(f'Ridge a=240: {np.std(rmse_r240)}')

    print('*** RMSE Mean')
    print(f'OLS: {np.mean(rmse_ols)}')
    print(f'Ridge a=0: {np.mean(rmse_r0)}')
    print(f'Ridge a=240: {np.mean(rmse_r240)}')

    print('\n*** Mean Coefs SD')
    show_mean_coef_std(coef_ols, 'OLS')
    show_mean_coef_std(coef_r0, 'Ridge a=0')
    show_mean_coef_std(coef_r240, 'Ridge a=240')

def main():
    (X, y, X_train, X_test, X_train_scaled, X_test_scaled,
    y_train, y_test) = get_data()

    X_train_df = format_training_data_for_OLS(X.columns, X_train_scaled, X_train.index)
    X_test_df = format_training_data_for_OLS(X.columns, X_test_scaled, X_test.index)

    ols_model = sm.OLS(y_train, X_train_df).fit()
    print(ols_model.summary())

    show_rmse(ols_model, 'OLS', X_test_df, y_test)

    # mix_iter limits how long optimizer runs
    ridge_L2__0 = Ridge(alpha=0, max_iter=5000)
    model, coefs_L2__0 = fit_and_evaluate_ridge(ridge_L2__0, 'Ridge L2=0',
                                                X_train_scaled, X_test_scaled, y_train, y_test, X.columns)

    ridge_L2__240 = Ridge(alpha=240, max_iter=5000)
    mode, coefs_L2__240 = fit_and_evaluate_ridge(ridge_L2__240, 'Ridge L2=240',
                                           X_train_scaled, X_test_scaled, y_train, y_test, X.columns)
    compare_coefficients(coefs_L2__0, coefs_L2__240)
    show_VIF(X_train_df)

    for _ in range(5):
        check_variance_across_splits(X, y, n_splits=50, alpha=(0, 240))

if __name__ == '__main__':
    main()