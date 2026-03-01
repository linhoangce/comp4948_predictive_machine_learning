import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

def preprocess_data(path):
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, 0:13].values
    y = df.iloc[:, 13].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1, 1))
    y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1, 1))

    return X_train_scaled, X_test_scaled, \
            y_train_scaled.ravel(), y_test_scaled.ravel(), y_test, scaler_y

def build_and_evaluate_model(X_train, X_test,
                             y_train, y_test, scaler_y):
    model = MLPRegressor(
        hidden_layer_sizes=(150, 100, 50),
        max_iter=1000,
        activation='relu',
        solver='adam',
        verbose=1
    )
    model.fit(X_train, y_train)

    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(np.array(y_pred_scaled).reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'RMSE: {rmse:.4f}')

    return model

def draw_plot(model):
    plt.plot(model.loss_curve_)
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()

def random_search(model, X_train, X_test, y_train, y_test, scaler_y):
    param_grid = {
        'hidden_layer_sizes': [(150, 100, 50), (120, 80, 40), (128, 64, 64)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.001, 0.05, 0.1, 0.0001],
        'learning_rate': ['constant', 'adaptive']
    }

    grid = RandomizedSearchCV(model,
                              param_distributions=param_grid,
                              n_jobs=-1, cv=10,
                              scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    print('Best Parameters')
    print(grid.best_params_)

    print('\nEvaluate RandomizedSearchCV Model')
    y_pred_scaled = grid.predict(X_test)
    y_pred = scaler_y.inverse_transform(np.array(y_pred_scaled).reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'RMSE: {rmse:.4f}')
    return grid.best_estimator_

def main():
    PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\housing.data"

    X_train, X_test, y_train, y_test, y_test_orig, scaler_y = \
        preprocess_data(PATH)
    model = build_and_evaluate_model(X_train, X_test, y_train,
                             y_test_orig, scaler_y)
    draw_plot(model)

    print('='*100)
    print('GRID SEARCHING...............')
    gs_model = random_search(model, X_train, X_test, y_train,
                           y_test_orig, scaler_y)
    draw_plot(gs_model)


if __name__ == '__main__':
    main()