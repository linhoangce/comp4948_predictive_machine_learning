import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

plt.style.use('ggplot')

def preprocess_data(path):
    df = pd.read_csv(path)
    df.iris_type = pd.Categorical(df.iris_type)
    df['flowertype'] = df.iris_type.cat.codes
    X = df.drop(columns=['flowertype'])
    y = df['flowertype']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def build_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = MLPClassifier(max_iter=5000)
    model.fit(X_train, y_train)
    print(model.get_params())

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

    return model

def draw_plots(model):
    plt.plot(model.loss_curve_)
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()

def grid_search(model, X_train, X_test, y_train, y_test):
    param_grid = {
        'solver': ['adam', 'sgd'],
        'learning_rate': ['constant', 'adaptive', 'invscaling'],
        'hidden_layer_sizes': [(200, 200), (300, 200), (150, 150)],
        'activation': ['logistic', 'relu', 'tanh']
    }

    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        scoring='accuracy',
                        n_jobs=-1,
                        cv=5, verbose=1,
                        return_train_score=False)
    grid.fit(X_train, y_train)
    print('Best Parameters')
    print(grid.best_params_)

    y_pred = grid.predict(X_test)
    print('='*60 + '\n')
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    draw_plots(grid.best_estimator_)


def main():
    PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\iris_numeric.csv"
    X_train, X_test, y_train, y_test = preprocess_data(PATH)
    model = build_and_evaluate_model(X_train, X_test, y_train, y_test)
    draw_plots(model)

    print('*'*100)
    print('='*15 + 'GRID SEARCH' + '='*15)
    print('*'*100)
    grid_search(model, X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()