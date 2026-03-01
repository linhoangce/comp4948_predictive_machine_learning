import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from skorch import NeuralNetBinaryClassifier
from torch.nn import init


class FluDiagnoser(nn.Module):
    def __init__(self, n_layers, activation, n_inputs, n_outputs,
                 n_neurons,
                 weight_init=init.xavier_uniform_,
                 weight_constraint=3.0):
        super().__init__()
        assert n_layers >= 1

        self.layers = nn.ModuleList()
        self.acts = nn.ModuleList()

        self.layers.append(nn.Linear(n_inputs, n_neurons))
        self.acts.append(activation)

        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(n_neurons, n_neurons))
            self.acts.append(activation)
        self.output = nn.Linear(n_neurons, n_outputs)
        self.prob = nn.Sigmoid()

        self.weight_constraint = weight_constraint
        weight_init(self.layers[0].weight)
        weight_init(self.output.weight)

    def forward(self, x):
        with torch.no_grad():
            norm = self.layers[0].weight.norm(2, dim=0, keepdim=True) \
                        .clamp(min=self.weight_constraint / 2)
            desired = torch.clamp(norm, max=self.weight_constraint)
            self.layers[0].weight *= desired / norm

        for i in range(len(self.layers)):
            x = self.acts[i](self.layers[i](x))

        return self.prob(self.output(x))

def preprocess_data(path):
    df = pd.read_csv(path, sep=',')
    X = df[['A', 'B']]
    y = df[['Diagnosed']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y
    )
    return X, y, X_train, X_test, y_train, y_test

def logistic_model(X_train, X_test, y_train, y_test):
    logistic_model = LogisticRegression(fit_intercept=True,
                                        solver='liblinear',
                                        random_state=0)
    logistic_model.fit(X_train, y_train)
    y_pred = logistic_model.predict(X_test)

    print(f'Model Coeffs:\n{logistic_model.coef_}')
    print(f'Intercept:\n{logistic_model.intercept_}')

    conf_mat = pd.crosstab(np.array(y_test['Diagnosed']),
                           y_pred,
                           rownames=['Actual'],
                           colnames=['Predicted'])
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
    print(conf_mat)

def gridsearch_neural_net(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).reshape(-1)

    param_grid = {
        'optimizer__lr': [0.01, 0.1, 0.5, 1],
        'optimizer__momentum': [0.1, 0.5, 0.9, 0.99],
        # 'module__weight_constraint': [1.0, 2.0, 3.0],
        'module__n_neurons': [8, 16, 32, 64, 128],
        'module__n_layers': [2, 3, 4],
        'module__activation': [nn.ReLU(), nn.Tanh()],
        'module__weight_init': [init.xavier_normal_, init.kaiming_normal_]
    }

    net = NeuralNetBinaryClassifier(FluDiagnoser,
                                    module__n_inputs=X_tensor.shape[1],
                                    module__n_outputs=1,
                                    criterion=nn.BCEWithLogitsLoss,
                                    optimizer=optim.SGD,
                                    max_epochs=100,
                                    batch_size=8,
                                    verbose=False)
    grid = GridSearchCV(estimator=net, param_grid=param_grid,
                        n_jobs=-1, cv=5, verbose=2, error_score='raise')
    grid_result = grid.fit(X_tensor, y_tensor)

    print(f'BEST using: {grid_result.best_score_}, {grid_result.best_params_}')
    means = grid_result.cv_results_['mean_test_score']
    std = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, params in zip(means, std, params):
        print(mean)
        print(stdev)
        print(params)

def main():
    PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\fluDiagnosis.csv"

    X, y, X_train, X_test, y_train, y_test = preprocess_data(PATH)
    logistic_model(X_train, X_test, y_train, y_test)
    print('='*60 + '\n')
    gridsearch_neural_net(X, y)


if __name__ == '__main__':
    main()