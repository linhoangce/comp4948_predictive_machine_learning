import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetBinaryClassifier

def preprocess_data(path):
    df = pd.read_csv(path, header=None)

    X = df.iloc[:, 0:60]
    y = df.iloc[:, 60]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return X_tensor, y_tensor

class SonarClassifier(nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.acts = nn.ModuleList()

        for i in range(n_layers):
            self.layers.append(nn.Linear(60, 60))
            self.acts.append(nn.ReLU())

        self.output = nn.Linear(60, 1)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.acts[i](self.layers[i](x))
        x = self.output(x)
        return x

def main():
    PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\sonar.csv"

    X, y = preprocess_data(PATH)

    model = NeuralNetBinaryClassifier(
        SonarClassifier,
        criterion=nn.BCEWithLogitsLoss,
        optimizer=optim.Adam,
        lr=0.0001,
        max_epochs=150,
        batch_size=10,
        verbose=False
    )

    params_grid = {
        'module__n_layers': [1, 3, 5],
        'lr': [0.1, 0.01, 0.001, 0.0001],
        'max_epochs': [100, 150]
    }

    grid_search = GridSearchCV(model, params_grid, scoring='accuracy', verbose=2, cv=3)
    result = grid_search.fit(X, y)

    print(f'Best Using: {result.best_score_} {result.best_params_}')

    means = result.cv_results_['mean_test_score']
    stds = result.cv_results_['std_test_score']
    params = result.cv_results_['params']
    for mean, std, param in zip(means, stds, params):
        print(means, stds, params)


if __name__ == '__main__':
    main()