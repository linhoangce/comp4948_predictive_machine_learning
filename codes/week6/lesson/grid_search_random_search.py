import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier


class MyNeuralNet(nn.Module):
    def __init__(self, n_neurons, dropout):
        super().__init__()
        self.dense0 = nn.Linear(2, n_neurons)
        self.act0 = nn.ReLU()
        self.dropout0 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(n_neurons, n_neurons)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.output = nn.Linear(n_neurons, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.act0(self.dense0(x))
        x = self.dropout0(x)
        x = self.act1(self.dense1(x))
        x = self.dropout1(x)
        x = self.softmax(self.output(x))
        return x

def build_model(X, y):
    nn = NeuralNetClassifier(MyNeuralNet, verbose=0, train_split=False)

    params = {
        'max_epochs': [10, 20],
        'lr': [0.1, 0.001],
        'module__n_neurons': [5, 10],
        'module__dropout': [0.1, 0.5],
        'optimizer': [optim.Adam, optim.SGD, optim.RMSprop]
    }

    gs = GridSearchCV(nn, params, refit=True,
                    cv=3, scoring='balanced_accuracy', verbose=1,
                    error_score='raise')
    return gs.fit(X, y)

def evaluate_model(model, X_test, y_test):
    print(model)
    y_pred = model.predict(X_test)
    report = classification_report(y_pred, y_test)
    print(report)

def main():
    PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\fluDiagnosis.csv"

    df = pd.read_csv(PATH)

    X = df.drop(columns=['Diagnosed']).values.astype(np.float32)
    y = df['Diagnosed'].values.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_model(X_train_scaled, y_train)
    print("Linh's flu diagnosis net work best parameters")
    print(model.best_params_)

    evaluate_model(model.best_estimator_,
                   X_test_scaled, y_test)


if __name__ == '__main__':
    main()