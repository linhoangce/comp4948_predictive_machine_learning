import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from skorch import NeuralNetBinaryClassifier

def preprocess_data(path):
    df = pd.read_csv(path, header=None)

    X = df.iloc[:, 0:60]
    y = df.iloc[:, 60]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    return X, y

class SonarClassifier(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_neurons):
        super().__init__()
        self.dense0 = nn.Linear(n_inputs, n_neurons)
        self.act0 = nn.ReLU()
        self.dense1 = nn.Linear(n_neurons, n_neurons)
        self.act1 = nn.ReLU()
        self.dense2 = nn.Linear(n_neurons, n_neurons)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(n_neurons, n_outputs)

    def forward(self, x):
        return self.output(
            self.act2(self.dense2(
                self.act1(self.dense1(
                    self.act0(self.dense0(x))
                ))
            ))
        )

def main():
    PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\sonar.csv"
    X, y = preprocess_data(PATH)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []

    for train_idx, test_idx in kfold.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        model = NeuralNetBinaryClassifier(
            SonarClassifier(X_train.shape[1], 1, n_neurons=64),
            criterion=nn.BCEWithLogitsLoss,
            lr=0.0001,
            optimizer=optim.Adam,
            batch_size=16,
            max_epochs=150,
            verbose=False
        )
        model.fit(X_train_tensor, y_train_tensor)
        y_pred = model.predict(X_test_tensor)

        acc = accuracy_score(y_test_tensor, y_pred)
        fold_accuracies.append(acc)
        print(f'Fold Acc: {acc:.2f}')

    print('\nCROSS-VALIDATION RESUlLTs\n')
    print(f'Mean acc: {np.mean(fold_accuracies):.2f}')
    print(f'Std acc: {np.std(fold_accuracies):.2f}')


if __name__ == '__main__':
    main()