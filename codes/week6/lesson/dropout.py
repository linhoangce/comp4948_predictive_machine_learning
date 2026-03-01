import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

def get_x_and_y_tensors(path):
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, 0:60]
    y = df.iloc[:, 60]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.float32).reshape(-1, 1)

    return X_tensor, y_tensor

class SonarClassifier(nn.Module):
    P = 0.2
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(60, 60)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(self.P)
        self.layer2 = nn.Linear(60, 60)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(self.P)
        self.layer3 = nn.Linear(60, 30)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(self.P)
        self.output = nn.Linear(30, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.dropout1(x)
        x = self.act2(self.layer2(x))
        x = self.dropout2(x)
        x = self.act3(self.layer3(x))
        x = self.dropout3(x)
        x = self.sigmoid(self.output(x))
        return x

def train(model, optimizer, loss_fn, X_train, X_val, y_train, y_val,
          epochs=300, batch_size=16):
    batch_start = torch.arange(0, len(X_train), batch_size)

    model.train()

    for epoch in range(epochs):
        for start in batch_start:
            X_batch = X_train[start : start + batch_size]
            y_batch = y_train[start : start + batch_size]

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred = model(X_val)
            acc = float((y_pred.round() == y_val).float().mean())

    return acc

def main():
    PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\sonar.csv"

    X, y = get_x_and_y_tensors(PATH)

    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    accuracies = []
    baseline_strings = []

    for i in range(4):
        for train_idx, test_idx in kfold.split(X, y):
            model = SonarClassifier()
            loss_fn = nn.BCELoss()
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)

            acc = train(model, optimizer, loss_fn, X[train_idx], X[test_idx],
                        y[train_idx], y[test_idx])
            print(f'Accuracy: {acc:.2f}')
            accuracies.append(acc)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        baseline = f'Baseline: mean={mean_acc*100:.2f}. std={std_acc*100:.2f}'
        baseline_strings.append(baseline)

    for baseline in baseline_strings:
        print(baseline)


if __name__ == '__main__':
    main()