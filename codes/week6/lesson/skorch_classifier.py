import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring


class MyNeuralNet(nn.Module):
    def __init__(self, n_neurons, dropout=0.1):
        super().__init__()
        self.dense0 = nn.Linear(4, n_neurons)
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
    my_network = MyNeuralNet(n_neurons=25, dropout=0.1)
    nn = NeuralNetClassifier(my_network,
                             max_epochs=200,
                             lr=0.001,
                             batch_size=128,
                             optimizer=optim.RMSprop,
                             callbacks=[EpochScoring(scoring='accuracy',
                                                    name='linh_train_acc',
                                                    on_train=True)])
    model = nn.fit(X, y)
    return model, nn

def evaluate_model(model, X_test, y_test):
    print(model)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

def main():
    PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\bill_authentication.csv"

    df = pd.read_csv(PATH)
    X = df.drop(columns=['Class'], axis=1).values.astype(np.float32)
    y = df['Class'].values.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model, net = build_model(X_train_scaled, y_train)
    evaluate_model(model, X_test_scaled, y_test)

    plt.subplots(figsize=(12, 5), sharex=True, sharey=False)
    plt.subplot(1, 2, 1)
    plt.plot(net.history[:, 'train_loss'], color='blue', label='train')
    plt.plot(net.history[:, 'valid_loss'], color='orange', label='val')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(net.history[:, 'linh_train_acc'], color='blue', label='train')
    plt.plot(net.history[:, 'valid_acc'], color='orange', label='val')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()