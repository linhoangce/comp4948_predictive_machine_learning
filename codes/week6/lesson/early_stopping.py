import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, EarlyStopping

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def preprocess_data(path):
    df = pd.read_csv(path)
    df = pd.get_dummies(df,
                        columns=['Gender', 'Ever_Married', 'Graduated',
                            'Profession', 'Spending_Score', 'Var_1'],
                        drop_first=True)
    df['Segmentation'] = df['Segmentation'].replace({
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3
    })

    X = df.drop('Segmentation', axis=1).astype(np.float32)
    y = df['Segmentation'].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, shuffle=True
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

class CustomerClassifier(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_neurons, dropout):
        super().__init__()
        self.dense0 = nn.Linear(n_inputs, n_neurons)
        self.act0 = nn.ReLU()
        self.dropout0 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(n_neurons, n_neurons)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(n_neurons, n_neurons)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.output = nn.Linear(n_neurons, n_outputs)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.output(
            self.dropout2(self.act2(self.dense2(
                self.dropout1(self.act1(self.dense1(
                    self.dropout0(self.act0(self.dense0(x)))
                )))
            )))
        ))


def build_net(X_train, y_train):
    model = CustomerClassifier(X_train.shape[1], 4,
                               n_neurons=32, dropout=0.1)
    net = NeuralNetClassifier(model,
                              max_epochs=500,
                              lr=0.005,
                              batch_size=32,
                              optimizer=optim.Adam,
                              callbacks=[EpochScoring(scoring='accuracy',
                                                      name='train_acc'),
                                         EarlyStopping(patience=50)])
    fit_net = net.fit(X_train, y_train)
    return fit_net, net

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

def draw_plots(history):
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(history[:, 'train_acc'], label='train')
    ax[0].plot(history[:, 'valid_acc'], label='valid')
    ax[0].set_title('Accuracy')
    ax[0].legend()

    ax[1].plot(history[:, 'train_loss'], label='train')
    ax[1].plot(history[:, 'valid_loss'], label='valid')
    ax[1].set_title('Loss')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

def main():
    PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\CustomerSegmentation.csv"

    X_train, X_test, y_train, y_test = preprocess_data(PATH)
    model, net = build_net(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    draw_plots(net.history)


if __name__ == '__main__':
    main()