import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring

class FluClassifier(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_neurons, dropout):
        super().__init__()
        self.dense0 = nn.Linear(n_inputs, n_neurons)
        self.act0 = nn.ReLU()
        self.dropout0 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(n_neurons, n_neurons)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.output = nn.Linear(n_neurons, n_outputs)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.output(
            self.dropout1(self.act1(self.dense1(
                self.dropout0(self.act0(self.dense0(x)))
            )))
        ))

def build_model(X, y):
    cls = FluClassifier(n_inputs=2, n_outputs=2, n_neurons=10, dropout=0.5)
    nn = NeuralNetClassifier(cls,
                             max_epochs=20,
                             lr=0.1,
                             batch_size=16,
                             optimizer=optim.Adam,
                             callbacks=[EpochScoring(scoring='accuracy',
                                                     name='train_acc',
                                                     on_train=True)])

    model = nn.fit(X, y)
    return model, nn

def evaluate_model(model, X_test, y_test):
    print(model)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

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

    model, nn = build_model(X_train_scaled, y_train)
    evaluate_model(model, X_test_scaled, y_test)

    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(nn.history[:, 'train_acc'], label='train')
    axes[0].plot(nn.history[:, 'valid_acc'], label='validation')
    axes[0].legend()
    axes[0].set_title('Accuracy')

    axes[1].plot(nn.history[:, 'train_loss'], label='train')
    axes[1].plot(nn.history[:, 'valid_loss'], label='loss' )
    axes[1].legend()
    axes[1].set_title('Loss')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()