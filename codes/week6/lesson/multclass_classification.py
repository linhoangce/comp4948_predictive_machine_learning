import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, EarlyStopping


class IrisClassifier(nn.Module):
    def __init__(self, n_features, n_neurons, n_outputs, dropout):
        super().__init__()
        self.dense0 = nn.Linear(n_features, n_neurons)
        self.act0 = nn.ReLU()
        self.dropout0 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(n_neurons, n_neurons)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.output = nn.Linear(n_neurons, n_outputs)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return (self.softmax(self.output(
            self.dropout1(self.act1(self.dense1(
                self.dropout0(self.act0(self.dense0(x)))
            )))
        )))

def evaluate_model(model, X_test, y_test):
    print(model)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

def build_model(X_train, y_train):
    model = IrisClassifier(4, 25, 4, dropout=0.1)
    net = NeuralNetClassifier(model,
                              max_epochs=200,
                              lr=0.001,
                              batch_size=128,
                              optimizer=optim.RMSprop,
                              callbacks=[EpochScoring(scoring='accuracy',
                                                      name='train_acc',
                                                      on_train=True)]
                                         # EarlyStopping(patience=100)]
                              )
    fit_model = net.fit(X_train, y_train)
    return fit_model, net

def preprocessing():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def draw_plots(history):
    _, ax = plt.subplots(1, 2)

    ax[0].plot(history[:, 'train_acc'], label='train')
    ax[0].plot(history[:, 'valid_acc'], label='test')
    ax[0].set_title('Accuracy')
    ax[0].legend()

    ax[1].plot(history[:, 'train_loss'], label='train')
    ax[1].plot(history[:, 'valid_loss'], label='test')
    ax[1].set_title('Loss')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

def main():
    X_train, X_test, y_train, y_test = preprocessing()

    model, net = build_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    draw_plots(net.history)


if __name__ == '__main__':
    main()