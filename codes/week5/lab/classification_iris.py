import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def get_tensor():
    PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\iris_alpha.csv"

    df = pd.read_csv(PATH)
    X = df.drop(columns=['iris_type'], axis=1).to_numpy(dtype=np.float32)
    y = df['iris_type']
    y = pd.get_dummies(y).to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, shuffle=True, stratify=y_train
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled)
    X_val_tensor = torch.tensor(X_val_scaled)
    X_test_tensor = torch.tensor(X_test_scaled)
    y_train_tensor = torch.tensor(y_train)
    y_val_tensor = torch.tensor(y_val)
    y_test_tensor = torch.tensor(y_test)

    return X_train_tensor, X_val_tensor, X_test_tensor, \
            y_train_tensor, y_val_tensor, y_test_tensor

class IrisClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 8)
        self.act = nn.ReLU()
        self.output = nn.Linear(8, 3)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x

def training(model, optimizer, loss_fn, X_train, y_train, X_val, y_val, epochs,
             batches_per_epoch, batch_size):
    best_weights = None
    best_acc = -np.inf
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        perm = torch.randperm(len(X_train))

        epoch_loss = []
        epoch_acc = []

        model.train()
        for i in range(batches_per_epoch):
            start = i * batch_size
            idx = perm[start : start + batch_size]
            X_batch = X_train[idx]
            y_batch = y_train[idx]

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, dim=1)).float().mean()
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))

        model.eval()
        with torch.inference_mode():
            y_pred_val = model(X_val)
            loss_val = loss_fn(y_pred_val, y_val)
            acc_val = (torch.argmax(y_pred_val, 1) == torch.argmax(y_val, dim=1)).float().mean()

        train_losses.append(np.mean(epoch_loss))
        train_accs.append(np.mean(epoch_acc))
        val_losses.append(float(loss_val))
        val_accs.append(float(acc_val))

        if acc_val > best_acc:
            best_acc = acc_val
            best_weights = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch}, Validation: Cross-entropy={loss_val:.4f} |'
              f' Accuracy={acc_val:.2f}')

    return model, best_weights

def show_metrics(y_true, y_pred):
    print(classification_report(
        y_true, y_pred,
        target_names=['setosa', 'versicolor', 'virginica']
    ))

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = get_tensor()

    EPOCHS = 12
    BATCH_SIZE = 5
    BATCHES_PER_EPOCH = len(X_train) // BATCH_SIZE

    model = IrisClassifier()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    model, best_weights = training(model, optimizer, loss_fn,
                                   X_train, y_train, X_val, y_val,
                                   EPOCHS, BATCHES_PER_EPOCH, BATCH_SIZE)

    model.eval()
    with torch.inference_mode():
        logits = model(X_test)
        y_pred = torch.argmax(logits, dim=1).numpy()
        y_true = torch.argmax(y_test, dim=1).numpy()
    show_metrics(y_true, y_pred)

    torch.save({
        'model': model.state_dict(),
        'optim': optimizer.state_dict()
    }, 'checkpoint.pth')

    checkpoint = torch.load('checkpoint.pth')

    new_model = IrisClassifier()
    new_model.load_state_dict(checkpoint['model'])
    optimizer = optim.Adam(new_model.parameters())
    optimizer.load_state_dict(checkpoint['optim'])

    NEW_EPOCHS = 50
    new_model, best_weights = training(new_model, optimizer, loss_fn,
                                       X_train, y_train, X_val, y_val,
                                       NEW_EPOCHS, BATCHES_PER_EPOCH, BATCH_SIZE)
    new_model.eval()
    with torch.inference_mode():
        logits_new = new_model(X_test)
        y_pred_new = torch.argmax(logits_new, dim=1).numpy()
        y_true_new = torch.argmax(y_test, dim=1).numpy()

    show_metrics(y_true_new, y_pred_new)


if __name__ == '__main__':
    main()