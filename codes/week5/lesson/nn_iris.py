import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class NNIris(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 8)
        self.act = nn.ReLU()
        self.output = nn.Linear(8, 3)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x

def main():
    PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\iris_alpha.csv"

    df = pd.read_csv(PATH)

    X = df.drop(columns=['iris_type']).to_numpy(dtype=np.float32)
    y = df['iris_type']
    y = pd.get_dummies(y, dtype=int)
    y = y.to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    model = NNIris()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 200
    BATCH = 5
    batches_per_epoch = len(X_train) // BATCH

    best_acc = -np.inf
    best_weight = None
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in range(EPOCHS):
        # shuffle training indices each epoch
        # create random permutation of integers with each integer
        # appearing only once
        perm = torch.randperm(len(X_train_tensor))

        epoch_loss = []
        epoch_acc = []

        model.train()

        for i in range(batches_per_epoch):
            start = i * BATCH
            idx = perm[start : start + BATCH]
            X_batch = X_train_tensor[idx]
            y_batch = y_train_tensor[idx]

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))

        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor)
            loss_test = loss_fn(y_pred, y_test_tensor)
            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test_tensor, 1)).float().mean()

        train_losses.append(np.mean(epoch_loss))
        train_accs.append(np.mean(epoch_acc))
        test_losses.append(float(loss_test))
        test_accs.append(float(acc))

        if acc > best_acc:
            best_acc = acc
            best_weight = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch} | validation: Cross-entropy={loss_test:4f}, acc={acc*100:.2f}%')

    model.load_state_dict(best_weight)

    plt.plot(train_losses, label='train')
    plt.plot(test_losses, label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy')
    plt.legend()
    plt.show()

    plt.plot(train_accs, label='Train')
    plt.plot(test_accs, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()