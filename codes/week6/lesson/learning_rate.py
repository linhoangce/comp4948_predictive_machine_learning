import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def main():
    PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\ionosphere.csv"

    df = pd.read_csv(PATH, header=None)
    X = df.iloc[:, 0:34]
    y = df.iloc[:, 34]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.float32).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, shuffle=True
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = nn.Sequential(
        nn.Linear(34, 34),
        nn.ReLU(),
        nn.Linear(34, 1),
        nn.Sigmoid()
    )

    epochs = 50
    batch_size = 16
    batch_start = torch.arange(0, len(X_train), batch_size)

    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

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

        before_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        after_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch}: SGD LR = {before_lr:.4f} -> {after_lr:.4f}')

    model.eval()
    y_val = model(X_test)
    acc = (y_val.round() == y_test).float().mean()
    acc = float(acc)
    print(f'Test Acc: {acc:.2f}')


if __name__ == '__main__':
    main()