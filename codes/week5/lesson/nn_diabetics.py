import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\alt_diabetes.csv"

    df = pd.read_csv(PATH)

    X = df.drop(columns=['diabetes'], axis=1)
    y = df['diabetes']
    X = X.to_numpy(dtype=np.float32)
    y = y.to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        stratify=y, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2,
        stratify=y_train, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    X_val = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

    model = nn.Sequential(
        nn.Linear(8, 12),
        nn.ReLU(),
        nn.Linear(12, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 10
    BATCH = 10
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        correct = 0
        samples = 0

        for i in range(0, len(X_train), BATCH):
            X_batch = X_train[i : i + BATCH]
            y_batch = y_train[i : i + BATCH]

            # print(f'Batch: {i}')
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)

            preds = (y_pred >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            samples += y_batch.size(0)

        avg_loss = epoch_loss / samples
        train_acc = correct / samples
        train_losses.append(avg_loss)
        train_accs.append(train_acc)

        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_val)
            val_loss = loss_fn(y_pred_val, y_val).item()
            y_pred_val = (y_pred_val >= 0.5).float()
            val_acc = (y_pred_val == y_val).float().mean().item()

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Epoch {epoch + 1}: '
              f'loss={avg_loss:.4f} | acc={train_acc:.4f} '
              f'val_loss={val_loss:.4f} | val_acc={val_acc:.4f}')

    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test)
        test_loss = loss_fn(y_pred_test, y_test).item()
        y_pred_test = (y_pred_test >= 0.5).float()
        test_acc = (y_pred_test == y_test).float().mean().item()

    print('\n Test Evaluation')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Acc: {test_acc:.4f}')

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()