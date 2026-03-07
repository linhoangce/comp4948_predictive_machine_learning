import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def generate_data():
    torch.manual_seed(0)

    X = torch.linspace(0, 10, steps=100).unsqueeze(1) # shape (100, 1)
    y = 2 * X + 1 + 0.1 * torch.randn_like(X) # add tiny noise
    return X, y

def get_data(train_ratio=0.8):
    X, y = generate_data()

    n = len(X)
    n_train = int(n * train_ratio)

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print(len(X_train))

    # convert to numpy for sklearn processing
    X_train_np = X_train.numpy()
    X_test_np = X_test.numpy()
    y_train_np = y_train.numpy()
    y_test_np = y_test.numpy()

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train_np)
    X_test_scaled = x_scaler.transform(X_test_np)

    y_train_scaled = y_scaler.fit_transform(y_train_np)
    y_test_scaled = y_scaler.transform(y_test_np)

    # back to tensor
    X_train_tensor = torch.from_numpy(X_train_scaled).float()
    X_test_tensor = torch.from_numpy(X_test_scaled).float()
    y_train_tensor = torch.from_numpy(y_train_scaled).float()
    y_test_tensor = torch.from_numpy(y_test_scaled).float()

    return X_train, X_test, y_train, y_test, X_train_tensor, X_test_tensor, \
            y_train_tensor, y_test_tensor, x_scaler, y_scaler

def main():
    (X_train, X_test, y_train, y_test, X_train_tensor,
     X_test_tensor,  y_train_tensor, y_test_tensor,
     x_scaler, y_scaler) = get_data(train_ratio=0.8)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(50):
        model.train()
        train_loss = 0.0

        for batch_idx, (xb, yb) in enumerate(train_loader):
            print(f"len(xb): {len(xb)}")
            print(f"batch num: {batch_idx}")

            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                y_pred = model(xb)
                loss = loss_fn(y_pred, yb)
                test_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: train_loss = {train_loss:.4f} | test_loss = {test_loss:.4f}\n")

    model.eval()
    with torch.inference_mode():
        preds_test_n = model(X_test_tensor).numpy()
        preds_test = y_scaler.inverse_transform(preds_test_n)

    print("="* 100 + "\n")
    for i in range(0, len(X_test), len(X_test) // 5):
        print(f"{X_test[i].item():5.2f} {y_test[i].item():8.3f} {preds_test[i, 0]:8.3f}")


if __name__ == "__main__":
    main()