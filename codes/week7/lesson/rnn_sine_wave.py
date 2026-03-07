import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error

from torch.utils.data import DataLoader, TensorDataset

class SimpleRNNRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size, # total features per time step
                         # 20 time steps but only one number each
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dense = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x):
        out, _ = self.rnn(x) # out: (batch, seq_len, hidden)
        last = out[:, -1, :] # last time step: (batch, hidden)
        pred = self.dense(last) # (batch, 1)
        return pred

def process_data(df):
    lag_cols = [c for c in df.columns if c.startswith("t-")]
    lag_cols = sorted(lag_cols, key=lambda s: int(s.split("-")[1])) # t-1, t-2...

    print(f"Columns in CSV: {list(df.columns)}")
    print(f"Using feature columns: {lag_cols}")

    y = df["y"].values.astype(np.float32)
    X = df[lag_cols].values.astype(np.float32)

    # RNN expects (batch, seq_len, input_size)
    # convert 2D matrix -> 3D, shape (N, 20, 1)
    X = X[..., np.newaxis]
    print(f"converted shape: {X.shape}")

    # to tensor
    X = torch.tensor(X)
    y = torch.tensor(y).unsqueeze(-1) # (N, 1)

    print(f"X tensor shape: {X.shape}")
    print(f"y tensor shape: {y.shape}")

    split = int(len(X) * 0.8)

    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_dataloader, test_dataloader

def main():
    PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\sine_wave_data.csv"

    df = pd.read_csv(PATH)

    train_dataloader, test_dataloader = process_data(df)

    model = SimpleRNNRegressor(hidden_size=64)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 10
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0

        for xb, yb in train_dataloader:
            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item() * xb.size(0)

        train_mse = total / len(train_dataloader.dataset)
        print(f"Epoch {epoch:02d}: train MSE = {train_mse:.4f}")

    model.eval()
    preds = []
    targets = []

    with torch.inference_mode():
        for xb, yb in test_dataloader:
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            targets.append(yb.numpy())

    preds = np.vstack(preds).squeeze()
    targets = np.vstack(targets).squeeze()

    plt.figure(figsize=(10, 4))
    plt.plot(targets, label="Actual")
    plt.plot(preds, label="Predicted")
    plt.title("Simple RNN: One-step Forecast (from CSV)")
    plt.legend()
    plt.show()

    rmse = root_mean_squared_error(targets, preds)
    print(f"RMSE: {rmse}")


if __name__ == "__main__":
    main()