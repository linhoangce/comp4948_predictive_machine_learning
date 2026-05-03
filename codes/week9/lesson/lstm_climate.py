import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import root_mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class AirModel(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,           # total features per time step
                                    # 20 time steps but only one number each
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)       # out: (batch, seq_len, hidden)
        last = out[:, -1, :]        # last time step shape (batch, hidden)
        pred = self.dense(last)
        return pred

def load_data(path, target_col, test_size):

    df = pd.read_csv(path)

    for i in range(1, 21):
        df[f"t-{i}"] = df["meantemp"].shift(i)
    df = df.dropna()

    lag_cols = [c for c in df.columns if c.startswith("t-")]
    print(f"Columns in CSV: {df.columns}")
    print(f"Features: {lag_cols}")

    # create arrays for N = # rows
    y = df[target_col].values.astype(np.float32) # (N,)
    X = df[lag_cols].values.astype(np.float32) # (N, 20)

    # add one more dimension to X
    X = X[..., np.newaxis] # (N, 20, 1)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    X_train = X[:-test_size]
    X_test = X[-test_size:]
    y_train = y[:-test_size]
    y_test = y[-test_size:]

    print(y_test)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = X_train.reshape(X_train.shape[0], -1) # (N_train, 20)
    X_test = X_test.reshape(X_test.shape[0], -1) # (N_test, 20)
    y_train = y_train.reshape(-1, 1) # (N_train, 1)
    y_test = y_test.reshape(-1, 1)

    print(f"X shape after reshape: {X_train.shape}")

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)

    # convert numpy to torch tensors - (rows, lags, 1 feature)
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(-1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(-1)
    print(f"X shape after tensor transform: {X_train_tensor.shape}")
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    return train_loader, test_loader, scaler_X, scaler_y

def main():
    PATH = "../data/DailyDelhiClimateTest.csv"
    TEST_SIZE = 3

    train_loader, test_loader, scaler_X, scaler_y = load_data(PATH, "meantemp", TEST_SIZE)

    model = AirModel(hidden_size=256)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 1500

    for epoch in range(1, EPOCHS+1):
        model.train()
        total = 0.0

        for xb, yb in train_loader:
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item() * xb.size(0)

        if epoch % 50 == 0:
            train_mse = total / len(train_loader.dataset)
            print(f"Epoch {epoch:02d}: train MSE = {train_mse:4f}")

    model.eval()
    preds = []
    targets = []

    with torch.inference_mode():
        for xb, yb in test_loader:
            pred = model(xb).numpy()
            preds.append(pred)
            targets.append(yb.numpy())

    preds = np.vstack(preds).squeeze()
    targets = np.vstack(targets).squeeze()

    # inverse transform for plotting in original units
    preds_orig = scaler_y.inverse_transform(preds.reshape(-1, 1)).squeeze()

    plt.figure(figsize=(10, 4))
    plt.plot(targets, label="Actual")
    plt.plot(preds_orig, label="Predicted")
    plt.title("LSTM Forecast")
    plt.legend()
    plt.show()

    rmse = root_mean_squared_error(targets, preds_orig)
    print(f"RMSE: {rmse:.4f}")

    print(f"PRed: {preds_orig}")
    print(f"Actual {targets}")


if __name__ == "__main__":
    main()