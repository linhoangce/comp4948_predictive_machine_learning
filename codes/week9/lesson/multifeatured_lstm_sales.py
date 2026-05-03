import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import root_mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

class SalesModel(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        pred = self.dense(last)
        return pred

def create_lag_features(df, base_cols, total_lags):
    lag_cols = {f"t-{i}__{col}": df[col].shift(i)
                for i in range(1, total_lags+1)
                for col in base_cols}
    lag_df = pd.DataFrame(lag_cols, index=df.index)
    df = pd.concat([df, lag_df], axis=1)
    df = df.dropna()
    return df, list(lag_df.columns)

def preprocess_data(path, target_col, test_size, total_lags):
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    base_cols = df.drop(columns=[target_col]).columns

    df, lag_cols = create_lag_features(df, base_cols, total_lags)

    X = df[lag_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    X_train = X[:-test_size]
    X_test = X[-test_size:]
    y_train = y[:-test_size]
    y_test = y[-test_size:]

    N_FEATURES = len(base_cols)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    print(f"X shape: {X_train_scaled.shape}")

    # reshape train and test arrays to 3D
    # shape (rows, seq_len, n_features)
    X_train_scaled = X_train_scaled.reshape(
        X_train_scaled.shape[0],
        total_lags,
        N_FEATURES
    )
    X_test_scaled = X_test_scaled.reshape(
        X_test_scaled.shape[0],
        total_lags,
        N_FEATURES)
    print(f"X reshape: {X_train_scaled.shape}")

    # reshape target to 2D for scaling
    y_train = y_train.reshape(-1, 1)
    y_train_scaled = scaler_y.fit_transform(y_train)

    # convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    return train_loader, test_loader, scaler_X, scaler_y, N_FEATURES

def main():

    PATH = "../data/product_daily_sales.csv"
    TOTAL_LAGS = 5
    TEST_SIZE = 150

    train_loader, test_loader, scaler_X, scaler_y, n_features = preprocess_data(
        PATH,
        "product_0",
        test_size=TEST_SIZE,
        total_lags=TOTAL_LAGS
    )

    model = SalesModel(input_size=n_features, hidden_size=64)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 1500

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print("loss.item:", loss.item())
            total_loss += loss.item() * xb.size(0)

        if epoch % 50 == 0:
            # print(f"loss: {total_loss}")
            train_mse = total_loss / len(train_loader.dataset)
            print(f"Epoch {epoch:02d}: train MSE = {train_mse:4f}")

    model.eval()
    preds = []
    targets = []

    with torch.inference_mode():
        for xb, yb in test_loader:
            pred = model(xb)
            preds.append(pred)
            targets.append(yb)


    preds = np.concatenate(preds, axis=0).flatten()
    targets = np.concatenate(targets, axis=0).flatten()

    preds_orig = scaler_y.inverse_transform(preds.reshape(-1, 1)).squeeze()
    rmse = root_mean_squared_error(targets, preds_orig)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"PRed: {preds_orig}")
    print(f"Actual {targets}")

    plt.figure(figsize=(10, 4))
    plt.plot(targets, label="actual")
    plt.plot(preds_orig, label="predicted")
    plt.title("Multifeatured LSTM Sales Forcast")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()