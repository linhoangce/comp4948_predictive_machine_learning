import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from rnn_sine_wave import SimpleRNNRegressor, process_data

def main():
    PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\NVDA_AMD_OpenPrices.csv"

    df = pd.read_csv(PATH)

    print(f"Columns in CSV: {df.columns}")

    feature_cols = ["NVDA_Ot-1", "NVDA_Ot-2"]
    target_col = "NVDA_Open"

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # reshape inputs into 3D matrix
    X = X[..., np.newaxis] # (N, 2, 1)

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y).unsqueeze(1) # change shape to (N, 1)

    split = int(len(X) * 0.99)

    X_train = X_tensor[:split]
    y_train = y_tensor[:split]
    X_test = X_tensor[split:]
    y_test = y_tensor[split:]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    model = SimpleRNNRegressor(input_size=1, hidden_size=32)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(100):
        model.train()
        total = 0.0

        for xb, yb in train_loader:
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item() * xb.size(0)

        train_mse = total / len(train_loader.dataset)
        print(f"Epoch {epoch:02d}: train MSE = {train_mse:.4f}")

    model.eval()
    preds = []
    targets = []

    with torch.inference_mode():
        for xb, yb in test_loader:
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            targets.append(yb.numpy())

    preds = np.vstack(preds).squeeze()
    targets = np.vstack(targets).squeeze()

    plt.figure(figsize=(10, 4))
    plt.plot(targets, label="Actual")
    plt.plot(preds, label="Predicted")
    plt.title("SImple RNN: Stock Price prediction")
    plt.legend()
    plt.show()

    rmse = root_mean_squared_error(targets, preds)
    print(f"Test RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()