import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

def get_data():
    X, y = fetch_california_housing(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    return X_train_tensor, X_val_tensor, X_test_tensor, \
            y_train_tensor, y_val_tensor, y_test_tensor

def train(model, optimizer, loss_fn, X_train, X_val, y_train, y_val,
          batch_size, epochs):
    best_val_rmse = np.inf
    best_weights = None
    train_hist, val_hist = [], []

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            idx = perm[i : i + batch_size]
            X_batch = X_train[idx]
            y_batch = y_train[idx]

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss))

        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_val)
            rmse_val = mape_loss(y_pred_val, y_val)

        train_rmse = float(np.mean(epoch_losses))
        train_hist.append(train_rmse)
        val_hist.append(rmse_val)

        if rmse_val < best_val_rmse:
            best_val_rmse = rmse_val
            best_weights = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch+1}: Train MAPE={train_rmse:.4f} | Val MAPE={rmse_val:4f}')

    return model, best_weights, best_val_rmse, train_hist, val_hist

def get_model():
    return nn.Sequential(
        nn.Linear(8, 24),
        nn.ReLU(),
        nn.Linear(24, 12),
        nn.ReLU(),
        nn.Linear(12, 6),
        nn.ReLU(),
        nn.Linear(6, 1)
    )

def mape_loss(output, target):
    return torch.mean(torch.abs(target - output) / target)

def main():
    EPOCHS = 100
    BATCH_SIZE = 10

    X_train, X_val, X_test, y_train, y_val, y_test = get_data()

    model = get_model()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model, best_weights, best_val_mape, train_hist, val_hist = \
                        train(model, optimizer, loss_fn, X_train,
                                X_val, y_train, y_val, BATCH_SIZE, EPOCHS)

    model.eval()
    with torch.inference_mode():
        y_pred_test = model(X_test)
        test_mape = mape_loss(y_pred_test, y_test)

    print(f"\nBest VAL  MAPE: {best_val_mape:.4f}")
    print(f"Final TEST MAPE: {test_mape:.4f}")

    plt.plot(train_hist, label='Train RMSE')
    plt.plot(val_hist, label='Val RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()

    torch.save({
        'model': model.state_dict(),
        'optim': optimizer.state_dict()
    }, 'reg_model.pth')

    print('='*100)
    print(f'\nLoading Saved Model')

    state_dict = torch.load('reg_model.pth')
    new_model = get_model()
    new_model.load_state_dict(state_dict['model'])
    optimizer = optim.Adam(new_model.parameters(), lr=1e-3)
    optimizer.load_state_dict(state_dict['optim'])

    new_model.eval()
    with torch.inference_mode():
        y_pred = new_model(X_test)
        rmse_test = loss_fn(y_pred, y_test)

    print(f'\nLoaded Model RMSE: {rmse_test:.4f}')


if __name__ == '__main__':
    main()