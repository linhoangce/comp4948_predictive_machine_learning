import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split


def visualize_data(X, y):
    X_np = X.numpy()
    y_np = y.numpy().ravel()

    mask0 = y_np == 0
    mask1 = y_np == 1

    plt.figure(figsize=(6, 6))

    plt.scatter(X_np[mask0, 0], X_np[mask0, 1], s=10, label='y=0', alpha=0.6)
    plt.scatter(X_np[mask1, 0], X_np[mask1, 1], s=10, label='y=1', alpha=0.6)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('make_circles dataset')
    plt.legend()
    plt.axis('equal')
    plt.show()

def get_data():
    X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
    visualize_data(X, y)
    return X, y

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Linear(2, 5)
        self.act0 = nn.LeakyReLU()
        self.layer1 = nn.Linear(5, 5)
        self.act1 = nn.LeakyReLU()
        self.layer2 = nn.Linear(5, 5)
        self.act2 = nn.LeakyReLU()
        self.layer3 = nn.Linear(5, 5)
        self.act3 = nn.LeakyReLU()
        self.layer4 = nn.Linear(5, 1)
        # self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.act0(self.layer0(x))
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.layer4(x)
        return x

def train_loop(model, loss_fn, optimizer, X_train, y_train,
               X_val, y_val, epochs=300, batch_size=32):
    batch_start = torch.arange(0, len(X_train), batch_size)

    bce_hist = []
    acc_hist = []
    grad_hist = [[], [], [], [], []]

    for epoch in range(epochs):
        model.train()

        layer_grad = [[], [], [], [], []]

        for start in batch_start:
            X_batch = X_train[start : start + batch_size]
            y_batch = y_train[start: start + batch_size]

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            layers = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]

            for n, layer in enumerate(layers):
                mean_grad = float(layer.weight.grad.abs().mean())
                layer_grad[n].append(mean_grad)

        model.eval()
        with torch.no_grad():
            logits = model(X_val)
            # y_pred_val = (torch.sigmoid(logits))
            bce = float(loss_fn(logits, y_val))
            y_pred_val = (torch.sigmoid(logits) > 0.5)
            acc = float((y_pred_val == y_val).float().mean())

        bce_hist.append(bce)
        acc_hist.append(acc)

        for n in range(len(layer_grad)):
            grads = layer_grad[n]
            total = sum(grads)
            count = len(grads)
            avg_grad = total / count
            grad_hist[n].append(avg_grad)

        if epoch % 10 == 9:
            print(f'Epoch {epoch+1}, BCE = {bce:.4f} | Accuracy = {acc:.2f}')

    return bce_hist, acc_hist, grad_hist

def plot_average_gradients(loss_hist, acc_hist, grad_hist, title):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_hist, label='BCE Loss')
    plt.plot(acc_hist, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylim(0, 1)
    plt.title(title + 'Loss & Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    n_layers = len(grad_hist)

    for n in range(n_layers):
        grads = grad_hist[n]
        label = f'layer {n}'
        plt.plot(grads, label=label)

    plt.xlabel('Epochs')
    plt.legend()

    plt.suptitle(str(nn.LeakyReLU))
    plt.title(f'{title} Average Gradients')
    plt.tight_layout()
    plt.show()

def main():
    X, y = get_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    bce_hist, acc_hist, grad_hist = train_loop(model, loss_fn, optimizer,
                                               X_train, y_train, X_test, y_test)

    plot_average_gradients(bce_hist, acc_hist, grad_hist, 'LeakyReLU')


if __name__ == '__main__':
    main()