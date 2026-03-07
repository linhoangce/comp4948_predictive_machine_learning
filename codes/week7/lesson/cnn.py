import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def prepare_data_loader():
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )

    # load dataset
    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True,
        download=True,
        transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False,
        download=True,
        transform=transform
    )

    # plot the eight image
    plt.imshow(train_set.data[7])
    plt.show()

    print("Image dimensions")
    image, label = train_set[7]
    print(image.shape)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

    return train_loader, test_loader

class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=36, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(in_channels=36, out_channels=36, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))

        self.flat = nn.Flatten()

        # fully connected layer:
        # input from previous flatten layer:
        # 36 * 16 * 16 = 9216
        # 16 = 32 images W / 2 pool
        # 16 = 32 images H / 2 pool
        self.dense3 = nn.Linear(9216, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.output = nn.Linear(512, 10)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        x = self.act2(self.conv2(x))
        x = self.pool2(x)
        x = self.flat(x)
        x = self.act3(self.dense3(x))
        x = self.drop3(x)
        return self.output(x)

def main():
    train_loader, test_loader = prepare_data_loader()

    model = CIFAR10Model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(4):
        model.train()

        for inputs, labels in train_loader:
            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = 0
        count = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                y_pred = model(inputs)
                acc += (torch.argmax(y_pred, 1) == labels).float().sum()
                count += len(labels)

        acc /= count
        print(f"Epoch {epoch}: acc = {acc:.2f}")

    torch.save(model.state_dict(), "cifar10_model.pth")


if __name__ == "__main__":
    main()