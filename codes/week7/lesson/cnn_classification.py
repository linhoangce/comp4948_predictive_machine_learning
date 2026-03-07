import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=36, kernel_size=(3, 3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(in_channels=36, out_channels=36, kernel_size=(3, 3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

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
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    model = CIFAR10Model()
    model.load_state_dict(torch.load("cifar10_model.pth", map_location="cpu"))
    model.eval()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor()
    ])

    image = Image.open("data/test.png").convert("RGB")
    print(f"original dim: {np.array(image).shape}")
    x = transform(image).unsqueeze(0) # shape (1, 3, 32, 32)
    print(f"after transform: {x.shape}")

    with torch.inference_mode():
        outputs = model(x)
        predicted_class = torch.argmax(outputs, dim=1).item()

    print(f"Predicted class: {class_names[predicted_class]}")


if __name__ == "__main__":
    main()