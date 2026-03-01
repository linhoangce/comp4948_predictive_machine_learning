import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split

### DATA           #######################################
# Setup data.
import pandas as pd
import numpy as np


# Load the flower feature data into a DataFrame.
df   = pd.DataFrame(columns=['Length', 'Width', 'IsRed'])
data = [
    {'Length':3,    'Width':1.5, 'IsRed': 1},
    {'Length':2,    'Width':1,   'IsRed': 0},
    {'Length':4,    'Width':1.5, 'IsRed': 1},
    {'Length':3,    'Width':1,   'IsRed': 0},
    {'Length':3.5,  'Width':.5,  'IsRed': 1},
    {'Length':2,    'Width':.5,  'IsRed': 0},
    {'Length':5.5,  'Width':1,   'IsRed': 1},
    {'Length':1,    'Width':1,   'IsRed': 0},
    {'Length':4.5,  'Width':1,  'IsRed':1}]
df = pd.DataFrame.from_records(data)

X = df.drop('IsRed', axis=1)
y = df['IsRed']

##########################################################
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# define standard scaler
from sklearn.preprocessing     import StandardScaler
scaler = StandardScaler()

# transform data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test  = torch.tensor(X_test_scaled, dtype=torch.float32)

# Reshapes array.
# unsqueeze() creates array of single dimensional arrays.
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)
y_test  = torch.tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1)

import torch
import torch.nn as nn

# Define the neural network architecture
class BinaryClassificationNet(nn.Module):
    def __init__(self):
        super(BinaryClassificationNet, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)     # Hidden layer.
        x = self.sigmoid(x) # Activation function.
        x = self.fc2(x)     # Output layer.
        x = self.sigmoid(x) # Activation function.
        return x

# Instantiate the model
model = BinaryClassificationNet()

# Define the loss function and optimizer
criterion = nn.BCELoss() # Binary cross entropy.
# Use stochastic gradient descent to update weights & bias.
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(2000):
    # print("Epoch: " + str(epoch))
    # Forward pass
    output = model(X_train)
    loss = criterion(output, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluate the model
with torch.no_grad():
    outputs = model(X_test)
    predictions = outputs.round()
    accuracy = (predictions == y_test).float().mean()
    print(f'Accuracy: {accuracy}')
