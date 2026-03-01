import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class FluDiagnoseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(2, 12)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(12, 8)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(8, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x

def main():
    PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\fluDiagnosis.csv"

    df = pd.read_csv(PATH)
    X = df.drop(columns=['Diagnosed'], axis=1).to_numpy(dtype=np.float32)
    y = df['Diagnosed'].to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    model = FluDiagnoseClassifier()
    model(X_train_tensor)


if __name__ == '__main__':
    main()