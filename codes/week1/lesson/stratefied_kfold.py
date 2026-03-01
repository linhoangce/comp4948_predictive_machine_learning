import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold

path = r"C:\Users\linho\Desktop\CST\term4\pa\data\healthcare-dataset-stroke-data.csv"
df = pd.read_csv(path)

X = df.drop(columns=['stroke'])
y = df['stroke']

kfold = StratifiedKFold(n_splits=3, shuffle=True)

train_counts = []
test_counts = []

for train_idx, test_idx in kfold.split(X, y):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    train_count = y_train.value_counts().get(1, 0)
    train_counts.append(train_count)
    test_count = y_test.value_counts().get(1, 0)
    test_counts.append(test_count)

def plot_label_counts(data, title):
    plt.rcParams.update({'font.size': 20})
    plt.bar(['0', '1', '2'], data, edgecolor='black')
    plt.xticks(['0', '1', '2'])
    plt.title("Label 1" + title + "counts")
    plt.show()

# plot_label_counts(train_counts, "Train")
# plot_label_counts(test_counts, "Test")

path = r"C:\Users\linho\Desktop\CST\term4\pa\data\fluDiagnosis.csv"

df = pd.read_csv(path)

y = df['Diagnosed']
X = df.drop(columns=['Diagnosed'])

kfold = StratifiedKFold(n_splits=3, shuffle=True)

f1s = []

for train_idx, test_idx in kfold.split(X, y):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1s.append(f1_score(y_test, y_pred))

print(f'Average F1: {np.mean(f1s):.4f}')
print(f'F1 Std: {np.std(f1s):.4f}')