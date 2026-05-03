import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/bankruptcy_asgn2.csv")
X = df.drop(columns=["Bankrupt?"])
y = df["Bankrupt?"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train_arr = np.array(X_train)
y_train_arr = np.array(y_train)
X_test_arr = np.array(X_test)
y_test_arr = np.array(y_test)
# ✅ train ONLY on normal class (class 0)
X_train_normal = X_train_arr[y_train_arr == 0]

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.032,  # approximate % of anomalies = 220/6819 ≈ 0.032
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train_normal)

# predict: Isolation Forest returns 1 (normal) or -1 (anomaly)
# convert to 0 (normal) and 1 (anomaly) to match your labels
y_pred_raw = iso_forest.predict(X_test_arr)
y_pred = (y_pred_raw == -1).astype(int)

print(classification_report(y_test_arr, y_pred))
print("F1 class 1:", f1_score(y_test_arr, y_pred, pos_label=1))