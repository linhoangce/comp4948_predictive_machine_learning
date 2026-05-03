import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import optuna
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel
from sklearn.metrics import f1_score, precision_recall_curve, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from optuna.exceptions import TrialPruned

optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_csv("data/bankruptcy_asgn2.csv")
X = df.drop(columns=["Bankrupt?"])
y = df["Bankrupt?"]

feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train_arr = np.array(X_train)
X_test_arr = np.array(X_test)
y_train_arr = np.array(y_train)
y_test_arr = np.array(y_test)

n_total_features = X_train_arr.shape[0]
count_0 = (y_train_arr == 0).sum()
count_1 = (y_train_arr == 1).sum()
print(f"Total features: {n_total_features}")
print(f"Class distribution - 0: {count_0}, 1: {count_1}")

# -- model definition --------------------------------------------------------
class BankruptcyModel(nn.Module):
    def __init__(self, n_features, hidden_size, n_layers, dropout):
        super().__init__()
        layers = []
        in_size = n_features
        for _ in range(n_layers):
            layers += [
                nn.Linear(n_features, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -- helpers -----------------------------------------------------------------
def make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def get_selector(selector_type, n_features):
    if selector_type == "f_classif":
        return SelectKBest(score_func=f_classif, k=n_features)
    elif selector_type == "mutual_info":
        return SelectKBest(score_func=mutual_info_classif, k=n_features)
    elif selector_type == "xgboost":
        return SelectFromModel(
            xgb.XGBClassifier(n_estimators=200, random_state=42, verbosity=0),
            max_features=n_features,
            threshold=-np.inf
        )
    else:
        raise ValueError(f"Unknown selector_type: {selector_type}")

def train_fold(trial, params, X_tr, y_tr, X_val, y_val, fold=0):
    # scale inside fold to avoid leakage
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    # feature selection on real imbalanced data before SMOTE
    selector = get_selector(params["selector_type"], params["n_features"])
    selector.fit(X_tr, y_tr)
    X_tr = selector.transform(X_tr)
    X_val = selector.transform(X_val)

    X_tr_smote, y_tr_smote = SMOTEENN(random_state=42).fit_resample(X_tr, y_tr)

    train_loader = make_loader(X_tr_smote, y_tr_smote, params["batch_size"], shuffle=True)
    val_loader = make_loader(X_val, y_val, params["batch_size"], shuffle=True)

    model = BankruptcyModel(
        n_features=params["n_features"],
        hidden_size=params["hidden_size"],
        n_layers=params["n_layers"],
        dropout=params["dropout"]
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"])
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_f1 = 0.0
    patience = 10
    no_improve = 0

    for epoch in range(params["epochs"]):
        model.train()

        for xb, yb in train_loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # validation step
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                probs = torch.sigmoid(model(xb))
                y_pred = (probs > params["threshold"]).int()
                val_preds.extend(y_pred.cpu().numpy().flatten())
                val_labels.extend(yb.cpu().numpy().flatten())

        val_f1 = f1_score(val_labels, val_preds, pos_label=1, zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve = 0
        else:
            no_improve += 1

        # report to Optuna using global step to avoid duplicate step error
        if trial is not None:
            global_step = fold * params["epochs"] + epoch
            trial.report(val_f1, step=global_step)
            if trial.should_prune():
                raise TrialPruned()

        if no_improve >= patience:
            break

    return best_val_f1, scaler, selector

4