import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import xgboost as xgb
import optuna
from optuna.exceptions import TrialPruned
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel, VarianceThreshold
from sklearn.metrics import f1_score, precision_recall_curve, classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

optuna.logging.set_verbosity(optuna.logging.WARNING)

# -- data loading ------------------------------------------------------------
df = pd.read_csv("data/bankruptcy_asgn2.csv")
X  = df.drop(columns=["Bankrupt?"])
y  = df["Bankrupt?"]

feature_names = list(X.columns)

var_filter = VarianceThreshold(threshold=0.0)
X_filtered = var_filter.fit_transform(X)

# update feature names to match
feature_names = [feature_names[i] for i in var_filter.get_support(indices=True)]
X = pd.DataFrame(X_filtered, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train_arr = np.array(X_train, dtype=np.float32)
y_train_arr = np.array(y_train, dtype=np.float32)
X_test_arr  = np.array(X_test,  dtype=np.float32)
y_test_arr  = np.array(y_test,  dtype=np.float32)

n_total_features = X_train_arr.shape[1]
print(f"Total features: {n_total_features}")
print(f"Class distribution - 0: {(y_train_arr == 0).sum()}, 1: {(y_train_arr == 1).sum()}")


# -- model definition --------------------------------------------------------
class BankruptcyModel(nn.Module):
    def __init__(self, n_features, hidden_size, n_layers, dropout):
        super().__init__()
        layers = []
        in_size = n_features
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_size = hidden_size
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -- helpers -----------------------------------------------------------------
def make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def get_selector(selector_type, n_features):
    if selector_type == "f_classif":
        return SelectKBest(score_func=f_classif, k=n_features)
    elif selector_type == "mutual_info":
        return SelectKBest(score_func=mutual_info_classif, k=n_features)
    elif selector_type == "xgboost":
        return SelectFromModel(
            xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0),
            max_features=n_features,
            threshold=-np.inf,
        )
    else:
        raise ValueError(f"Unknown selector_type: {selector_type}")


def train_fold(trial, params, X_tr, y_tr, X_val, y_val, fold=0):
    # scale inside fold to avoid leakage
    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    # feature selection on real imbalanced data before SMOTE
    selector = get_selector(params["selector_type"], params["n_features"])
    selector.fit(X_tr, y_tr)
    X_tr  = selector.transform(X_tr)
    X_val = selector.transform(X_val)

    # SMOTE only on training fold — val stays real and imbalanced
    X_tr_res, y_tr_res = SMOTE(random_state=42).fit_resample(X_tr, y_tr)

    train_loader = make_loader(X_tr_res, y_tr_res, params["batch_size"], shuffle=True)
    val_loader   = make_loader(X_val,    y_val,     params["batch_size"], shuffle=False)

    model = BankruptcyModel(
        n_features=params["n_features"],
        hidden_size=params["hidden_size"],
        n_layers=params["n_layers"],
        dropout=params["dropout"],
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"])
    loss_fn   = nn.BCEWithLogitsLoss()  # no pos_weight — SMOTE handles imbalance

    best_val_f1 = 0.0
    patience    = 10
    no_improve  = 0

    for epoch in range(params["epochs"]):
        # training step
        model.train()
        for xb, yb in train_loader:
            logits = model(xb)
            loss   = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # validation step
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                probs  = torch.sigmoid(model(xb))
                y_pred = (probs > params["threshold"]).int()
                val_preds.extend(y_pred.cpu().numpy().flatten())
                val_labels.extend(yb.cpu().numpy().flatten())

        val_f1 = f1_score(val_labels, val_preds, pos_label=1, zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve  = 0
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


# -- Optuna objective --------------------------------------------------------
def objective(trial):
    params = {
        "hidden_size":   trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
        "n_layers":      trial.suggest_int("n_layers", 1, 4),
        "dropout":       trial.suggest_float("dropout", 0.1, 0.5),
        "lr":            trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay":  trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size":    trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "epochs":        trial.suggest_int("epochs", 30, 150),
        "threshold":     trial.suggest_float("threshold", 0.2, 0.7),
        "n_features":    trial.suggest_int("n_features", 10, n_total_features),
        "selector_type": trial.suggest_categorical(
                             "selector_type", ["f_classif", "mutual_info", "xgboost"]
                         ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_arr, y_train_arr)):
        X_tr,  X_val = X_train_arr[tr_idx], X_train_arr[val_idx]
        y_tr,  y_val = y_train_arr[tr_idx],  y_train_arr[val_idx]

        try:
            val_f1, _, _ = train_fold(trial, params, X_tr, y_tr, X_val, y_val, fold=fold)
        except TrialPruned:
            raise

        fold_scores.append(val_f1)

    # return mean F1 across all folds
    return np.mean(fold_scores)


# -- run study ---------------------------------------------------------------
sampler = optuna.samplers.TPESampler(seed=42)
pruner  = optuna.pruners.MedianPruner(
    n_startup_trials=10,
    n_warmup_steps=20,
    interval_steps=5,
)
study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=100, catch=(Exception,))

# -- extract best params -----------------------------------------------------
best          = study.best_params.copy()
threshold     = best.pop("threshold")
epochs        = best.pop("epochs")
n_features    = best.pop("n_features")
selector_type = best.pop("selector_type")

print("\n" + "=" * 60)
print("BEST OPTUNA RESULTS")
print("=" * 60)
print(f"Best CV F1 (class 1): {study.best_value:.4f}")
print(f"Best trial:           #{study.best_trial.number}")
print(f"Selector type:        {selector_type}")
print(f"n_features selected:  {n_features}")
print(f"Threshold:            {threshold:.4f}")
print(f"Epochs:               {epochs}")
print("Model hyperparameters:")
for k, v in best.items():
    print(f"  {k}: {v}")

# -- final model on full training data ---------------------------------------
scaler_final   = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train_arr)
X_test_scaled  = scaler_final.transform(X_test_arr)

# refit selector on full scaled training data
final_selector = get_selector(selector_type, n_features)
final_selector.fit(X_train_scaled, y_train_arr)
X_train_sel = final_selector.transform(X_train_scaled)
X_test_sel  = final_selector.transform(X_test_scaled)

# print selected feature names
selected_mask     = final_selector.get_support()
selected_features = [feature_names[i] for i, s in enumerate(selected_mask) if s]

print(f"\nSelected {len(selected_features)} features (out of {n_total_features}):")
for i, feat in enumerate(selected_features, 1):
    print(f"  {i:>2}. {feat}")

# SMOTE on full selected training data — val/test stay real
X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train_sel, y_train_arr)

final_model = BankruptcyModel(
    n_features=n_features,
    hidden_size=best["hidden_size"],
    n_layers=best["n_layers"],
    dropout=best["dropout"],
)
optimizer_final = optim.Adam(
    final_model.parameters(),
    lr=best["lr"],
    weight_decay=best["weight_decay"],
)
scheduler_final = optim.lr_scheduler.CosineAnnealingLR(optimizer_final, T_max=epochs)
loss_fn_final   = nn.BCEWithLogitsLoss()

train_loader_final = make_loader(X_train_res, y_train_res, best["batch_size"])

print("\nTraining final model...")
final_model.train()
for epoch in range(epochs):
    for xb, yb in train_loader_final:
        logits = final_model(xb)
        loss   = loss_fn_final(logits, yb)
        optimizer_final.zero_grad()
        loss.backward()
        optimizer_final.step()
    scheduler_final.step()
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch + 1}/{epochs}")

# -- threshold search via PR curve -------------------------------------------
final_model.eval()
all_probs = []
with torch.no_grad():
    for (xb,) in DataLoader(
        TensorDataset(torch.tensor(X_test_sel, dtype=torch.float32)),
        batch_size=256,
    ):
        probs = torch.sigmoid(final_model(xb)).cpu().numpy().flatten()
        all_probs.extend(probs)

y_prob = np.array(all_probs)
precision_arr, recall_arr, thresh_arr = precision_recall_curve(y_test_arr, y_prob)
f1_arr    = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-9)
pr_thresh = float(thresh_arr[np.argmax(f1_arr[:-1])])

# -- final evaluation --------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

for name, t in [("Optuna threshold", threshold), ("PR curve threshold", pr_thresh)]:
    y_pred = (y_prob > t).astype(int)
    f1     = f1_score(y_test_arr, y_pred, pos_label=1, zero_division=0)
    print(f"\n-- {name}: {t:.4f}  |  F1 class 1: {f1:.4f} --")
    print(classification_report(y_test_arr, y_pred))

# -- save results ------------------------------------------------------------
results = {
    "best_cv_f1":        study.best_value,
    "best_trial":        study.best_trial.number,
    "optuna_threshold":  threshold,
    "pr_threshold":      pr_thresh,
    "epochs":            epochs,
    "n_features":        n_features,
    "selector_type":     selector_type,
    "model_params":      best,
    "selected_features": selected_features,
}

with open("data/best_params_nn.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to data/best_params_nn.json")
print("=" * 60)