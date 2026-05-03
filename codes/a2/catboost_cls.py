import warnings

import numpy as np
import optuna
import pandas as pd
from optuna_integration import CatBoostPruningCallback
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
from catboost import CatBoostClassifier

warnings.filterwarnings(
    "ignore",
    message="CatBoostPruningCallback is experimental"
)

df = pd.read_csv("data/bankruptcy_asgn2.csv")
X = df.drop(columns=["Bankrupt?"])
y = df["Bankrupt?"]

model = CatBoostClassifier()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train_arr = np.array(X_train)
X_test_arr  = np.array(X_test)
y_train_arr = np.array(y_train)
y_test_arr  = np.array(y_test)

count_0 = (y_train_arr == 0).sum()
count_1 = (y_train_arr == 1).sum()
pos_weight = count_0 / count_1
class_weights = {0: 1, 1: pos_weight}  # same 30:1 ratio

# X_train_res, y_train_res = SMOTEENN(random_state=42).fit_resample(X_train_arr, y_train_arr)

def f1_class1_metric(y_pred, y_test):
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    return "f1_class1", f1

def objective(trial):
    params = {
        "iterations":           trial.suggest_int("iterations", 100, 1500),
        "learning_rate":        trial.suggest_float("learning_rate", 1e-4, 0.5, log=True),
        "depth":                trial.suggest_int("depth", 4, 10),
        "colsample_bylevel":    trial.suggest_float("colsample_bylevel", 0.01, 0.1, log=True),
        "boosting_type":        trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type":       trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
        "random_strength":      trial.suggest_float("random_strength", 0.3, 1.0),
        "objective":            "Logloss",
        "class_weights":        class_weights,
        "used_ram_limit":       "3gb",
        "eval_metric":          "F1"
    }

    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif params["bootstrap_type"] == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.1, 1.0, log=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_arr, y_train_arr)):
        X_tr, X_val = X_train_arr[tr_idx], X_train_arr[val_idx]
        y_tr, y_val = y_train_arr[tr_idx], y_train_arr[val_idx]

        gbm = CatBoostClassifier(**params)

        pruning_callback = CatBoostPruningCallback(
            trial,
            "F1",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The reported value is ignored")
            gbm.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=0,
                early_stopping_rounds=100,
                callbacks=[pruning_callback],
            )

        pruning_callback.check_pruned()

        y_prob = gbm.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
        f1_arr = 2 * (precision * recall) / (precision + recall + 1e-9)
        thresh = thresholds[np.argmax(f1_arr[:-1])]
        y_pred = (y_prob > thresh).astype(int)
        fold_scores.append(f1_score(y_val, y_pred, pos_label=1, zero_division=0))

    return np.mean(fold_scores)

study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    direction="maximize"
)
study.optimize(objective, n_trials=100, timeout=600)

print(f"Number of finished trials: {len(study.trials)}")

print("Best trial:")
trial = study.best_trial

print("    Value: {}".format(trial.value))
print("    Params")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# ------------ final model -------------------------------------
best = study.best_params.copy()

# re-add fixed params that were not in Optuna search space
best.pop("objective", None)
best["loss_function"] = "Logloss"
best["eval_metric"] = "F1"
best["class_weights"] = class_weights

model = CatBoostClassifier(**best, verbose=0)
model.fit(X_train_arr, y_train_arr)

y_prob = model.predict_proba(X_test_arr)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test_arr, y_prob)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_thresh = thresholds[np.argmax(f1_scores[:-1])]

print(f"Best threshold from PR curve: {best_thresh:.4f}")
y_pred = (y_prob > best_thresh).astype(int)
print(classification_report(y_test_arr, y_pred))
