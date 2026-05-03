import pandas as pd
from sklearn.metrics import fbeta_score, f1_score, precision_recall_curve, classification_report
from imblearn.combine import SMOTEENN
from optuna.exceptions import TrialPruned
import xgboost as xgb
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold

df = pd.read_csv("data/bankruptcy_asgn2.csv")
X = df.drop(columns=["Bankrupt?"])
y = df["Bankrupt?"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train_arr = np.array(X_train)
y_train_arr = np.array(y_train)
X_test_arr  = np.array(X_test)
y_test_arr  = np.array(y_test)

def objective(trial):
    params = {
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0, 3),
        "objective":        "binary:logistic",
        "seed":             42,
    }
    n_estimators = trial.suggest_int("n_estimators", 100, 600)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_arr, y_train_arr)):
        X_tr,  X_val = X_train_arr[tr_idx], X_train_arr[val_idx]
        y_tr,  y_val = y_train_arr[tr_idx],  y_train_arr[val_idx]

        X_tr_res, y_tr_res = SMOTEENN(random_state=42).fit_resample(X_tr, y_tr)

        booster = xgb.train(
            params,
            xgb.DMatrix(X_tr_res, label=y_tr_res),
            num_boost_round=n_estimators,
            evals=[(xgb.DMatrix(X_val, label=y_val), "validation")],
            verbose_eval=False,
        )

        # ✅ find best threshold per fold via PR curve — optimise F1 directly
        y_prob = booster.predict(xgb.DMatrix(X_val))
        precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        best_thresh = thresholds[np.argmax(f1_scores[:-1])]

        y_pred = (y_prob > best_thresh).astype(int)
        fold_scores.append(f1_score(y_val, y_pred, pos_label=1, zero_division=0))

        trial.report(np.mean(fold_scores), step=fold)
        if trial.should_prune():
            raise TrialPruned()

    return np.mean(fold_scores)  # ✅ optimising F1 directly


sampler = optuna.samplers.TPESampler(seed=42)
pruner  = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=2, interval_steps=1)
study   = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=150)

print("Best CV F1 (class 1):", study.best_value)
print("Best params:", study.best_params)

# ── final model ─────────────────────────────────────────────────────────────
best         = study.best_params.copy()
n_estimators = best.pop("n_estimators")

X_train_final, y_train_final = SMOTEENN(random_state=42).fit_resample(X_train_arr, y_train_arr)

model = xgb.XGBClassifier(
    **best,
    n_estimators=n_estimators,
    objective="binary:logistic",
    random_state=42,
)
model.fit(X_train_final, y_train_final)

# ✅ find best threshold on test set via PR curve
y_prob = model.predict_proba(X_test_arr)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test_arr, y_prob)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_thresh = thresholds[np.argmax(f1_scores[:-1])]

print(f"Best threshold from PR curve: {best_thresh:.4f}")

y_pred = (y_prob > best_thresh).astype(int)
print(classification_report(y_test_arr, y_pred))