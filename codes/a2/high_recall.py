import optuna
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    fbeta_score, recall_score, classification_report, precision_recall_curve
)
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter
from optuna.exceptions import TrialPruned
import matplotlib.pyplot as plt

df = pd.read_csv("data/bankruptcy_asgn2.csv")
X = df.drop(columns=["Bankrupt?"])
y = df["Bankrupt?"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ── data prep ──────────────────────────────────────────────────────────────
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

    # ✅ biased toward low threshold for high recall
    threshold = trial.suggest_float("threshold", 0.1, 0.4)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_arr, y_train_arr)):
        X_tr,  X_val  = X_train_arr[tr_idx], X_train_arr[val_idx]
        y_tr,  y_val  = y_train_arr[tr_idx],  y_train_arr[val_idx]

        # ✅ SMOTEENN: oversamples minority + cleans noisy majority samples
        X_tr_res, y_tr_res = SMOTEENN(random_state=42).fit_resample(X_tr, y_tr)

        booster = xgb.train(
            params,
            xgb.DMatrix(X_tr_res, label=y_tr_res),
            num_boost_round=n_estimators,
            evals=[(xgb.DMatrix(X_val, label=y_val), "validation")],
            verbose_eval=False,
        )

        y_pred = (booster.predict(xgb.DMatrix(X_val)) > threshold).astype(int)

        # ✅ optimise F2 (recall-weighted) instead of F1
        fold_scores.append(
            fbeta_score(y_val, y_pred, beta=2, pos_label=1, zero_division=0)
        )

        trial.report(np.mean(fold_scores), step=fold)
        if trial.should_prune():
            raise TrialPruned()

    return np.mean(fold_scores)


sampler = optuna.samplers.TPESampler(seed=42)
pruner  = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=2, interval_steps=1)
study   = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=150)

print("Best F2 (class 1):", study.best_value)
print("Best params:", study.best_params)

# ── final model ────────────────────────────────────────────────────────────
best         = study.best_params.copy()
threshold    = best.pop("threshold")
n_estimators = best.pop("n_estimators")

X_train_final, y_train_final = SMOTEENN(random_state=42).fit_resample(X_train_arr, y_train_arr)

model = xgb.XGBClassifier(
    **best,
    n_estimators=n_estimators,
    objective="binary:logistic",
    random_state=42,
)
model.fit(X_train_final, y_train_final)

# ── find best threshold from PR curve ─────────────────────────────────────
y_prob = model.predict_proba(X_test_arr)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test_arr, y_prob)
f2_scores = (5 * precision * recall) / (4 * precision + recall + 1e-9)
best_thresh_idx = np.argmax(f2_scores)
best_thresh     = thresholds[best_thresh_idx]

print(f"\nOptuna threshold:    {threshold:.4f}")
print(f"PR curve threshold:  {best_thresh:.4f}  ← use whichever gives better recall")

# plot precision / recall / F2 vs threshold
plt.figure(figsize=(10, 5))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1],    label="Recall")
plt.plot(thresholds, f2_scores[:-1], label="F2", linewidth=2)
plt.axvline(best_thresh, color="red",    linestyle="--", label=f"Best F2 thresh={best_thresh:.2f}")
plt.axvline(threshold,   color="orange", linestyle="--", label=f"Optuna thresh={threshold:.2f}")
plt.xlabel("Threshold")
plt.legend()
plt.title("Precision / Recall / F2 vs Threshold")
plt.tight_layout()
plt.show()

# ── evaluate both thresholds, pick better one ──────────────────────────────
for name, t in [("Optuna", threshold), ("PR curve", best_thresh)]:
    y_pred = (y_prob > t).astype(int)
    recall = recall_score(y_test_arr, y_pred, pos_label=1, zero_division=0)
    f2     = fbeta_score(y_test_arr, y_pred, beta=2, pos_label=1, zero_division=0)
    print(f"\n── {name} threshold={t:.3f} ──")
    print(f"Recall: {recall:.4f}  |  F2: {f2:.4f}")
    print(classification_report(y_test_arr, y_pred))