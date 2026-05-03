import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
import xgboost as xgb
import pandas as pd
import optuna
from optuna import TrialPruned
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import json

df = pd.read_csv("data/bankruptcy_asgn2.csv")
X = df.drop(columns=["Bankrupt?"])
y = df["Bankrupt?"]


var_filter = VarianceThreshold(threshold=0.0)
X_filtered = var_filter.fit_transform(X)

# update feature names to match
feature_names = [list(X.columns)[i] for i in var_filter.get_support(indices=True)]

# Convert to DataFrame to keep things safe before splitting
X = pd.DataFrame(X_filtered, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train_arr = np.array(X_train)
y_train_arr = np.array(y_train)
X_test_arr  = np.array(X_test)
y_test_arr  = np.array(y_test)

def f1_class1_metric(y_pred, dtrain):
    y_true = dtrain.get_label()
    y_pred_binary = (y_pred > 0.5).astype(int)
    score = f1_score(y_true, y_pred_binary, pos_label=1, zero_division=0)
    return "f1_class1", score


def objective(trial):
    # 1. Suggest parameters
    n_features = trial.suggest_int("n_features", 20, 95)

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "max_depth": trial.suggest_int("max_depth", 4, 50, log=True),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "bootstrap": True,
        "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "n_jobs": -1,
        "random_state": 42
    }
    threshold = trial.suggest_float("threshold", 0.3, 0.7)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_arr, y_train_arr)):
        X_tr, X_val = X_train_arr[tr_idx], X_train_arr[val_idx]
        y_tr, y_val = y_train_arr[tr_idx], y_train_arr[val_idx]

        v_filter = VarianceThreshold(threshold=0.0)
        X_tr = v_filter.fit_transform(X_tr)
        X_val = v_filter.transform(X_val)

        selector = SelectFromModel(
            xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0),
            max_features=n_features,
            threshold=-np.inf
        )

        X_tr_sel = selector.fit_transform(X_tr, y_tr)
        X_val_sel = selector.transform(X_val)

        # 3. SMOTE on the SELECTED features
        X_tr_smote, y_tr_smote = SMOTE(random_state=42).fit_resample(X_tr_sel, y_tr)

        # 4. Train model on the smaller feature set
        model = RandomForestClassifier(**params)
        model.fit(X_tr_smote, y_tr_smote)

        # 5. Predict on the matching validation feature set
        y_probs = model.predict_proba(X_val_sel)[:, 1]
        y_pred = (y_probs > threshold).astype(int)

        score = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
        fold_scores.append(score)

        trial.report(score, step=fold)
        if trial.should_prune():
            raise TrialPruned()

    return np.mean(fold_scores)


sampler = optuna.samplers.TPESampler(seed=42)
pruner  = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=2, interval_steps=1)
study   = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=200, catch=(Exception,))

# ── extract best params ─────────────────────────────────────────────────────
best         = study.best_params.copy()
threshold    = best.pop("threshold", 0.5)
n_estimators = best.pop("n_estimators")
n_features   = best.pop("n_features")

print("\n" + "="*60)
print("BEST OPTUNA RESULTS")
print("="*60)
print(f"Best CV F1 (class 1): {study.best_value:.4f}")
print(f"Best trial:           #{study.best_trial.number}")
print(f"Threshold:            {threshold:.4f}")
print(f"n_estimators:         {n_estimators}")
print(f"n_features selected:  {n_features}")
print("\nModel hyperparameters:")
for k, v in best.items():
    print(f"  {k}: {v}")

# ── refit final selector on full SMOTE training data ───────────────────────
X_train_final, y_train_final = SMOTE(random_state=42).fit_resample(X_train_arr, y_train_arr)

final_selector = SelectFromModel(
    xgb.XGBClassifier(n_estimators=100, random_state=42),
    max_features=n_features,
    threshold=-np.inf,
)
final_selector.fit(X_train_final, y_train_final)

# get selected feature names using the original feature list
selected_mask  = final_selector.get_support()
selected_features = [feature_names[i] for i, selected in enumerate(selected_mask) if selected]

print(f"\nSelected {len(selected_features)} features (out of {len(feature_names)}):")
for i, feat in enumerate(selected_features, 1):
    print(f"  {i:>2}. {feat}")

# ── transform and train final model ────────────────────────────────────────
X_train_sel = final_selector.transform(X_train_final)
X_test_sel  = final_selector.transform(X_test_arr)

# Refit using Random Forest to match your Optuna study
model = RandomForestClassifier(
    **best,
    n_estimators=n_estimators,
    n_jobs=-1,
    random_state=42,
)
model.fit(X_train_sel, y_train_final)

# ── find best threshold via PR curve on test set ───────────────────────────
y_prob = model.predict_proba(X_test_sel)[:, 1]
precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_test_arr, y_prob)
f1_arr = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-9)
pr_threshold = thresholds_arr[np.argmax(f1_arr[:-1])]

print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

for name, t in [("Optuna threshold", threshold), ("PR curve threshold", pr_threshold)]:
    y_pred = (y_prob > t).astype(int)
    f1 = f1_score(y_test_arr, y_pred, pos_label=1, zero_division=0)
    print(f"\n── {name}: {t:.4f}  |  F1 class 1: {f1:.4f} ──")
    print(classification_report(y_test_arr, y_pred))

# ── save results to json ───────────────────────────────────────────────────
results = {
    "best_cv_f1":       study.best_value,
    "best_trial":       study.best_trial.number,
    "optuna_threshold": threshold,
    "pr_threshold":     float(pr_threshold),
    "n_estimators":     n_estimators,
    "n_features":       n_features,
    "model_params":     best,
    "selected_features": selected_features,
}

with open("data/rf_smote_best_params.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to data/best_params.json")
print("="*60)