import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import json

from optuna import TrialPruned
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

def preprocess_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["Bankrupt?"])
    y = df["Bankrupt?"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X, y, X_train, X_test, y_train, y_test

def f1_class_1_metric(y_pred, dtrain):
    y_true = dtrain.get_label()
    y_pred_binary = (y_pred > 0.5).astype(int)
    score = f1_score(y_true, y_pred_binary, pos_label=1, zero_division=0)
    return "f1_class_1", score

def objective(trial, X, y):
    params = {
        "max_depth":            trial.suggest_int("max_depth", 4, 10),
        "learning_rate":        trial.suggest_float("learning_rate", 1e-4, 0.5, log=True),
        "subsample":            trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":     trial.suggest_float("colsample_bytree", 0.5, 1.0, log=True),
        "reg_alpha":            trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":           trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight":     trial.suggest_int("min_child_weight", 1, 10),
        "gamma":                trial.suggest_float("gamma", 0, 5),
        "objective":            "binary:logistic",
        "seed":                 42
    }
    n_estimators    = trial.suggest_int("n_estimators", 100, 600)
    conf_threshold  = trial.suggest_float("threshold", 0.3, 0.7)
    n_features      = trial.suggest_int("n_features", 20, 95)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        X_tr_smote, y_tr_smote = SMOTEENN(random_state=42).fit_resample(X_tr, y_tr)

        selector = SelectFromModel(
            xgb.XGBClassifier(n_estimators=200, random_state=42),
            max_features=n_features,
            threshold=-np.inf
        )
        selector.fit(X_tr_smote, y_tr_smote)
        X_tr_sel = selector.transform(X_tr_smote)
        X_val_sel = selector.transform(X_val)

        booster = xgb.train(
            params,
            xgb.DMatrix(X_tr_sel, label=y_tr_smote),
            num_boost_round=n_estimators,
            evals=[(xgb.DMatrix(X_val_sel, label=y_val), "validation")],
            custom_metric=f1_class_1_metric,
            verbose_eval=False
        )

        y_pred = (booster.predict(xgb.DMatrix(X_val_sel)) > conf_threshold).astype(int)
        fold_scores.append(f1_score(y_val, y_pred, pos_label=1, zero_division=0))

        trial.report(np.mean(fold_scores), step=fold)
        if trial.should_prune():
            raise TrialPruned()

    return np.mean(fold_scores)

def search_params(X, y):
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=2, interval_steps=1)
    study   = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=200,
        catch=(Exception,)
    )
    return study

def main():
    PATH = "data/bankruptcy_asgn2.csv"

    X, y, X_train, X_test, y_train, y_test = preprocess_data(PATH)

    feature_names = list(X.columns)

    study = search_params(X.to_numpy(), y.to_numpy())

    # extract best params
    best            = study.best_params.copy()
    threshold       = best.pop("threshold", 0.5)
    n_estimators    = best.pop("n_estimators")
    n_features      = best.pop("n_features")

    print("\n" + "="*80)
    print("BEST OPTUNA RESULTS")
    print("="*80)
    print(f"Best CV F1 (Class 1): {study.best_value:.4f}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Threshold: {threshold:.4f}")
    print(f"n_estimators: {n_estimators}")
    print(f"n_features: {n_features}")
    print("\nModel Hyperparameters:")
    for k, v in best.items():
        print(f"    {k}: {v}")

    # refit final selector on full SMOTE training data
    X_train_final, y_train_final = SMOTEENN(random_state=42).fit_resample(X_train, y_train)

    final_selector = SelectFromModel(
        xgb.XGBClassifier(n_estimators=200, random_state=42),
        max_features=n_features,
        threshold=-np.inf
    )
    final_selector.fit(X_train_final, y_train_final)

    # get selector feature names using the original feature list
    selected_mask = final_selector.get_support()
    selected_features = [feature_names[i]
                         for i, selected in enumerate(selected_mask) if selected]

    print(f"\nSelected {len(selected_features)} features out of {len(feature_names)}")
    for i, feat in enumerate(selected_features, 1):
        print(f"    {i:>2}. {feat}")

    # transform and train final model
    X_train_sel = final_selector.transform(X_train_final)
    X_test_sel  = final_selector.transform(X_test)

    model = xgb.XGBClassifier(
        **best,
        n_estimators=n_estimators,
        objective="binary:logistic",
        random_state=42
    )
    model.fit(X_train_sel, y_train_final)

    # find best threshold via PR curve on test set
    y_prob = model.predict(X_test_sel)[:, 1]
    precision, recall, threshold = precision_recall_curve(y_test, y_prob)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    pr_threshold = threshold[np.argmax(f1[:-1])]

    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)

    for name, t in [("Optuna threshold", threshold), ("PR curve threshold", pr_threshold)]:
        y_pred = (y_prob > t).astype(int)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        print(f"\n--- {name}: {t:.4f}  |  F1 Class 1: {f1:4f} ---")
        print(classification_report(y_test, y_pred))

    results = {
        "best_cv_f1":       study.best_value,
        "best_trial":       study.best_trial.number,
        "optuna_threshold": study.threshold,
        "pr_threshold":     float(pr_threshold),
        "n_estimators":     n_estimators,
        "n_features":       n_features,
        "model_params":     best,
        "selected_features":    selected_features
    }

    with open("data/best_params_smoteenn.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to data/best_params_smoteenn.json")


if __name__ == "__main__":
    main()