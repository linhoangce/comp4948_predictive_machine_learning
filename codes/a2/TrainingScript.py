import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import torch.nn as nn
import torch
import torch.optim as optim
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _get_data(path, return_X_y=False):
    df = pd.read_csv(path)
    X = df.drop(columns=["Bankrupt?"])
    y = df["Bankrupt?"]
    if return_X_y:
        return X, y
    return df

# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
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
                nn.Dropout(dropout)
            ]
            in_size = hidden_size
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def _make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

# def _prepare_nn_dataset(X_train, X_val, X_test, y_train, y_val, y_test,
#                         features, batch_size, shuffle):
#     X_train = X_train[features].to_numpy()
#     X_val = X_val[features].to_numpy()
#     X_test = X_test[features].to_numpy()
#     y_train = y_train.to_numpy()
#     y_val = y_val.to_numpy()
#     y_test = y_test.to_numpy()
#
#     count_0 = (y_train == 0).sum()
#     count_1 = (y_train == 1).sum()
#     pos_weight = torch.tensor([count_0 / count_1],
#                               dtype=torch.float32)
#
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.transform(X_val)
#     X_test_scaled = scaler.transform(X_test)
#
#     train_loader = _make_loader(X_train_scaled, y_train, batch_size, shuffle=shuffle)
#     val_loader = _make_loader(X_val_scaled, y_val, batch_size, shuffle=False)
#     test_loader = _make_loader(X_test_scaled, y_test, batch_size, shuffle=False)
#
#     return train_loader, val_loader, test_loader, scaler, pos_weight

def _prepare_xgb_dataset(X_train, X_val, X_test,
                         y_train, y_val, y_test, features):
    X_train = X_train[features]
    X_val = X_val[features]
    X_test = X_test[features]

    X_train_smote, y_train_smote = SMOTEENN(random_state=42).fit_resample(
        X_train, y_train
    )

    return X_train_smote, X_val, X_test, y_train_smote, y_val, y_test

def _prepare_rf_dataset(X_train, X_val, X_test,
                        y_train, y_val, y_test, features):
    X_train = X_train[features]
    X_val = X_val[features]
    X_test = X_test[features]

    X_train_smote, y_train_smote = SMOTE(random_state=42).fit_resample(
        X_train, y_train
    )

    return X_train_smote, X_val, X_test, y_train_smote, y_val, y_test

def _fit_xgboost_cls(X_train, y_train):
    model = xgb.XGBClassifier(
        max_depth=7,
        learning_rate=0.04808381637743888,
        subsample=0.9521533385305752,
        colsample_bytree=0.5132407670456729,
        reg_alpha=1.8891877473069133e-05,
        reg_lambda=2.448830695774989,
        min_child_weight=1,
        gamma=0.2190194401075139,
        n_estimators=439
    )
    model.fit(X_train, y_train)
    return model


def _fit_random_forest_cls(X_train, y_train):
    model = RandomForestClassifier(
        max_depth=20,
        max_features=0.9225715472584318,
        min_samples_split=16,
        min_samples_leaf=16,
        max_samples=0.8384988859968542,
        criterion="entropy",
        n_estimators=1075,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def _train_neural_net_cls(X_train, y_train, n_features, hidden_size,
                          n_layers, dropout, lr, weight_decay, epochs, threshold,
                          batch_size, X_val=None, y_val=None, verbose=False, log_steps=10):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    count_0 = (y_train == 0).sum()
    count_1 = (y_train == 1).sum()
    pos_weight = torch.tensor([count_0 / count_1], dtype=torch.float32)

    train_loader = _make_loader(X_train_scaled, y_train, batch_size, shuffle=True)

    val_loader = None
    if X_val is not None and y_val is not None:
        X_val_scaled = scaler.transform(X_val)
        val_loader = _make_loader(X_val_scaled, y_val, batch_size, shuffle=False)

    model = BankruptcyModel(
        n_features=n_features,
        hidden_size=hidden_size,
        n_layers=n_layers,
        dropout=dropout
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("\n" + "="*80)
    print(f"\t\tTRAINING NEURAL NETWORK ON {epochs} EPOCHS")
    print("="*80)
    for epoch in range(epochs):
        model.train()

        train_preds, val_preds = [], []
        train_labels, val_labels = [], []

        train_loss = 0

        for xb, yb in train_loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits)
            y_pred = (probs > threshold).int().cpu().numpy()
            train_loss += loss.item() * xb.size(0)
            train_preds.extend(y_pred.flatten().tolist())
            train_labels.extend(yb.cpu().numpy().flatten().tolist())

        scheduler.step()

        if verbose and (epoch + 1) % log_steps == 0:
            train_f1 = f1_score(train_labels, train_preds, pos_label=1, zero_division=0)
            train_loss_avg = train_loss / len(train_loader.dataset)
            log = (f"  Epoch {epoch+1}/{epochs} | "
                   f"Loss: {train_loss_avg:4f} | Train F1: {train_f1:4f}")

            if val_loader is not None:
                # ----- VALIDATION -----
                model.eval()
                val_pred, val_labels = [], []

                with torch.no_grad():
                    for xb, yb in val_loader:
                        probs = torch.sigmoid(model(xb))
                        y_pred = (probs > threshold).int().cpu().numpy()
                        val_preds.extend(y_pred.flatten().tolist())
                        val_labels.extend(yb.cpu().numpy().flatten().tolist())

                val_f1 = f1_score(val_labels, val_preds, pos_label=1, zero_division=0)
                log += f" | Val F1: {val_f1:.4f}"

                model.train()

            print(log)

    return model, scaler

def _get_nn_probs(model, scaler, X, batch_size=64):
    """
    Returns sigmoid probabilities from a trained NN.
    """
    X_scaled = scaler.transform(X)
    ds = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs = []
    model.eval()
    with torch.inference_mode():
        for (xb,) in loader:
            p = torch.sigmoid(model(xb)).cpu().numpy().flatten()
            probs.extend(p)
    return np.array(probs)

def evaluate_model(name, y_test, y_pred):
    print(f"\n {'='*30} {name} {'='*30}")
    print(classification_report(y_test, y_pred))


# ---------------------------------------------------------------------------
# OOF stacking
# ---------------------------------------------------------------------------
def _build_out_of_fold_predictions(X_train, y_train,
                                  nn_features, xgb_features, rf_features,
                                  nn_params, xgb_params, rf_params,
                                  n_splits=5, random_state=42):
    """
    Produces out-of-fold probability prediction for all three base models
    over the entire traiing set using StratifiedKFold.

    :return:
        oof_df : pd.DataFrame, shape (n_train, 3)
        Columns: NeuralNet, XGBoost, RandomForest — all probabilities.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_nn = np.zeros(len(X_train))
    oof_xgb = np.zeros(len(X_train))
    oof_rf = np.zeros(len(X_train))

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n--- OOF Fold {fold+1}/{n_splits} ---")

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # ======= NEURAL NET ========
        X_train_nn = X_tr[nn_features].to_numpy(dtype=np.float32)
        X_val_nn   = X_val[nn_features].to_numpy(dtype=np.float32)
        y_train_np = y_tr.to_numpy(dtype=np.float32)

        nn_model, nn_scaler = _train_neural_net_cls(
            X_train_nn, y_train_np,
            n_features=len(nn_features),
            **nn_params,
            verbose=False
        )

        oof_nn[val_idx] = _get_nn_probs(nn_model, nn_scaler, X_val_nn)

        # ====== XGBoost =========
        X_train_xgb = X_train[xgb_features]
        X_val_xgb   = X_val[xgb_features]
        X_train_xgb_smote, y_train_xgb_smote = SMOTEENN(random_state=42).fit_resample(
            X_train_xgb, y_train
        )

        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train_xgb_smote, y_train_xgb_smote)
        oof_xgb[val_idx] = xgb_model.predict_proba(X_val_xgb)[:, 1]

        # ======= RandomForest ========
        X_train_rf = X_train[rf_features]
        X_val_rf   = X_val[rf_features]
        X_train_rf_smote, y_train_rf_smote = SMOTE(random_state=42).fit_resample(
            X_train_rf, y_train
        )

        rf_model= RandomForestClassifier(**rf_params)
        rf_model.fit(X_train_rf_smote, y_train_rf_smote)
        oof_rf[val_idx] = rf_model.predict_proba(X_val_rf)[:, 1]

    oof_df = pd.DataFrame({
        "NeuralNet":    oof_nn,
        "XGBoost":      oof_xgb,
        "RandomForest": oof_rf
    })
    return oof_df


def main():
    PATH = "data/bankruptcy_asgn2.csv"

    with open("models/selected_features.json", "r") as f:
        features = json.load(f)

    NN_FEATURES = features["nn_features"]
    XGB_FEATURES = features["xgb_features"]
    RF_FEATURES = features["rf_features"]

    NN_PARAMS = dict(
        hidden_size=32,
        n_layers=2,
        dropout=0.12851924650459373,
        lr=0.00015590984928259534,
        weight_decay=0.0017637726683144917,
        batch_size=32,
        epochs=200,
        threshold=0.8822
    )

    XGB_PARAMS = dict(
        max_depth=7,
        learning_rate=0.04808381637743888,
        subsample=0.9521533385305752,
        colsample_bytree=0.5132407670456729,
        reg_alpha=1.8891877473069133e-05,
        reg_lambda=2.448830695774989,
        min_child_weight=1,
        gamma=0.2190194401075139,
        n_estimators=439
    )
    RF_PARAMS = dict(
        max_depth=20,
        max_features=0.9225715472584318,
        min_samples_split=16,
        min_samples_leaf=16,
        max_samples=0.8384988859968542,
        criterion="entropy",
        n_estimators=1075,
        n_jobs=-1,
        random_state=42
    )

    XGB_THRESHOLD = 0.6875752500686682
    RF_THRESHOLD  = 0.6751


    # ----------- Load and clean data --------------------------------------------------
    X, y = _get_data(PATH, return_X_y=True)
    full_features = list(X.columns)

    # remove constant features to avoid division by zero
    var_filter = VarianceThreshold(threshold=0.0)
    X_filtered = var_filter.fit_transform(X)
    feature_names = [full_features[i] for i in var_filter.get_support(indices=True)]
    X = pd.DataFrame(X_filtered, columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # =========================================================
    # STEP 1 — OOF predictions on full training set
    # =========================================================
    print(f"\n {'='*80}")
    print("STEP 1: Building OOF Predictions (5 Folds)")
    print("="*80)

    oof_df = _build_out_of_fold_predictions(
        X_train, y_train,
        nn_features=NN_FEATURES,
        xgb_features=XGB_FEATURES,
        rf_features=RF_FEATURES,
        nn_params=NN_PARAMS,
        xgb_params=XGB_PARAMS,
        rf_params=RF_PARAMS,
        n_splits=5
    )

    # =========================================================
    # STEP 2 — Fit meta-learner on OOF probabilities
    # =========================================================
    print(f"\n {'=' * 80}")
    print("STEP 1: Fitting Stacking Meta-Learner on OOF Predictions")
    print("=" * 80)

    stacked_model = LogisticRegression()
    stacked_model.fit(oof_df, y_train)

    # quick sanity check
    oof_stacked_pred = stacked_model.predict(oof_df)
    print("\nOOF Stacked Model Performance (optimistic upper bound):")
    print(classification_report(y_train, oof_stacked_pred, zero_division=0))

    # =========================================================
    # STEP 3 — Retrain all base models on the FULL training set
    # =========================================================
    print(f"\n {'=' * 80}")
    print("STEP 1: Retraining Base Models on Full Training Dataset")
    print("=" * 80)

    # --------------- NeuralNet ------------------------------
    print("\nTraining final Neural Net....")
    X_train_nn = X_train[NN_FEATURES].to_numpy(dtype=np.float32)
    y_train_np = y_train.to_numpy(dtype=np.float32)
    final_nn, final_nn_scaler = _train_neural_net_cls(
        X_train_nn, y_train_np,
        n_features=len(NN_FEATURES),
        **NN_PARAMS,
        verbose=True,
        log_steps=1
    )

    # -------------- XGBoost ---------------------------------
    print("\nTraining final XGBoost...")
    X_train_xgb =X_train[XGB_FEATURES]
    X_train_xgb_smote, y_train_xgb_smote = SMOTEENN(random_state=42).fit_resample(X_train_xgb, y_train)
    final_xgb = xgb.XGBClassifier(**XGB_PARAMS)
    final_xgb.fit(X_train_xgb_smote, y_train_xgb_smote)

    # -------------- RandomForest ----------------------------
    print("\nTraining final RandomForest...")
    X_train_rf = X_train[RF_FEATURES]
    X_train_rf_smote, y_train_rf_smote = SMOTE(random_state=42).fit_resample(X_train_rf, y_train)
    final_rf = RandomForestClassifier(**RF_PARAMS)
    final_rf.fit(X_train_rf_smote, y_train_rf_smote)

    # =========================================================
    # STEP 4 — Evaluate on test set
    # =========================================================
    print(f"\n {'=' * 80}")
    print("STEP 1: Evaluating On Test")
    print("=" * 80)

    X_test_nn = X_test[NN_FEATURES].to_numpy(dtype=np.float32)
    X_test_xgb = X_test[XGB_FEATURES]
    X_test_rf = X_test[RF_FEATURES]

    nn_prob_test = _get_nn_probs(final_nn, final_nn_scaler, X_test_nn)
    xgb_prob_test = final_xgb.predict_proba(X_test_xgb)[:, 1]
    rf_prob_test = final_rf.predict_proba(X_test_rf)[:, 1]

    nn_pred_test = (nn_prob_test > NN_PARAMS["threshold"]).astype(int)
    xgb_pred_test = (xgb_prob_test > XGB_THRESHOLD).astype(int)
    rf_pred_test = (rf_prob_test > RF_THRESHOLD).astype(int)

    evaluate_model("Neural Net", y_test, nn_pred_test)
    evaluate_model("XGBoost", y_test, xgb_pred_test)
    evaluate_model("Random Forest", y_test, rf_pred_test)

    # Stacked model
    df_test_probs = pd.DataFrame({
        "NeuralNet":    nn_prob_test,
        "XGBoost":      xgb_prob_test,
        "RandomForest": rf_prob_test
    })

    y_pred_stacked = stacked_model.predict(df_test_probs)
    evaluate_model("Stacked (LogReg)", y_test, y_pred_stacked)

    # =========================================================
    # STEP 6 — Retrain final production models on FULL dataset
    # =========================================================
    print("\n" + "=" * 80)
    print("STEP 6 — Retraining production models on full dataset (train + test)")
    print("=" * 80)

    X_full = pd.concat([X_train, X_test]).reset_index(drop=True)
    y_full = pd.concat([y_train, y_test]).reset_index(drop=True)

    # --- Neural Net ---
    print("\nTraining production Neural Net...")
    X_full_nn = X_full[NN_FEATURES].to_numpy(dtype=np.float32)
    y_full_np = y_full.to_numpy(dtype=np.float32)
    prod_nn, prod_nn_scaler = _train_neural_net_cls(
        X_full_nn, y_full_np,
        n_features=len(NN_FEATURES),
        **NN_PARAMS,
        verbose=True,
    )

    # --- XGBoost ---
    print("\nTraining production XGBoost...")
    X_full_xgb = X_full[XGB_FEATURES]
    X_full_xgb_res, y_full_xgb_res = SMOTEENN(random_state=42).fit_resample(X_full_xgb, y_full)
    prod_xgb = xgb.XGBClassifier(**XGB_PARAMS)
    prod_xgb.fit(X_full_xgb_res, y_full_xgb_res)

    # --- Random Forest ---
    print("\nTraining production Random Forest...")
    X_full_rf = X_full[RF_FEATURES]
    X_full_rf_res, y_full_rf_res = SMOTE(random_state=42).fit_resample(X_full_rf, y_full)
    prod_rf = RandomForestClassifier(**RF_PARAMS)
    prod_rf.fit(X_full_rf_res, y_full_rf_res)

    # --- OOF on full dataset to refit meta-learner ---
    print("\nBuilding OOF predictions on full dataset for meta-learner...")
    oof_full_df = _build_out_of_fold_predictions(
        X_full, y_full,
        nn_features=NN_FEATURES,
        xgb_features=XGB_FEATURES,
        rf_features=RF_FEATURES,
        nn_params=NN_PARAMS,
        xgb_params=XGB_PARAMS,
        rf_params=RF_PARAMS,
        n_splits=5,
    )
    prod_stacked = LogisticRegression()
    prod_stacked.fit(oof_full_df, y_full)

    # --- Save production models ---
    joblib.dump(prod_nn_scaler, "model_200/prod_nn_scaler.joblib")
    joblib.dump(var_filter, "model_200/prod_var_filter.joblib")
    joblib.dump(prod_xgb, "model_200/prod_xgb_classifier.joblib")
    joblib.dump(prod_rf, "model_200/prod_rf_classifier.joblib")
    joblib.dump(prod_stacked, "model_200/prod_stacked_logreg.joblib")
    torch.save(prod_nn.state_dict(), "model_200/prod_bankruptcy_nn_classifier.pt")

    print("\nProduction models saved to models/")


if __name__ == "__main__":
    main()