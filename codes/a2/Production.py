import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

from TrainingScript import BankruptcyModel, _get_nn_probs


def _load_models(model_dir="model_200"):
    xgb_model = joblib.load(f"{model_dir}/prod_xgb_classifier.joblib")
    rf_model = joblib.load(f"{model_dir}/prod_rf_classifier.joblib")
    stacked_model = joblib.load(f"{model_dir}/prod_stacked_logreg.joblib")
    nn_scaler = joblib.load(f"{model_dir}/prod_nn_scaler.joblib")
    var_filter = joblib.load(f"{model_dir}/prod_var_filter.joblib")
    return xgb_model, rf_model, stacked_model, nn_scaler, var_filter


def _load_nn(model_dir, n_features, hidden_size, n_layers, dropout):
    model = BankruptcyModel(n_features, hidden_size, n_layers, dropout)
    model.load_state_dict(torch.load(
        f"{model_dir}/prod_bankruptcy_nn_classifier.pt",
        map_location="cpu"
    ))
    model.eval()
    return model


def predict(file_path,
            feature_path="model_200/selected_features.json",
            models_dir="model_200"):
    # -------- Load Features ---------------------------------------
    with open(feature_path, "r") as f:
        features = json.load(f)

    NN_FEATURES = features["nn_features"]
    XGB_FEATURES = features["xgb_features"]
    RF_FEATURES = features["rf_features"]

    # -------- Load & Clean Input ----------------------------------
    df = pd.read_csv(file_path)
    y_true = df["Bankrupt?"]

    if "Bankrupt?" in df.columns:
        df = df.drop(columns=["Bankrupt?"])

    # -------- Load Models ------------------------------------------
    xgb_model, rf_model, stacked_model, nn_scaler, var_filter = _load_models(models_dir)

    nn_model = _load_nn(
        models_dir,
        n_features=len(NN_FEATURES),
        hidden_size=32,
        n_layers=2,
        dropout=0.12851924650459373,
    )

    full_features = list(df.columns)
    X_filtered = var_filter.transform(df)
    feature_names = [full_features[i] for i in var_filter.get_support(indices=True)]
    X = pd.DataFrame(X_filtered, columns=feature_names)

    # -------- Predict probabilities with Base Models ----------------
    xgb_probs = xgb_model.predict_proba(X[XGB_FEATURES])[:, 1]
    rf_probs = rf_model.predict_proba(X[RF_FEATURES])[:, 1]
    nn_probs = _get_nn_probs(nn_model, nn_scaler,
                             X[NN_FEATURES].to_numpy(dtype=np.float32))

    # -------- Stacked Predictions -----------------------------------
    df_probs = pd.DataFrame({
        "NeuralNet": nn_probs,
        "XGBoost": xgb_probs,
        "RandomForest": rf_probs
    })

    predictions = stacked_model.predict(df_probs)
    probabilities = stacked_model.predict_proba(df_probs)[:, 1]

    result = pd.DataFrame({
        "Bankrupt_Predicted": predictions,
        "Bankruptcy_Probability": probabilities,
        "NN_Prob": nn_probs,
        "XGB_Prob": xgb_probs,
        "RF_Prob": rf_probs
    })

    return result, y_true

if __name__ == "__main__":
    FILE_PATH = "data/bankruptcy_mystery.csv"

    results, y_true = predict(FILE_PATH)

    print(results)
    y_pred = results["Bankrupt_Predicted"]
    print(classification_report(y_true, y_pred))

    results.to_csv("data/predictions.csv")
    print(f"Predictions saved to data/")