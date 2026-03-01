import joblib
import pandas as pd
import numpy as np

def bin_income_np(X):
    s = pd.Series(np.asarray(X).reshape(-1))
    cut = pd.cut(s, bins=[-np.inf, 55, 75, 95, np.inf])
    dummies = pd.get_dummies(cut, dtype=int)

    # lock column order
    cats = pd.IntervalIndex.from_breaks([-np.inf, 55, 75, 95, np.inf])
    dummies = dummies.reindex(columns=cats, fill_value=0)
    return dummies.to_numpy()

def bin_days_since_last_purchase(X):
    x = np.asarray(X, dtype=int).reshape(-1)
    bins = [0, 7, 30, 90, np.inf]
    bin_idx = np.digitize(x, bins[1:-1], right=True)

    one_hot = np.eye(len(bins)-1, dtype=int)[bin_idx]
    return one_hot

def income_bin_feature_names_out(*args):
    return np.array([
        'income_k_(-inf,55]',
        'income_k_(55,75]',
        'income_k_(75,95]',
        'income_k_(95,inf]'
    ])

def days_since_last_purchase_bin_names_out(*args):
    return np.array([
        'recent',
        'active',
        'cooling',
        'dormant'
    ])

loaded_pipeline = joblib.load('model_pipeline.joblib')

new_data = pd.DataFrame({
    'age': [45],
    'gender': ['Female'],
    'income_k': [88],
    'days_since_last_purchase': [30],
    'city': ["Paris"]
})

prediction = loaded_pipeline.predict(new_data)
probability = loaded_pipeline.predict_proba(new_data)

print(f'Prediction: {prediction}')
print(f'Probability: {probability}')
