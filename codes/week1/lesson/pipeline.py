import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer


def get_data():
    data = {
        "age": [
            25, 34, 45, np.nan, 52, 41, 29, 60, 48, 33,
            37, 55, 62, 44, 28, np.nan, 39, 47, 51, 31,
            36, 58, 63, 42, 27, 49, 54, 35, np.nan, 46,
            40, 59, 61, 32, 38
        ],
        "gender": [
            "Male", "Female", "Female", "Male", "Female",
            "Male", "Male", "Female", "Male", "Female",
            "Male", "Female", "Male", "Female", "Male",
            np.nan, "Female", "Male", "Female", "Male",
            "Female", "Male", "Female", "Male", "Female",
            "Male", "Female", "Male", "Female", "Male",
            "Female", "Male", "Female", "Male", "Female"
        ],
        "income_k": [
            45, 62, 58, 40, 80, 72, 50, 95, 66, 55,
            60, 88, 105, 70, 48, 52, 63, 75, 82, 54,
            57, 92, 110, 69, 46, 78, 86, 59, 61, 73,
            68, 98, 102, 56, 64
        ],
        "city": [
            "New York", "Paris", "London", "Toronto", "Paris",
            "London", "New York", "Toronto", "Paris", "London",
            "New York", "Paris", "Toronto", "London", "New York",
            "Paris", "London", "Toronto", "Paris", "New York",
            "London", "Toronto", "Paris", "New York", "London",
            "Paris", "Toronto", "New York", "London", "Paris",
            "Toronto", "New York", "Paris", "London", "Toronto"
        ],
        "days_since_last_purchase": [
            5, 40, 18, 120, 12, 30, 75, 4, 22, 15,
            35, 8, 2, 28, 90, 60, 20, 10, 6, 55,
            80, 7, 3, 45, 100, 14, 9, 25, 17, 32,
            11, 5, 1, 65, 19
        ],
        "email_open_rate": [
            0.35, 0.62, 0.48, np.nan, 0.81,
            0.55, 0.40, 0.90, 0.60, 0.50,
            np.nan, 0.72, 0.95, 0.58, 0.30,
            np.nan, 0.65, 0.78, 0.82, 0.45,
            0.52, 0.88, np.nan, 0.57, 0.33,
            0.70, 0.76, 0.49, np.nan, 0.66,
            0.61, 0.92, 0.97, 0.54, 0.59
        ],
        "target": [
            0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0,
            0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0
        ]
    }
    return pd.DataFrame(data)

def get_variable_groups():
    numerical_features = ['age', 'email_open_rate']
    categorical_features = ['gender', 'city']
    target = 'target'
    feature_income = ['income_k']
    feature_day = ['days_since_last_purchase']
    return numerical_features, categorical_features, target, feature_income, feature_day

def get_preprocessor(numerical_features, categorical_features, feature_income, feature_day):
    binned_income_transformer = Pipeline(steps=[
        ("imputer", KNNImputer(n_neighbors=3)),
        ("bin", FunctionTransformer(
            bin_income_np,
            validate=False,
            feature_names_out=income_bin_feature_names_out
        )),
    ])

    binned_day_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=3)),
        ('bin', FunctionTransformer(
            bin_days_since_last_purchase,
            validate=False,
            feature_names_out=days_since_last_purchase_bin_names_out
        ))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=3, weights='uniform')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('income_bins', binned_income_transformer, feature_income),
            ('day_bins', binned_day_transformer, feature_day),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor

def get_pipeline(preprocessor):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])
    return pipeline

def crossfold_evaluate(X, y, preprocessor):
    pipeline = get_pipeline(preprocessor)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    scores = cross_validate(
        pipeline,
        X, y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall"
        }
    )
    return scores


def show_crossfold_results(scores):
    print(f'Accuracy: {scores["test_accuracy"]}')
    print(f"F1: {scores['test_f1']}")
    print(f"Precision: {scores['test_precision']}")
    print(f"Recall: {scores['test_recall']}")

    print(f"Mean scores:")
    print(f"Accuracy: {scores['test_accuracy'].mean()}")
    print(f"F1: {scores['test_f1'].mean()}")
    print(f"Precision: {scores['test_precision'].mean()}")
    print(f"Recall: {scores['test_recall'].mean()}")

def perform_holdout_evaluation(X, y, preprocessor):
    pipeline = get_pipeline(preprocessor)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)
    print(f'Holdout Test Accuracy: {accuracy:.4f}')
    view_features_and_sample_data(X, pipeline)

def fit_and_save_pipeline(X, y, preprocessor):
    pipeline = get_pipeline(preprocessor)
    pipeline.fit(X, y)
    joblib.dump(pipeline, 'model_pipeline.joblib')
    print('Saved model_pipeline.joblib')

def view_features_and_sample_data(X, pipeline):
    preprocessor = pipeline.named_steps['preprocessor']

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    feature_names = preprocessor.get_feature_names_out()

    print('*** Final feature list:')
    for name in feature_names:
        print(name)

    X_prepared = preprocessor.transform(X.head(3))
    X_prepared_df = pd.DataFrame(
        X_prepared, columns=feature_names
    )
    print(X_prepared_df)


# *args is needed to absorb arguments passed by sklearn's Function Transformer
def income_bin_feature_names_out(*args):
    return np.array([
        "income_k_(-inf,55)",
        'income_k_(55,75)',
        'income_k_(75,95)',
        'income_k_(95,inf)'
    ])

def days_since_last_purchase_bin_names_out(*args):
    return np.array([
        'recent',
        'active',
        'cooling',
        'dormant'
    ])

def bin_income_np(X):
    x = np.asarray(X, dtype=float).reshape(-1)

    # fixed bin edges
    bins = [-np.inf, 55, 75, 95, np.inf]
    bin_idx = np.digitize(x, bins[1:-1], right=True)

    # one-hot encode with fixed width = 4
    one_hot = np.eye(len(bins) - 1, dtype=int)[bin_idx]
    return one_hot

def bin_days_since_last_purchase(X):
    x = np.asarray(X, dtype=int).reshape(-1)
    bins = [0, 7, 30, 90, np.inf]
    bin_idx = np.digitize(x, bins[1:-1], right=True)

    one_hot = np.eye(len(bins)-1, dtype=int)[bin_idx]
    return one_hot

(numerical_features, categorical_features,
 target, feature_income, feature_day) = get_variable_groups()
preprocessor = get_preprocessor(numerical_features, categorical_features,
                                feature_income, feature_day)

df = get_data()
X = df[numerical_features + categorical_features + feature_income + feature_day]
y = df[target]

cv_scores = crossfold_evaluate(X, y, preprocessor)
show_crossfold_results(cv_scores)

perform_holdout_evaluation(X, y, preprocessor)
fit_and_save_pipeline(X, y, preprocessor)

loaded_pipeline = joblib.load('model_pipeline.joblib')

new_data = pd.DataFrame({
    'age': [45],
    'gender': ['Female'],
    'income_k': [88],
    'days_since_last_purchase': [30],
    'email_open_rate': [0.4],
    'city': ['Paris']
})

prediction = loaded_pipeline.predict(new_data)
probability = loaded_pipeline.predict_proba(new_data)

print(f'Prediction: {prediction}')
print(f'Probability: {probability}')

