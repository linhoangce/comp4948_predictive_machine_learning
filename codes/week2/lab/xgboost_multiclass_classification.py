import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

path = r"C:\Users\linho\Desktop\CST\term4\pa\data\housing_classification.csv"

df = pd.read_csv(path)

X = df.drop(columns=['price'], axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

NUM_CLASSES = 3

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    min_child_weight=1,
    gamma=0.5,
    colsample_bytree=0.7,
    eval_metric='mlogloss',
    objective='multi:softprob',
    num_class=NUM_CLASSES,
    reg_lambda=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

xgb_clf = xgb.XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    # random_state=42,
    n_jobs=-1,
    num_class=NUM_CLASSES
)

param_grid = {
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 3, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 400, 8000],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.5, 1.0],
    'reg_lambda': [1, 5, 10],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_grid = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_grid,
    scoring={
        'AUC': 'roc_auc_ovr',
        'LogLoss': 'neg_log_loss',
        'Accuracy': 'accuracy'
    },
    refit='AUC',
    cv=cv,
    n_jobs=-1,
    verbose=3,
    n_iter=1000
)

# random_grid.fit(X_train, y_train)
#
# print(f'Best CV AUC: {random_grid.best_score_}')
# print(f'Best params: {random_grid.best_params_}')
#
# best_model = random_grid.best_estimator_
# y_pred = best_model.predict(X_test)
# print(classification_report(y_test, y_pred))

param_grid_refined = {
    'max_depth': [4, 5],
    'min_child_weight': [1, 3],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [400, 8000],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [1.0],
    'reg_lambda': [1, 5, 10],
}

full_grid = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    scoring={
        'AUC': 'roc_auc_ovr',
        'LogLoss': 'neg_log_loss',
        'Accuracy': 'accuracy'
    },
    refit='AUC',
    cv=cv,
    n_jobs=-1,
    verbose=3
)

full_grid.fit(X_train, y_train)

print(f'Best CV AUC: {full_grid.best_score_}')
print(f'Best params: {full_grid.best_params_}')

best_model = full_grid.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))