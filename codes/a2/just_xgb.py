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


FEATURES = [
    " ROA(C) before interest and depreciation before interest",
    " ROA(A) before interest and % after tax",
    " Operating Gross Margin",
    " Realized Sales Gross Margin",
    " Pre-tax net Interest Rate",
    " After-tax net Interest Rate",
    " Non-industry income and expenditure/revenue",
    " Continuous interest rate (after tax)",
    " Research and development expense rate",
    " Interest-bearing debt interest rate",
    " Tax rate (A)",
    " Net Value Per Share (B)",
    " Persistent EPS in the Last Four Seasons",
    " Cash Flow Per Share",
    " Revenue Per Share (Yuan \u00a5)",
    " Operating Profit Per Share (Yuan \u00a5)",
    " Continuous Net Profit Growth Rate",
    " Total Asset Growth Rate",
    " Net Value Growth Rate",
    " Interest Expense Ratio",
    " Total debt/Total net worth",
    " Borrowing dependency",
    " Contingent liabilities/Net worth",
    " Net profit before tax/Paid-in capital",
    " Average Collection Days",
    " Inventory Turnover Rate (times)",
    " Fixed Assets Turnover Frequency",
    " Revenue per person",
    " Operating profit per person",
    " Allocation rate per person",
    " Working Capital to Total Assets",
    " Current Assets/Total Assets",
    " Cash/Total Assets",
    " Cash/Current Liability",
    " Current Liability to Assets",
    " Inventory/Working Capital",
    " Inventory/Current Liability",
    " Working Capital/Equity",
    " Current Liabilities/Equity",
    " Long-term Liability to Current Assets",
    " Retained Earnings to Total Assets",
    " Total income/Total expense",
    " Total expense/Assets",
    " Current Asset Turnover Rate",
    " Quick Asset Turnover Rate",
    " Working capitcal Turnover Rate",
    " Cash Turnover Rate",
    " Cash Flow to Sales",
    " Equity to Long-term Liability",
    " Cash Flow to Total Assets",
    " Cash Flow to Liability",
    " Current Liability to Current Assets",
    " Net Income to Total Assets",
    " Total assets to GNP price",
    " Net Income to Stockholder's Equity",
    " Degree of Financial Leverage (DFL)",
    " Interest Coverage Ratio (Interest expense to EBIT)",
    " Equity to Liability"
  ]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train_final, y_train_final = SMOTEENN(random_state=42).fit_resample(X_train[FEATURES], y_train)

model = xgb.XGBClassifier(
    max_depth=7,
    learning_rate=0.04808381637743888,
    subsample=0.9521533385305752,
    colsample_bytree=0.5132407670456729,
    reg_alpha=1.8891877473069133e-05,
    reg_lambda=2.448830695774989,
    min_child_weight=1,
    gamma=0.2190194401075139,
    n_estimators=439,
)

model.fit(X_train_final, y_train_final)

y_pred_prob = model.predict_proba(X_test[FEATURES])[:, 1]
y_pred = (y_pred_prob > 0.6875752500686682).astype(int)

print(classification_report(y_test, y_pred))