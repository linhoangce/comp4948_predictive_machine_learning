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

FEATURES = [
    " ROA(C) before interest and depreciation before interest",
    " ROA(A) before interest and % after tax",
    " ROA(B) before interest and depreciation after tax",
    " Operating Gross Margin",
    " Realized Sales Gross Margin",
    " Operating Profit Rate",
    " Pre-tax net Interest Rate",
    " After-tax net Interest Rate",
    " Non-industry income and expenditure/revenue",
    " Continuous interest rate (after tax)",
    " Operating Expense Rate",
    " Research and development expense rate",
    " Cash flow rate",
    " Interest-bearing debt interest rate",
    " Tax rate (A)",
    " Net Value Per Share (B)",
    " Persistent EPS in the Last Four Seasons",
    " Cash Flow Per Share",
    " Revenue Per Share (Yuan \u00a5)",
    " Operating Profit Per Share (Yuan \u00a5)",
    " Realized Sales Gross Profit Growth Rate",
    " Continuous Net Profit Growth Rate",
    " Total Asset Growth Rate",
    " Net Value Growth Rate",
    " Total Asset Return Growth Rate Ratio",
    " Cash Reinvestment %",
    " Quick Ratio",
    " Interest Expense Ratio",
    " Total debt/Total net worth",
    " Debt ratio %",
    " Long-term fund suitability ratio (A)",
    " Borrowing dependency",
    " Contingent liabilities/Net worth",
    " Net profit before tax/Paid-in capital",
    " Inventory and accounts receivable/Net value",
    " Total Asset Turnover",
    " Average Collection Days",
    " Inventory Turnover Rate (times)",
    " Fixed Assets Turnover Frequency",
    " Net Worth Turnover Rate (times)",
    " Revenue per person",
    " Operating profit per person",
    " Allocation rate per person",
    " Working Capital to Total Assets",
    " Quick Assets/Total Assets",
    " Current Assets/Total Assets",
    " Cash/Total Assets",
    " Cash/Current Liability",
    " Current Liability to Assets",
    " Operating Funds to Liability",
    " Inventory/Working Capital",
    " Inventory/Current Liability",
    " Current Liabilities/Liability",
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
    " CFO to Assets",
    " Cash Flow to Equity",
    " Current Liability to Current Assets",
    " Net Income to Total Assets",
    " Total assets to GNP price",
    " No-credit Interval",
    " Net Income to Stockholder's Equity",
    " Degree of Financial Leverage (DFL)",
    " Interest Coverage Ratio (Interest expense to EBIT)",
    " Equity to Liability"
  ]

df = pd.read_csv("data/bankruptcy_asgn2.csv")
X = df.drop(columns=["Bankrupt?"])
y = df["Bankrupt?"]

X_train, X_test, y_train, y_test = train_test_split(
    X[FEATURES], y, test_size=0.2, stratify=y, random_state=42
)

X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

model = RandomForestClassifier(
    max_depth=20,
    max_features=0.9225715472584318,
    min_samples_split=16,
    min_samples_leaf=16,
    max_samples=0.8384988859968542,
    criterion="entropy",
    n_estimators=1075,
    n_jobs=-1,
    random_state=42,
)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.6751).astype(int)
print(classification_report(y_test, y_pred))


