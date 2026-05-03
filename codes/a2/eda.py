import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

df = pd.read_csv("data/bankruptcy_asgn2.csv")

TOP_FEATURES = [
    "Bankrupt?",
    " ROA(C) before interest and depreciation before interest",
    " ROA(A) before interest and % after tax",
    " ROA(B) before interest and depreciation after tax",
    " Tax rate (A)",
    " Net Value Per Share (B)",
    " Net Value Per Share (A)",
    " Net Value Per Share (C)",
    " Persistent EPS in the Last Four Seasons",
    " Operating Profit Per Share (Yuan \u00a5)",
    " Per Share Net profit before tax (Yuan \u00a5)",
    " Debt ratio %",
    " Net worth/Assets",
    " Borrowing dependency",
    " Operating profit/Paid-in capital",
    " Net profit before tax/Paid-in capital",
    " Working Capital to Total Assets",
    " Current Liability to Assets",
    " Working Capital/Equity",
    " Current Liabilities/Equity",
    " Retained Earnings to Total Assets",
    " Total expense/Assets",
    " Current Liability to Equity",
    " Equity to Long-term Liability",
    " CFO to Assets",
    " Current Liability to Current Assets",
    " Net Income to Total Assets",
    " Net Income to Stockholder's Equity",
    " Liability to Equity"
  ]

bankrupt_df = df[TOP_FEATURES]

# Check labels distribution
# sns.set_theme(context="paper")
#
# plt.figure(figsize=(10, 5))
# custom_color = {0: "royalblue", 1: "crimson"}
# ax = sns.countplot(x="Bankrupt?", data=bankrupt_df, hue="Bankrupt?", palette=custom_color, legend=False)
# ax.bar_label(ax.containers[0]) # add count to bar top
# plt.title(f"Class Distribution | Total: {len(bankrupt_df)}", fontsize=20)
# plt.xlabel("Class", fontsize=16)
# plt.ylabel("Count", fontsize=16)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.show()

# Histograms
# cols = 3
# rows = math.ceil(len(TOP_FEATURES) / cols)
#
# fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4))
# axes = axes.flatten() # for iteration
#
# for i, col_name in enumerate(TOP_FEATURES):
#     sns.histplot(
#         data=bankrupt_df,
#         x=col_name,
#         # hue="Bankrupt?",
#         kde=True,
#         ax=axes[i],
#         palette="magma",
#         element="step"
#     )
#
#     axes[i].set_title(col_name, fontsize=18)
#     axes[i].tick_params(labelsize=14)
#     axes[i].set_ylabel("Count", fontsize=14)
#     axes[i].set_xlabel("")
#
# # remove empty subplots
# for j in range(i + 1, len(axes)):
#     fig.delaxes(axes[j])
# plt.tight_layout()
# plt.show()

# =============== Correlation Heatmap =======================

# f, ax = plt.subplots(figsize=(35, 35))
# mat = bankrupt_df.corr("spearman")
# mask = np.triu(np.ones_like(mat, dtype=bool))
# cmap = sns.diverging_palette(230, 20, as_cmap=True)
# sns.heatmap(mat, mask=mask, cmap=cmap, vmax=1, center=0,
#             square=True, linewidth=.5, cbar_kws={"shrink": .5},
#             ax=ax
#         )
# ax.tick_params(axis="both", which="major", labelsize=28)
#
# cbar = ax.collections[0].colorbar
# cbar.ax.tick_params(labelsize=28)
# plt.xticks(rotation=45, ha="right")
# plt.subplots_adjust(bottom=0.25, left=0.25)
# plt.show()

# ================ Interesting Features Boxplots =============

f, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
axes = axes.flatten()

sns.boxplot(x="Bankrupt?", y=" Retained Earnings to Total Assets", data=bankrupt_df, ax=axes[0])
axes[0].set_title("Bankrupt vs Retained Earnings to Total Assets", fontsize=20)
axes[0].set_ylabel("Retained Earnings to Total Assets", fontsize=16)
axes[0].set_xlabel("")
axes[0].tick_params(axis="both", which="major", labelsize=16)

sns.boxplot(x="Bankrupt?", y=" Debt ratio %", data=bankrupt_df, ax=axes[1])
axes[1].set_title("Bankrupt vs Debt ratio %", fontsize=20)
axes[1].set_ylabel("Debt ratio %", fontsize=16)
axes[1].set_xlabel("")
axes[1].tick_params(axis="both", which="major", labelsize=16)

sns.boxplot(x="Bankrupt?", y=" Net Income to Total Assets", data=bankrupt_df, ax=axes[2])
axes[2].set_title("Bankrupt vs Net Income to Total Assets", fontsize=20)
axes[2].set_ylabel("Net Income to Total Assets", fontsize=16)
axes[2].set_xlabel("")
axes[2].tick_params(axis="both", which="major", labelsize=16)

sns.boxplot(x="Bankrupt?", y=" Borrowing dependency", data=bankrupt_df, ax=axes[3])
axes[3].set_title("Bankrupt vs Borrowing dependency", fontsize=20)
axes[3].set_ylabel("Borrowing dependency", fontsize=16)
axes[3].set_xlabel("")
axes[3].tick_params(axis="both", which="major", labelsize=16)

plt.tight_layout()
plt.show()


# =============== FEATURE DISTRIBUTION FOR BANKRUPT COMPANIES ======================
f, axes = plt.subplots(2, 2, figsize=(24, 12))
axes = axes.flatten()

retained_earnings = bankrupt_df[" Retained Earnings to Total Assets"].loc[bankrupt_df["Bankrupt?"] == 1].values
sns.histplot(retained_earnings, ax=axes[0], color="#FB8861", stat="density")
mu, std = norm.fit(retained_earnings)
x = np.linspace(min(retained_earnings), max(retained_earnings), 100)
p = norm.pdf(x, mu, std)
axes[0].plot(x, p, "k", linewidth=2)
axes[0].set_title(" Retained Earnings to Total Assets (Bankrupt)", fontsize=20)
axes[0].tick_params(axis="both", which="major", labelsize=16)
axes[0].set_ylabel("Density", fontsize=16)

debt = bankrupt_df[" Debt ratio %"].loc[bankrupt_df["Bankrupt?"] == 1].values
sns.histplot(debt, ax=axes[1], color='#56F9BB', stat="density")
mu, std = norm.fit(debt)
x = np.linspace(min(debt), max(debt), 100)
p = norm.pdf(x, mu, std)
axes[1].plot(x, p, "k", linewidth=2)
axes[1].set_title("Debt ratio % (Bankrupt)", fontsize=20)
axes[1].tick_params(axis="both", which="major", labelsize=16)
axes[1].set_ylabel("Density", fontsize=16)


net_total = bankrupt_df[" Net Income to Total Assets"].loc[bankrupt_df["Bankrupt?"] == 1].values
sns.histplot(debt, ax=axes[2], color='#00F9A8', stat="density")
mu, std = norm.fit(net_total)
x = np.linspace(min(net_total), max(net_total), 100)
p = norm.pdf(x, mu, std)
axes[2].plot(x, p, "k", linewidth=2)
axes[2].set_title("Net Income to Total Assets (Bankrupt)", fontsize=20)
axes[2].tick_params(axis="both", which="major", labelsize=16)
axes[2].set_ylabel("Density", fontsize=16)


borrow = bankrupt_df[" Borrowing dependency"].loc[bankrupt_df["Bankrupt?"] == 1].values
sns.histplot(debt, ax=axes[3], color='#C5B3F9', stat="density")
mu, std = norm.fit(borrow)
x = np.linspace(min(borrow), max(borrow), 100)
p = norm.pdf(x, mu, std)
axes[3].plot(x, p, "k", linewidth=2)
axes[3].set_title("Borrowing dependency (Bankrupt)", fontsize=20)
axes[3].tick_params(axis="both", which="major", labelsize=16)
axes[3].set_ylabel("Density", fontsize=16)


plt.show()