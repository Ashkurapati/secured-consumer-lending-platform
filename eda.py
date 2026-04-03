"""
eda.py
------
Step 3 of the Stepping Stones ML pipeline.

What this script does:
  1.  Class balance check
  2.  Default rate by grade, purpose, home ownership, emp_length, term
  3.  Distributions of key numeric features
  4.  Outlier detection on income and loan amount
  5.  Correlation heatmap
  6.  Default rate vs DTI, income, int_rate buckets

All charts saved to the eda_charts/ folder.

Run after data_cleaning.py:
    python eda.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

INPUT_CSV   = "cleaned_loanstats.csv"
OUTPUT_DIR  = "eda_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")
FIGSIZE = (10, 5)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Stepping Stones -- EDA")
print("=" * 60)

print("\nLoading cleaned_loanstats.csv ...")
df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"  Shape : {df.shape}")

# Sample for plots that don't need full data
df_sample = df.sample(20000, random_state=42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CLASS BALANCE
# ─────────────────────────────────────────────────────────────────────────────

print("\n[1/9] Class balance ...")

counts = df["default"].value_counts()
pcts   = df["default"].value_counts(normalize=True) * 100

print(f"  Paid off  (0) : {counts[0]:>9,}  ({pcts[0]:.2f}%)")
print(f"  Defaulted (1) : {counts[1]:>9,}  ({pcts[1]:.2f}%)")

fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(["Paid Off (0)", "Defaulted (1)"], counts.values,
       color=["#4CAF50", "#F44336"], edgecolor="white", width=0.5)
for i, (v, p) in enumerate(zip(counts.values, pcts.values)):
    ax.text(i, v + 5000, f"{v:,}\n({p:.1f}%)", ha="center", fontsize=11)
ax.set_title("Class Balance — Paid Off vs Defaulted", fontsize=14, pad=12)
ax.set_ylabel("Number of Loans")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_class_balance.png", dpi=120)
plt.close()
print(f"  Saved -> {OUTPUT_DIR}/01_class_balance.png")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DEFAULT RATE BY GRADE
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2/9] Default rate by grade ...")

grade_labels = {1:"A", 2:"B", 3:"C", 4:"D", 5:"E", 6:"F", 7:"G"}
df["grade_label"] = df["grade"].map(grade_labels)
grade_dr = df.groupby("grade_label")["default"].mean().reindex(["A","B","C","D","E","F","G"]) * 100
grade_ct = df.groupby("grade_label")["default"].count().reindex(["A","B","C","D","E","F","G"])

print(f"  {'Grade':<8} {'Default Rate':>14} {'Count':>10}")
for g in ["A","B","C","D","E","F","G"]:
    print(f"  {g:<8} {grade_dr[g]:>13.2f}%  {grade_ct[g]:>10,}")

fig, ax = plt.subplots(figsize=FIGSIZE)
bars = ax.bar(grade_dr.index, grade_dr.values,
              color=sns.color_palette("RdYlGn_r", 7), edgecolor="white")
for bar, val in zip(bars, grade_dr.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", fontsize=10)
ax.set_title("Default Rate by Loan Grade", fontsize=14, pad=12)
ax.set_xlabel("Grade (A = Lowest Risk, G = Highest Risk)")
ax.set_ylabel("Default Rate (%)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_default_by_grade.png", dpi=120)
plt.close()
df = df.drop(columns=["grade_label"])
print(f"  Saved -> {OUTPUT_DIR}/02_default_by_grade.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. DEFAULT RATE BY PURPOSE
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3/9] Default rate by purpose ...")

purpose_cols = [c for c in df.columns if c.startswith("purpose_")]
purpose_dr = {}
for col in purpose_cols:
    label = col.replace("purpose_", "").replace("_", " ").title()
    mask  = df[col] == 1
    rate  = df.loc[mask, "default"].mean() * 100
    count = mask.sum()
    purpose_dr[label] = {"rate": rate, "count": count}

purpose_df = pd.DataFrame(purpose_dr).T.sort_values("rate", ascending=True)

print(f"  {'Purpose':<25} {'Default Rate':>14} {'Count':>10}")
for p, row in purpose_df.iterrows():
    print(f"  {p:<25} {row['rate']:>13.2f}%  {int(row['count']):>10,}")

fig, ax = plt.subplots(figsize=(10, 7))
colors = sns.color_palette("RdYlGn_r", len(purpose_df))
bars = ax.barh(purpose_df.index, purpose_df["rate"].values,
               color=colors, edgecolor="white")
for bar, val in zip(bars, purpose_df["rate"].values):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=9)
ax.set_title("Default Rate by Loan Purpose", fontsize=14, pad=12)
ax.set_xlabel("Default Rate (%)")
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_default_by_purpose.png", dpi=120)
plt.close()
print(f"  Saved -> {OUTPUT_DIR}/03_default_by_purpose.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. DEFAULT RATE BY HOME OWNERSHIP
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4/9] Default rate by home ownership ...")

home_labels = {4:"OWN", 3:"MORTGAGE", 2:"RENT", 1:"OTHER"}
df["home_label"] = df["home_ownership"].map(home_labels)
home_dr = df.groupby("home_label")["default"].mean() * 100
home_ct = df.groupby("home_label")["default"].count()

print(f"  {'Ownership':<12} {'Default Rate':>14} {'Count':>10}")
for h in ["OWN","MORTGAGE","RENT","OTHER"]:
    if h in home_dr:
        print(f"  {h:<12} {home_dr[h]:>13.2f}%  {home_ct[h]:>10,}")

fig, ax = plt.subplots(figsize=(7, 5))
order = [h for h in ["OWN","MORTGAGE","RENT","OTHER"] if h in home_dr.index]
vals  = [home_dr[h] for h in order]
ax.bar(order, vals, color=["#4CAF50","#2196F3","#FF9800","#9E9E9E"],
       edgecolor="white", width=0.5)
for i, v in enumerate(vals):
    ax.text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=11)
ax.set_title("Default Rate by Home Ownership", fontsize=14, pad=12)
ax.set_ylabel("Default Rate (%)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_default_by_home_ownership.png", dpi=120)
plt.close()
df = df.drop(columns=["home_label"])
print(f"  Saved -> {OUTPUT_DIR}/04_default_by_home_ownership.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. DEFAULT RATE BY TERM
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5/9] Default rate by term ...")

term_dr = df.groupby("term")["default"].mean() * 100
term_ct = df.groupby("term")["default"].count()

print(f"  {'Term':<8} {'Default Rate':>14} {'Count':>10}")
for t in term_dr.index:
    print(f"  {t} mths  {term_dr[t]:>13.2f}%  {term_ct[t]:>10,}")

fig, ax = plt.subplots(figsize=(6, 5))
ax.bar([f"{t} months" for t in term_dr.index], term_dr.values,
       color=["#2196F3","#F44336"], edgecolor="white", width=0.4)
for i, v in enumerate(term_dr.values):
    ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=12)
ax.set_title("Default Rate by Loan Term", fontsize=14, pad=12)
ax.set_ylabel("Default Rate (%)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_default_by_term.png", dpi=120)
plt.close()
print(f"  Saved -> {OUTPUT_DIR}/05_default_by_term.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. DISTRIBUTIONS OF KEY NUMERIC FEATURES
# ─────────────────────────────────────────────────────────────────────────────

print("\n[6/9] Key numeric distributions ...")

# Cap income at 99th percentile for readability
inc_cap = df_sample["annual_inc"].quantile(0.99)
dti_cap = df_sample["dti"].quantile(0.99)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Distribution of Key Features (Defaulted vs Paid Off)", fontsize=14)

plots = [
    ("annual_inc",   inc_cap, "Annual Income ($)",       axes[0][0]),
    ("dti",          dti_cap, "Debt-to-Income Ratio",    axes[0][1]),
    ("int_rate",     None,    "Interest Rate (%)",       axes[0][2]),
    ("loan_amnt",    None,    "Loan Amount ($)",         axes[1][0]),
    ("revol_util",   None,    "Revolving Utilization %", axes[1][1]),
    ("credit_history_months", None, "Credit History (Months)", axes[1][2]),
]

for col, cap, label, ax in plots:
    data = df_sample.copy()
    if cap:
        data = data[data[col] <= cap]
    paid    = data.loc[data["default"] == 0, col]
    default = data.loc[data["default"] == 1, col]
    ax.hist(paid,    bins=40, alpha=0.6, color="#4CAF50", label="Paid Off",  density=True)
    ax.hist(default, bins=40, alpha=0.6, color="#F44336", label="Defaulted", density=True)
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_feature_distributions.png", dpi=120)
plt.close()
print(f"  Saved -> {OUTPUT_DIR}/06_feature_distributions.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. OUTLIER CHECK — INCOME & LOAN AMOUNT
# ─────────────────────────────────────────────────────────────────────────────

print("\n[7/9] Outlier check ...")

for col in ["annual_inc", "loan_amnt", "dti", "revol_bal"]:
    q1  = df[col].quantile(0.25)
    q3  = df[col].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 3 * iqr
    outliers = (df[col] > upper).sum()
    pct = outliers / len(df) * 100
    print(f"  {col:<25} upper fence: {upper:>12,.0f}  outliers: {outliers:>7,} ({pct:.2f}%)")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Outlier Check", fontsize=14)

inc_99 = df["annual_inc"].quantile(0.99)
axes[0].hist(df.loc[df["annual_inc"] <= inc_99, "annual_inc"],
             bins=50, color="#2196F3", edgecolor="white")
axes[0].set_title("Annual Income (capped at 99th pct)")
axes[0].set_xlabel("Annual Income ($)")
axes[0].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${int(x):,}"))

axes[1].hist(df["loan_amnt"], bins=50, color="#9C27B0", edgecolor="white")
axes[1].set_title("Loan Amount")
axes[1].set_xlabel("Loan Amount ($)")
axes[1].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${int(x):,}"))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_outliers.png", dpi=120)
plt.close()
print(f"  Saved -> {OUTPUT_DIR}/07_outliers.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. DEFAULT RATE BY DTI, INCOME & INTEREST RATE BUCKETS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[8/9] Default rate by DTI / income / interest rate buckets ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Default Rate by Key Risk Buckets", fontsize=14)

# DTI buckets
df["dti_bucket"] = pd.cut(df["dti"], bins=[0,10,20,30,40,100],
                           labels=["0-10","10-20","20-30","30-40","40+"])
dti_dr = df.groupby("dti_bucket", observed=True)["default"].mean() * 100
axes[0].bar(dti_dr.index.astype(str), dti_dr.values, color=sns.color_palette("RdYlGn_r", 5))
for i, v in enumerate(dti_dr.values):
    axes[0].text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=9)
axes[0].set_title("Default Rate by DTI")
axes[0].set_xlabel("DTI Bucket")
axes[0].set_ylabel("Default Rate (%)")
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())

# Income buckets (cap at 200k)
df_inc = df[df["annual_inc"] <= 200000].copy()
df_inc["inc_bucket"] = pd.cut(df_inc["annual_inc"],
                               bins=[0,30000,50000,75000,100000,200000],
                               labels=["<30k","30-50k","50-75k","75-100k","100-200k"])
inc_dr = df_inc.groupby("inc_bucket", observed=True)["default"].mean() * 100
axes[1].bar(inc_dr.index.astype(str), inc_dr.values, color=sns.color_palette("RdYlGn_r", 5))
for i, v in enumerate(inc_dr.values):
    axes[1].text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=9)
axes[1].set_title("Default Rate by Annual Income")
axes[1].set_xlabel("Income Bucket")
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())

# Interest rate buckets
df["rate_bucket"] = pd.cut(df["int_rate"], bins=[0,8,12,16,20,31],
                            labels=["<8%","8-12%","12-16%","16-20%","20%+"])
rate_dr = df.groupby("rate_bucket", observed=True)["default"].mean() * 100
axes[2].bar(rate_dr.index.astype(str), rate_dr.values, color=sns.color_palette("RdYlGn_r", 5))
for i, v in enumerate(rate_dr.values):
    axes[2].text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=9)
axes[2].set_title("Default Rate by Interest Rate")
axes[2].set_xlabel("Interest Rate Bucket")
axes[2].yaxis.set_major_formatter(mtick.PercentFormatter())

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_default_by_buckets.png", dpi=120)
plt.close()

# Clean up temp columns
df = df.drop(columns=["dti_bucket", "rate_bucket"], errors="ignore")

print(f"  Saved -> {OUTPUT_DIR}/08_default_by_buckets.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. CORRELATION HEATMAP (top features vs default)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[9/9] Correlation heatmap ...")

# Use top 20 most correlated features with default for readability
corr_with_target = df.corr(numeric_only=True)["default"].drop("default").abs()
top20 = corr_with_target.nlargest(20).index.tolist()

corr_matrix = df[top20 + ["default"]].corr(numeric_only=True)

print("  Top 10 features most correlated with default:")
top10 = corr_with_target.nlargest(10)
for feat, val in top10.items():
    direction = "+" if df[feat].corr(df["default"]) > 0 else "-"
    print(f"    {direction}{val:.4f}  {feat}")

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn_r", center=0, linewidths=0.5,
            ax=ax, annot_kws={"size": 8})
ax.set_title("Correlation Matrix — Top 20 Features vs Default", fontsize=14, pad=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_correlation_heatmap.png", dpi=120)
plt.close()
print(f"  Saved -> {OUTPUT_DIR}/09_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY STATS PRINTOUT
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  EDA Complete — all charts saved to eda_charts/")
print("=" * 60)
print(f"\n  01_class_balance.png          -- paid off vs defaulted counts")
print(f"  02_default_by_grade.png       -- default rate A through G")
print(f"  03_default_by_purpose.png     -- default rate by loan purpose")
print(f"  04_default_by_home_ownership  -- OWN vs MORTGAGE vs RENT")
print(f"  05_default_by_term.png        -- 36 vs 60 month loans")
print(f"  06_feature_distributions.png  -- income, DTI, rate distributions")
print(f"  07_outliers.png               -- income and loan amount spread")
print(f"  08_default_by_buckets.png     -- DTI / income / rate risk buckets")
print(f"  09_correlation_heatmap.png    -- top 20 features vs default")
print("=" * 60)
