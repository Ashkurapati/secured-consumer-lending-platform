"""
split_data.py
-------------
Step 3 of the Stepping Stones ML pipeline.

What this script does:
  1. Loads cleaned_loanstats.csv (output of data_cleaning.py)
  2. Drops funded_amnt (redundant with loan_amnt — confirmed by EDA)
  3. Separates features (X) and target (y)
  4. Stratified 80/20 train/test split — preserves the 78/22 class ratio
  5. Saves -> X_train.csv, X_test.csv, y_train.csv, y_test.csv

Run after data_cleaning.py:
    python split_data.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

INPUT_CSV = "cleaned_loanstats.csv"
TARGET    = "default"
TEST_SIZE = 0.20
SEED      = 42

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Stepping Stones -- Train/Test Split")
print("=" * 60)

print(f"\n[1/5] Loading {INPUT_CSV} ...")
df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"      Rows    : {len(df):,}")
print(f"      Columns : {len(df.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# DROP REDUNDANT COLUMN
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2/5] Dropping redundant columns ...")

cols_to_drop = []

# funded_amnt is nearly identical to loan_amnt (EDA confirmed correlation ~1.0)
# Keeping both adds noise, not signal
if "funded_amnt" in df.columns:
    cols_to_drop.append("funded_amnt")
    print(f"      Dropped: funded_amnt  (near-duplicate of loan_amnt)")

if cols_to_drop:
    df = df.drop(columns=cols_to_drop)

print(f"      Remaining columns: {len(df.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK CLASS DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3/5] Checking class distribution ...")

y = df[TARGET]
counts = y.value_counts()
print(f"      Class distribution in full dataset:")
print(f"        Paid off  (0) : {counts[0]:>10,}  ({counts[0]/len(y):.1%})")
print(f"        Defaulted (1) : {counts[1]:>10,}  ({counts[1]/len(y):.1%})")

# ─────────────────────────────────────────────────────────────────────────────
# STRATIFIED TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[4/5] Splitting data (80% train / 20% test, stratified) ...")

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=df[TARGET]     # preserves the 78/22 class ratio in both splits
)

print(f"\n      Train set : {len(train_df):>10,} rows")
print(f"      Test set  : {len(test_df):>10,} rows")

# Verify stratification worked
train_default_rate = train_df[TARGET].mean()
test_default_rate  = test_df[TARGET].mean()

print(f"\n      Default rate in train : {train_default_rate:.2%}")
print(f"      Default rate in test  : {test_default_rate:.2%}")

if abs(train_default_rate - test_default_rate) < 0.001:
    print("      Stratification check : PASSED (rates match within 0.1%)")
else:
    print("      [WARN] Stratification check : rates differ more than expected")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5/5] Saving splits ...")

train_df.to_csv("train.csv", index=False)
size_mb = os.path.getsize("train.csv") / 1024 ** 2
print(f"      Saved -> train.csv  ({size_mb:.1f} MB,  {len(train_df):,} rows)")

test_df.to_csv("test.csv", index=False)
size_mb = os.path.getsize("test.csv") / 1024 ** 2
print(f"      Saved -> test.csv   ({size_mb:.1f} MB,  {len(test_df):,} rows)")

print("\n" + "=" * 60)
print("  Done. Next step: class imbalance handling.")
print(f"  Train : {len(train_df):,} rows  |  Test : {len(test_df):,} rows")
print(f"  Columns : {len(train_df.columns)}  (includes '{TARGET}' target column)")
print(f"  Default rate preserved -> train: {train_default_rate:.2%} / test: {test_default_rate:.2%}")
print("=" * 60)
