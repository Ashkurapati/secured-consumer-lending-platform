"""
data_cleaning.py
----------------
Step 2 of the Stepping Stones ML pipeline.

What this script does:
  1. Loads clean_loanstats.csv (output of data_extraction.py)
  2. Fixes missing values — each column handled based on what null means
  3. Fixes data types — converts string columns to numeric
  4. Encodes categorical columns
  5. Saves -> cleaned_loanstats.csv

Run after data_extraction.py:
    python data_cleaning.py
"""

import pandas as pd
import numpy as np

INPUT_CSV  = "clean_loanstats.csv"
OUTPUT_CSV = "cleaned_loanstats.csv"

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Stepping Stones -- Data Cleaning")
print("=" * 60)

print("\n[1/6] Loading clean_loanstats.csv ...")
df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"      Rows    : {len(df):,}")
print(f"      Columns : {len(df.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — FIX MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2/6] Fixing missing values ...")

# --- Group 1: mths_since_* columns ---
# Null means the event NEVER happened. Fill with 999 so the model
# knows this is a "never occurred" signal, not just a missing value.
never_happened_cols = [
    "mths_since_last_record",
    "mths_since_recent_bc_dlq",
    "mths_since_last_major_derog",
    "mths_since_recent_revol_delinq",
    "mths_since_last_delinq",
    "mths_since_recent_inq",
]
for col in never_happened_cols:
    if col in df.columns:
        filled = df[col].isna().sum()
        df[col] = df[col].fillna(999)
        print(f"      {col:<40} filled {filled:>8,} nulls with 999")

# --- Group 2: Older loan fields (60% missing same rows) ---
# These fields weren't collected on older loans. Fill with 0.
older_loan_cols = [
    "open_acc_6m",
    "inq_last_12m",
    "all_util",
    "total_bal_il",
    "il_util",
]
for col in older_loan_cols:
    if col in df.columns:
        filled = df[col].isna().sum()
        df[col] = df[col].fillna(0)
        print(f"      {col:<40} filled {filled:>8,} nulls with 0")

# --- Group 3: Thin credit file borrowers (~5-9% missing) ---
# Null means no such account type exists for this borrower. Fill with 0.
thin_file_zero_cols = [
    "num_tl_120dpd_2m",
    "mo_sin_old_il_acct",
    "tot_coll_amt",
    "num_accts_ever_120_pd",
    "mo_sin_rcnt_tl",
    "tot_cur_bal",
    "tot_hi_cred_lim",
    "num_rev_accts",
    "num_tl_90g_dpd_24m",
    "num_bc_tl",
    "num_actv_rev_tl",
    "num_il_tl",
    "num_actv_bc_tl",
    "avg_cur_bal",
    "num_tl_30dpd",
    "bc_util",
    "percent_bc_gt_75",
    "num_sats",
    "num_bc_sats",
    "total_bc_limit",
    "mort_acc",
    "total_bal_ex_mort",
    "acc_open_past_24mths",
]
for col in thin_file_zero_cols:
    if col in df.columns:
        filled = df[col].isna().sum()
        df[col] = df[col].fillna(0)
        print(f"      {col:<40} filled {filled:>8,} nulls with 0")

# pct_tl_nvr_dlq: if no trades, borrower has never been delinquent -> 100
if "pct_tl_nvr_dlq" in df.columns:
    filled = df["pct_tl_nvr_dlq"].isna().sum()
    df["pct_tl_nvr_dlq"] = df["pct_tl_nvr_dlq"].fillna(100)
    print(f"      {'pct_tl_nvr_dlq':<40} filled {filled:>8,} nulls with 100")

# emp_length: categorical — fill with 'Unknown'
if "emp_length" in df.columns:
    filled = df["emp_length"].isna().sum()
    df["emp_length"] = df["emp_length"].fillna("Unknown")
    print(f"      {'emp_length':<40} filled {filled:>8,} nulls with 'Unknown'")

# --- Group 4: Near-complete columns (<1% missing) ---
# Count/event columns: fill with 0 (no event occurred)
near_complete_zero_cols = [
    "pub_rec_bankruptcies",
    "chargeoff_within_12_mths",
    "collections_12_mths_ex_med",
    "tax_liens",
    "delinq_2yrs",
    "delinq_amnt",
    "acc_now_delinq",
    "pub_rec",
    "open_acc",
    "total_acc",
]
for col in near_complete_zero_cols:
    if col in df.columns:
        filled = df[col].isna().sum()
        if filled > 0:
            df[col] = df[col].fillna(0)
            print(f"      {col:<40} filled {filled:>8,} nulls with 0")

# Continuous columns: fill with median
median_fill_cols = ["revol_util", "dti", "annual_inc", "inq_last_6mths"]
for col in median_fill_cols:
    if col in df.columns:
        filled = df[col].isna().sum()
        if filled > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"      {col:<40} filled {filled:>8,} nulls with median ({median_val:.2f})")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FIX DATA TYPES
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3/6] Fixing data types ...")

# --- term: " 36 months" / " 60 months" -> 36 / 60 (int) ---
if "term" in df.columns:
    df["term"] = df["term"].str.extract(r"(\d+)").astype(int)
    print(f"      term              : extracted numeric -> {df['term'].unique()}")

# --- int_rate: already float, confirm no % sign issues ---
if "int_rate" in df.columns:
    if df["int_rate"].dtype == object:
        df["int_rate"] = df["int_rate"].str.replace("%", "").astype(float)
        print("      int_rate          : stripped % sign")
    else:
        print(f"      int_rate          : already numeric, range {df['int_rate'].min():.2f}% - {df['int_rate'].max():.2f}%")

# --- earliest_cr_line: date string -> credit history length in months ---
# Convert "Jan-2000" to number of months before the most recent issue_d
if "earliest_cr_line" in df.columns:
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    reference_date = pd.Timestamp("2018-01-01")  # approximate end of dataset
    df["credit_history_months"] = (
        (reference_date.year - df["earliest_cr_line"].dt.year) * 12
        + (reference_date.month - df["earliest_cr_line"].dt.month)
    )
    df["credit_history_months"] = df["credit_history_months"].clip(lower=0)
    # Fill any remaining nulls with median
    median_ch = df["credit_history_months"].median()
    df["credit_history_months"] = df["credit_history_months"].fillna(median_ch)
    df = df.drop(columns=["earliest_cr_line"])
    print(f"      earliest_cr_line  : converted to credit_history_months (median: {median_ch:.0f} months)")

# --- emp_length: text -> numeric years ---
emp_map = {
    "< 1 year":  0,
    "1 year":    1,
    "2 years":   2,
    "3 years":   3,
    "4 years":   4,
    "5 years":   5,
    "6 years":   6,
    "7 years":   7,
    "8 years":   8,
    "9 years":   9,
    "10+ years": 10,
    "Unknown":   -1,   # -1 signals unknown employment length to the model
}
if "emp_length" in df.columns:
    df["emp_length"] = df["emp_length"].map(emp_map)
    unmapped = df["emp_length"].isna().sum()
    if unmapped > 0:
        df["emp_length"] = df["emp_length"].fillna(-1)
        print(f"      emp_length        : {unmapped:,} unmapped values set to -1")
    print(f"      emp_length        : converted to numeric, range {df['emp_length'].min()} - {df['emp_length'].max()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — ENCODE CATEGORICAL COLUMNS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4/6] Encoding categorical columns ...")

# --- grade: A=1, B=2, ..., G=7 ---
grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
if "grade" in df.columns:
    df["grade"] = df["grade"].map(grade_map)
    print(f"      grade             : ordinal encoded A=1 to G=7")

# --- sub_grade: A1=1, A2=2, ..., G5=35 ---
sub_grades = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
sub_grade_map = {sg: i + 1 for i, sg in enumerate(sub_grades)}
if "sub_grade" in df.columns:
    df["sub_grade"] = df["sub_grade"].map(sub_grade_map)
    print(f"      sub_grade         : ordinal encoded A1=1 to G5=35")

# --- home_ownership: ordinal by financial stability ---
home_map = {
    "OWN":      4,
    "MORTGAGE": 3,
    "RENT":     2,
    "OTHER":    1,
    "NONE":     1,
    "ANY":      1,
}
if "home_ownership" in df.columns:
    df["home_ownership"] = df["home_ownership"].map(home_map)
    df["home_ownership"] = df["home_ownership"].fillna(1)
    print(f"      home_ownership    : ordinal encoded OWN=4, MORTGAGE=3, RENT=2, OTHER=1")

# --- verification_status ---
verif_map = {
    "Verified":        2,
    "Source Verified": 1,
    "Not Verified":    0,
}
if "verification_status" in df.columns:
    df["verification_status"] = df["verification_status"].map(verif_map)
    df["verification_status"] = df["verification_status"].fillna(0)
    print(f"      verification_status: encoded Verified=2, Source Verified=1, Not Verified=0")

# --- purpose: one-hot encode (14 categories) ---
if "purpose" in df.columns:
    purpose_dummies = pd.get_dummies(df["purpose"], prefix="purpose", drop_first=False)
    df = pd.concat([df.drop(columns=["purpose"]), purpose_dummies], axis=1)
    print(f"      purpose           : one-hot encoded -> {len(purpose_dummies.columns)} columns: {list(purpose_dummies.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — FINAL CHECKS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5/6] Final checks ...")

# Check for any remaining nulls
remaining_nulls = df.isnull().sum()
remaining_nulls = remaining_nulls[remaining_nulls > 0]
if len(remaining_nulls) > 0:
    print(f"      [WARN] Remaining nulls found:")
    print(remaining_nulls.to_string())
else:
    print("      No remaining nulls -- dataset is complete")

# Memory usage
mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
print(f"      Final shape  : {df.shape}")
print(f"      Memory usage : {mem_mb:.1f} MB")
print(f"      All numeric  : {all(df.dtypes != object)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — SAVE
# ─────────────────────────────────────────────────────────────────────────────

print("\n[6/6] Saving ...")
df.to_csv(OUTPUT_CSV, index=False)

import os
size_mb = os.path.getsize(OUTPUT_CSV) / 1024 ** 2
print(f"      Saved -> {OUTPUT_CSV}  ({size_mb:.1f} MB)")

print("\n" + "=" * 60)
print("  Done. Next step: model training.")
print(f"  Final dataset: {len(df):,} rows x {len(df.columns)} columns")
print(f"  Target column: 'default'  (0=paid off, 1=defaulted)")
print(f"  Default rate : {df['default'].mean():.2%}")
print("=" * 60)
