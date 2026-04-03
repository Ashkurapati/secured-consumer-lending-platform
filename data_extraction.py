"""
data_extraction.py
------------------
Step 1 of the Stepping Stones ML pipeline.

What this script does:
  1. Loads loan.csv — keeps only the confirmed 60 columns
  2. Drops rows with ambiguous loan_status (Current, Grace Period, etc.)
  3. Maps loan_status → binary target column `default` (0 = paid, 1 = defaulted)
  4. Drops fully empty rows
  5. Saves → clean_loanstats.csv

Usage:
    python data_extraction.py
"""

import os
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

LOAN_CSV    = "loan.csv"
OUTPUT_CSV  = "clean_loanstats.csv"

# ─────────────────────────────────────────────────────────────────────────────
# COLUMNS TO EXTRACT
# (only columns confirmed present in loan.csv — 8 browseNotes/missing cols removed)
# ─────────────────────────────────────────────────────────────────────────────

COLUMNS_TO_KEEP = [
    # --- Target ---
    "loan_status",

    # --- Delinquency & default signals ---
    "acc_now_delinq",
    "delinq_2yrs",
    "delinq_amnt",
    "mths_since_last_delinq",
    "mths_since_last_major_derog",
    "mths_since_last_record",
    "mths_since_recent_bc_dlq",
    "mths_since_recent_revol_delinq",
    "mths_since_recent_inq",
    "num_accts_ever_120_pd",
    "num_tl_30dpd",
    "num_tl_90g_dpd_24m",
    "num_tl_120dpd_2m",
    "pct_tl_nvr_dlq",
    "chargeoff_within_12_mths",
    "collections_12_mths_ex_med",
    "tot_coll_amt",
    "pub_rec",
    "pub_rec_bankruptcies",
    "tax_liens",

    # --- Borrower profile ---
    "annual_inc",
    "dti",
    "emp_length",
    "home_ownership",
    "verification_status",
    "earliest_cr_line",

    # --- Loan details ---
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "purpose",

    # --- Credit utilization & account history ---
    "revol_util",
    "revol_bal",
    "open_acc",
    "open_acc_6m",
    "total_acc",
    "inq_last_6mths",
    "inq_last_12m",
    "avg_cur_bal",
    "bc_util",
    "all_util",
    "il_util",
    "mort_acc",
    "tot_cur_bal",
    "tot_hi_cred_lim",
    "num_rev_accts",
    "num_bc_sats",
    "num_actv_bc_tl",
    "num_actv_rev_tl",
    "num_bc_tl",
    "num_il_tl",
    "num_sats",
    "percent_bc_gt_75",
    "acc_open_past_24mths",
    "mo_sin_old_il_acct",
    "mo_sin_rcnt_tl",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_bal_il",
]

# loan_status values to DROP — outcome unknown, cannot be used for training
AMBIGUOUS_STATUSES = [
    "Current",
    "In Grace Period",
    "Late (16-30 days)",
]

# loan_status → binary default label
DEFAULT_MAP = {
    "Charged Off":                                           1,
    "Default":                                              1,
    "Late (31-120 days)":                                   1,
    "Does not meet the credit policy. Status:Charged Off":  1,
    "Fully Paid":                                           0,
    "Does not meet the credit policy. Status:Fully Paid":   0,
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("  Stepping Stones — Data Extraction")
    print("=" * 60)

    # ── STEP 1: Peek at actual columns in file ───────────────────────────────
    print("\n[1/5] Checking columns in loan.csv ...")
    peek     = pd.read_csv(LOAN_CSV, nrows=0, encoding="utf-8", encoding_errors="ignore")
    file_cols = set(peek.columns)

    available = [c for c in COLUMNS_TO_KEEP if c in file_cols]
    missing   = [c for c in COLUMNS_TO_KEEP if c not in file_cols]

    print(f"      Columns requested : {len(COLUMNS_TO_KEEP)}")
    print(f"      Columns found     : {len(available)}")
    if missing:
        print(f"      Skipped (not in file): {missing}")

    # ── STEP 2: Load only the columns we need ────────────────────────────────
    print("\n[2/5] Loading loan.csv (only selected columns) ...")
    df = pd.read_csv(
        LOAN_CSV,
        usecols=available,
        encoding="utf-8",
        encoding_errors="ignore",
        low_memory=False,
    )
    print(f"      Rows loaded  : {len(df):,}")
    print(f"      Columns      : {len(df.columns)}")
    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"      Memory usage : {mem_mb:.1f} MB")

    # ── STEP 3: Drop ambiguous loan_status rows ──────────────────────────────
    print("\n[3/5] Filtering loan_status ...")
    print(f"      Before : {len(df):,} rows")
    print(f"      loan_status breakdown:")
    print(df["loan_status"].value_counts().to_string())

    df = df[~df["loan_status"].isin(AMBIGUOUS_STATUSES)]
    print(f"\n      After dropping ambiguous rows : {len(df):,} rows")

    # ── STEP 4: Create binary target column `default` ────────────────────────
    print("\n[4/5] Creating target column ...")
    df["default"] = df["loan_status"].map(DEFAULT_MAP)

    unmapped = df["default"].isna().sum()
    if unmapped > 0:
        print(f"      [WARN] {unmapped:,} rows had unrecognised loan_status — dropping them")
        print(f"      Values: {df.loc[df['default'].isna(), 'loan_status'].unique()}")
        df = df.dropna(subset=["default"])

    df["default"] = df["default"].astype("int8")
    df = df.drop(columns=["loan_status"])

    print(f"      Total rows   : {len(df):,}")
    print(f"      Default rate : {df['default'].mean():.2%}")
    print(f"      Class counts :\n{df['default'].value_counts().to_string()}")

    # ── STEP 5: Drop fully empty rows & save ────────────────────────────────
    print("\n[5/5] Saving ...")
    before = len(df)
    df = df.dropna(how="all")
    print(f"      Fully empty rows dropped : {before - len(df):,}")
    print(f"      Final rows   : {len(df):,}")
    print(f"      Final columns: {len(df.columns)}")

    df.to_csv(OUTPUT_CSV, index=False)
    size_mb = os.path.getsize(OUTPUT_CSV) / 1024 ** 2
    print(f"\n      Saved -> {OUTPUT_CSV}  ({size_mb:.1f} MB)")

    print("\n" + "=" * 60)
    print("  Done. Next step: missing value analysis.")
    print("=" * 60)
