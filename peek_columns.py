"""
peek_columns.py
---------------
Peeks inside loan.csv and tells you:
  1. Every column actually in the file
  2. Which of our 67 planned columns are present
  3. Which are missing
  4. Total row count

Run this first before data_extraction.py.
"""

import pandas as pd

LOAN_CSV = "loan.csv"

# All 67 columns we planned to use
PLANNED_COLS = [
    "loan_status",
    # Delinquency signals
    "acc_now_delinq", "delinq_2yrs", "delinq_amnt",
    "mths_since_last_delinq", "mths_since_last_major_derog",
    "mths_since_recent_bc_dlq", "mths_since_recent_revol_delinq",
    "num_accts_ever_120_pd", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_120dpd_2m", "pct_tl_nvr_dlq", "chargeoff_within_12_mths",
    "collections_12_mths_ex_med", "tot_coll_amt", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "pub_rec",
    "pub_rec_bankruptcies", "tax_liens",
    # Indirect default / hardship signals
    "debt_settlement_flag", "settlement_status", "settlement_amount",
    "settlement_percentage", "hardship_flag", "hardship_status",
    "hardship_loan_status", "hardship_dpd", "pymnt_plan",
    # Borrower profile
    "annual_inc", "dti", "fico_range_low", "fico_range_high",
    "last_fico_range_low", "last_fico_range_high", "emp_length",
    "home_ownership", "verification_status", "earliest_cr_line",
    # Loan details
    "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "purpose", "issue_d",
    # Credit utilization
    "revol_util", "revol_bal", "open_acc", "total_acc",
    "inq_last_6mths", "avg_cur_bal", "bc_util", "all_util",
    "il_util", "mort_acc", "tot_cur_bal", "tot_hi_cred_lim",
    "num_rev_accts", "num_bc_sats",
    # browseNotes unique adds
    "expDefaultRate", "effective_int_rate", "reviewStatus",
    "mthsSinceRecentLoanDelinq",
]

print("=" * 60)
print("  Peeking inside loan.csv ...")
print("=" * 60)

# Read just the header row — zero data rows loaded
df_peek = pd.read_csv(LOAN_CSV, nrows=0, encoding="utf-8", encoding_errors="ignore")
file_cols = list(df_peek.columns)

print(f"\nTotal columns in file : {len(file_cols)}")

# Count rows without loading everything
print("Counting rows (this may take a moment)...")
row_count = sum(1 for _ in open(LOAN_CSV, encoding="utf-8", errors="ignore")) - 1
print(f"Total rows in file    : {row_count:,}")

# Cross-reference
file_cols_lower = {c.lower(): c for c in file_cols}
planned_lower   = {c.lower(): c for c in PLANNED_COLS}

found   = [planned_lower[c] for c in planned_lower if c in file_cols_lower]
missing = [planned_lower[c] for c in planned_lower if c not in file_cols_lower]
extra   = [c for c in file_cols if c.lower() not in planned_lower]

print(f"\n{'='*60}")
print(f"  PLANNED columns found in file   : {len(found)}")
print(f"  PLANNED columns NOT in file     : {len(missing)}")
print(f"  Extra columns in file (unused)  : {len(extra)}")
print(f"{'='*60}")

print(f"\nFOUND ({len(found)}):")
for c in sorted(found):
    print(f"     {c}")

print(f"\nMISSING from file ({len(missing)}):")
for c in sorted(missing):
    print(f"     {c}")

print(f"\nFICO-related columns in loan.csv:")
fico_cols = [c for c in file_cols if "fico" in c.lower() or "score" in c.lower() or "credit" in c.lower()]
for c in fico_cols:
    print(f"     {c}")

print(f"\nALL columns in loan.csv:")
for i, c in enumerate(file_cols, 1):
    print(f"  {i:3d}.  {c}")
