# Secured Consumer Lending Alternative to Person to Person Lending

## Overview
# Stepping Stones — Secured Consumer Lending Platform

A machine learning pipeline for **Probability of Default (PD)** modelling, built for Stepping Stones — a collateral-backed loan assessment platform designed for credit unions serving subprime borrowers.

The goal is to replace the manually coded risk rules in the app with a real ML model trained on 1.33 million LendingClub loans.


---

## Problem Statement

Traditional credit scoring excludes a large segment of subprime borrowers. Stepping Stones uses collateral-backed lending to serve this population, but needs a reliable way to estimate default risk so it can price collateral requirements accurately.

## Dataset

- **Source:** LendingClub loan data (`loan.csv`) — 2.26M rows, 145 columns
- **After filtering:** 1,328,284 rows with confirmed loan outcomes
- **Target variable:** `default` — binary (0 = Fully Paid, 1 = Defaulted)
- **Class distribution:** 78.6% paid off / 21.4% defaulted
- **Features used:** 75 columns covering delinquency signals, borrower profile, loan details, and credit utilisation

**Fair lending compliance:** Geographic columns (ZIP code, state, MSA) are excluded per ECOA and Fair Housing Act requirements. Post-origination columns (debt settlement, hardship flags, recoveries) are excluded to prevent data leakage.

---

## Repository Structure
- `src/` – source code
- `docs/` – optional documentation (Sphinx scaffold)
- `data/` – input/output data (if applicable)
```
secured-consumer-lending-platform/
│
├── peek_columns.py          # Step 0: Validates columns present in loan.csv
├── data_extraction.py       # Step 1: Extracts 63 columns, creates binary target
├── data_cleaning.py         # Step 2: Imputes nulls, encodes categoricals
├── eda.py                   # Step 3: Exploratory data analysis (9 charts)
├── split_data.py            # Step 4: Stratified 80/20 train/test split
│
├── eda_charts/              # Generated EDA visualisations (9 PNG files)
│
├── stepping_stones_app.py   # Streamlit app (awaiting model.pkl integration)
│
└── docs/                    # Sphinx documentation scaffold
```

---

## ML Pipeline

### Step 0 — Column Validation (`peek_columns.py`)
Reads the header of `loan.csv` and cross-references against planned columns. Identifies which columns are present and which are missing before loading the full dataset.

### Step 1 — Data Extraction (`data_extraction.py`)
- Loads `loan.csv` using `usecols` for memory efficiency (~650MB vs 2.9GB full load)
- Drops ambiguous loan statuses: `Current`, `In Grace Period`, `Late (16-30 days)`
- Maps `loan_status` to binary `default` column (0 = paid, 1 = defaulted)
- Output: `clean_loanstats.csv` — 1,328,284 rows × 63 columns

### Step 2 — Data Cleaning (`data_cleaning.py`)
- `mths_since_*` columns: null → `999` (event never occurred — domain-driven signal)
- Thin credit file columns: null → `0` (no such account type exists)
- `pct_tl_nvr_dlq`: null → `100` (never delinquent)
- `emp_length`: unknown → `-1` (out-of-range sentinel, not confused with 0 years)
- `earliest_cr_line` → `credit_history_months` (months from reference date 2018-01-01)
- `grade`: ordinal encoded A=1 to G=7
- `sub_grade`: ordinal encoded A1=1 to G5=35
- `home_ownership`: ordinal OWN=4, MORTGAGE=3, RENT=2, OTHER=1
- `verification_status`: Verified=2, Source Verified=1, Not Verified=0
- `purpose`: one-hot encoded → 14 columns
- Output: `cleaned_loanstats.csv` — 1,328,284 rows × 76 columns, zero nulls, all numeric

### Step 3 — Exploratory Data Analysis (`eda.py`)
Generates 9 charts saved to `eda_charts/`:
- Class balance
- Default rate by grade, purpose, home ownership, and loan term
- Feature distributions
- Outlier check
- Risk bucket analysis (DTI, income, interest rate)
- Correlation heatmap (top 20 features)

**Key findings:**
- `grade` shows perfect monotonic default increase: A (6.6%) → G (51.2%)
- `sub_grade` and `int_rate` are the strongest predictors (correlation ~0.27 with default)
- 60-month loans default at 2x the rate of 36-month loans
- Small business loans have the highest default rate by purpose
- `funded_amnt` is near-identical to `loan_amnt` — dropped before training

### Step 4 — Train/Test Split (`split_data.py`)
- Drops `funded_amnt` (redundant with `loan_amnt`)
- Stratified 80/20 split — preserves the 78/22 class ratio in both sets
- Output: `train.csv` (1,062,627 rows) and `test.csv` (265,657 rows)

---


## Getting Started
1. Clone the repository
2. Create a feature branch
3. Open a pull request early

## Documentation
This repository includes an optional Sphinx documentation scaffold.

### Prerequisites
```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn
```

### Run the pipeline
```bash
python peek_columns.py       # Check columns in loan.csv
python data_extraction.py    # Extract and label data
python data_cleaning.py      # Clean and encode
python eda.py                # Generate EDA charts
python split_data.py         # Train/test split
```

> **Note:** `loan.csv` is not included in this repository due to file size. The LendingClub dataset can be obtained from Kaggle.

---

## Fair Lending Compliance

This model complies with ECOA and the Fair Housing Act:
- No geographic features (ZIP code, state, MSA)
- No race, gender, age, or national origin proxies
- No post-origination features (data leakage prevention)

---

## Contributing
All changes must go through pull requests.
