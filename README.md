# Stepping Stones - Secured Consumer Lending Platform

An end-to-end machine learning platform for **Probability of Default (PD)** modelling, built for Stepping Stones.

The system replaces manual risk rules with a LightGBM model trained on 1.33 million real-world LendingClub loans, and exposes it through a live web application that computes collateral requirements, interest rates, and repayment schedules in real time.

---

## Problem Statement

Traditional credit scoring excludes a large segment of subprime borrowers. Stepping Stones uses collateral-backed lending to serve this population, but needs a reliable way to estimate default risk so it can price collateral requirements accurately and fairly.

---

## Repository Structure

```
secured-consumer-lending-platform/
│
├── peek_columns.py          # Step 0: Validates columns present in loan.csv
├── data_extraction.py       # Step 1: Extracts 63 columns, creates binary target
├── data_cleaning.py         # Step 2: Imputes nulls, encodes categoricals
├── eda.py                   # Step 3: Exploratory data analysis (9 charts)
├── split_data.py            # Step 4: Stratified 80/20 train/test split
├── train_model.py           # Step 5: Trains LightGBM PD model, saves model.pkl
├── tune_model.py            # Step 6: Hyperparameter tuning via RandomizedSearchCV
├── count.py                 # Utility: default rate breakdown by home ownership
│
├── model.pkl                # Trained LightGBM model (AUC 0.7374)
│
├── eda_charts/              # Generated EDA visualisations (9 PNG files)
│
├── website/
│   ├── api.py               # FastAPI backend - loads model, serves predictions
│   ├── index.html           # Multi-step loan application frontend
│   ├── script.js            # Form logic, validation, API calls
│   └── style.css            # Styling
│
├── stepping_stones_app.py   # Initial beta Streamlit prototype
│
├── requirements.txt         # Pinned Python dependencies
├── run.py                   # Single entry point - starts the API from project root
├── Dockerfile               # Docker image definition
├── docker-compose.yml       # Docker Compose configuration
├── Procfile                 # Render deployment start command
└── runtime.txt              # Python version pin for Render
```

---

## Dataset

- **Source:** LendingClub loan data (`loan.csv`) - 2.26M rows, 145 columns
- **After filtering:** 1,328,284 rows with confirmed loan outcomes
- **Target variable:** `default` - binary (0 = Fully Paid, 1 = Defaulted)
- **Class distribution:** 78.6% paid off / 21.4% defaulted
- **Features used:** 74 columns covering delinquency signals, borrower profile, loan details, and credit utilisation

**Fair lending compliance:** Geographic columns (ZIP code, state, MSA) are excluded per ECOA and Fair Housing Act requirements. Post-origination columns (debt settlement, hardship flags, recoveries) are excluded to prevent data leakage.

---

## ML Pipeline

### Step 0 - Column Validation (`peek_columns.py`)
Reads the header of `loan.csv` and cross-references against planned columns.

### Step 1 - Data Extraction (`data_extraction.py`)
- Loads `loan.csv` using `usecols` for memory efficiency (~650MB vs 2.9GB full load)
- Drops ambiguous loan statuses: `Current`, `In Grace Period`, `Late (16–30 days)`
- Maps `loan_status` to binary `default` column (0 = paid, 1 = defaulted)
- Output: `clean_loanstats.csv` - 1,328,284 rows × 63 columns

### Step 2 - Data Cleaning (`data_cleaning.py`)
- `mths_since_*` columns: null → `999` (event never occurred - domain-driven signal)
- Thin credit file columns: null → `0`
- `pct_tl_nvr_dlq`: null → `100`
- `emp_length`: unknown → `-1` (sentinel, not confused with 0 years)
- `earliest_cr_line` → `credit_history_months` (months from reference date 2018-01-01)
- `grade`: ordinal encoded A=1 to G=7
- `home_ownership`: ordinal OWN=4, MORTGAGE=3, RENT=2, OTHER=1
- `purpose`: one-hot encoded → 14 columns
- Output: `cleaned_loanstats.csv` — 1,328,284 rows × 76 columns, zero nulls, all numeric

### Step 3 - Exploratory Data Analysis (`eda.py`)
Generates 9 charts saved to `eda_charts/`. Key findings:
- `grade` shows monotonic default increase: A (6.6%) → G (51.2%)
- `int_rate` and `sub_grade` are the strongest predictors
- 60-month loans default at 2× the rate of 36-month loans
- Small business loans have the highest default rate by purpose

### Step 4 - Train/Test Split (`split_data.py`)
- Stratified 80/20 split — preserves the 78/22 class ratio in both sets
- Output: `train.csv` (1,062,627 rows) and `test.csv` (265,657 rows)

### Step 5 - Model Training (`train_model.py`)
Trained on Darwin HPC using SLURM workload manager.

| Model | AUC | KS Statistic |
|---|---|---|
| Logistic Regression (baseline) | 0.6324 | — |
| **LightGBM** | **0.7374** | **0.3474** |

- Class imbalance handled with `class_weight='balanced'`
- Best model saved as `model.pkl`

### Step 6 - Hyperparameter Tuning (`tune_model.py`)
- RandomizedSearchCV - 20 combinations on a 200K sample
- Tuned AUC: 0.7332 < baseline 0.7374 → baseline model kept

---

## Why LightGBM

- Tabular financial data - tree models consistently outperform neural networks
- **Legal explainability:** ECOA/CFPB requires lenders to explain denial reasons - LightGBM provides feature importance and SHAP values; neural networks are black boxes
- Fast inference with no GPU required at prediction time

---

## Credit Risk Business Logic

### Probability of Default (PD)
The model outputs a PD score between 0 and 1 for each applicant. The borrower's credit score is mapped to an internal interest rate which serves as the model's primary risk signal.

### Collateral Formula
```
Collateral = (PD × LGD × Loan Amount) + (Buffer % × Loan Amount)
```

**LGD (Loss Given Default) = 0.70** - industry standard for unsecured micro personal loans; assumes 30 cents recovery per dollar on default.

**Guarantor buffers:**

| Relationship | Buffer |
|---|---|
| Spouse / Partner | 5% |
| Family Member | 10% |
| Friend | 15% |
| No Guarantor | 25% |

**Zero collateral rule:** waived entirely if credit score > 700 AND DTI < 40% AND credit history ≥ 2 years.

**High risk rule:** PD > 0.80 or credit score < 500 → 100% collateral (loan amount = collateral).

### Interest Rate
Range: 12%–16% p.a., formula: `rate = 12% + (PD × 4%)`, rounded to nearest 0.5%.

### Loan Parameters
- Loan amount: $200–$5,000
- Repayment terms: 12, 18, 24, 36, or 48 months

---

## Web Application

A four-step loan application built with HTML/CSS/JavaScript frontend and FastAPI backend.

**Step 1 - Personal Details:** name, date of birth (18+ required), home ownership, employment length

**Step 2 - Financial Profile:** income, credit score, monthly debt, revolving credit, overall credit profile, delinquency history

**Step 3 - Guarantor:** optional co-signer with relationship type

**Step 4 - Decision:** instant approval with interest rate, collateral required, monthly payment, and full amortisation schedule or rejection with reason

### Live Demo
**https://steppingstones-q6q3.onrender.com**

---

### Option 1 - Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the API (from project root)
python run.py
```

Then open `http://localhost:8000` in your browser.

---

### Option 2 - Docker (works on any machine, fully offline after first build)

**Requirements:** Docker Desktop installed

```bash
# Clone the repo
git clone https://github.com/Ashkurapati/secured-consumer-lending-platform.git
cd secured-consumer-lending-platform

# Build and run (first time — needs internet)
docker-compose up --build

# Every time after (fully offline)
docker-compose up
```

Then open `http://localhost:8000` in your browser.

To stop:
```bash
docker-compose down
```

---

### Option 3 - Render (live public deployment)

Already deployed at **https://steppingstones-q6q3.onrender.com**

To deploy your own instance:
1. Fork this repo
2. Create a new Web Service on render.com
3. Connect your GitHub repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `python run.py`
6. Add environment variable: `PYTHON_VERSION = 3.11.9`

---

## Fair Lending Compliance

- No geographic features (ZIP code, state, MSA)
- No race, gender, age, or national origin proxies
- No post origination features (data leakage prevention)
- Model explainability via LightGBM feature importance satisfies ECOA adverse action notice requirements

---

## Prerequisites

```bash
Python 3.8+
pip install -r requirements.txt
```

> **Note:** `loan.csv` is not included due to file size. The LendingClub dataset can be obtained from Kaggle.

---

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
