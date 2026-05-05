"""
api.py — Stepping Stones FastAPI Backend
-----------------------------------------
Loads model.pkl, accepts loan application data,
computes PD, collateral, interest rate, repayment plan.

Install:
    pip install fastapi uvicorn

Run:
    uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Stepping Stones Lending API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
MODEL_PATH = os.path.join(BASE_DIR, "..", "model.pkl")
model = joblib.load(MODEL_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# MAPPINGS
# ─────────────────────────────────────────────────────────────────────────────

def credit_score_to_int_rate(cs: int) -> float:
    if cs >= 750: return 7.51
    elif cs >= 720: return 9.93
    elif cs >= 700: return 11.99
    elif cs >= 680: return 13.67
    elif cs >= 660: return 15.61
    elif cs >= 640: return 17.86
    elif cs >= 620: return 19.99
    elif cs >= 600: return 22.00
    elif cs >= 580: return 24.99
    elif cs >= 550: return 27.49
    else: return 29.99

def credit_score_to_grade(cs: int) -> int:
    if cs >= 750: return 1
    elif cs >= 700: return 2
    elif cs >= 660: return 3
    elif cs >= 620: return 4
    elif cs >= 580: return 5
    elif cs >= 540: return 6
    else: return 7

def grade_to_subgrade(grade: int, cs: int) -> int:
    ranges = {1:(750,850), 2:(700,749), 3:(660,699), 4:(620,659),
              5:(580,619), 6:(540,579), 7:(500,539)}
    lo, hi = ranges[grade]
    pos = (cs - lo) / max(hi - lo, 1)
    sub = 5 - min(4, int(pos * 5))
    return (grade - 1) * 5 + max(1, min(5, sub))

def calc_monthly_payment(principal: float, annual_rate_pct: float, term_months: int) -> float:
    r = annual_rate_pct / 100 / 12
    if r == 0:
        return principal / term_months
    return principal * r * (1 + r)**term_months / ((1 + r)**term_months - 1)

# ─────────────────────────────────────────────────────────────────────────────
# INPUT / OUTPUT MODELS
# ─────────────────────────────────────────────────────────────────────────────

class LoanApplication(BaseModel):
    # Personal
    first_name: str
    last_name: str
    home_ownership: str           # OWN, MORTGAGE, RENT, OTHER
    emp_length: int               # years employed, -1 = unknown

    # Loan
    loan_amnt: float
    purpose: str                  # debt_consolidation, car, etc.
    term: int                     # 36 or 60

    # Financial — top 15 features + delinquency signals
    annual_inc: float
    credit_score: int
    monthly_debt: float           # used to derive DTI
    revol_util: float             # revolving utilisation %
    revol_bal: float              # current revolving balance
    total_bc_limit: float         # total credit card limit
    total_acc: int                # total credit accounts
    avg_cur_bal: float            # average balance per account
    total_bal_ex_mort: float      # total debt excluding mortgage
    tot_hi_cred_lim: float        # total high credit limit across all accounts
    mo_sin_old_il_acct: int       # months since oldest installment account
    delinq_2yrs: int              # late payments in past 2 years
    pub_rec: int                  # public records
    pub_rec_bankruptcies: int     # bankruptcies

    # Guarantor
    has_guarantor: bool
    guarantor_relationship: Optional[str] = None  # spouse, family, friend, none

# ─────────────────────────────────────────────────────────────────────────────
# PREDICT ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/predict")
def predict(data: LoanApplication):

    # ── Rejection checks ─────────────────────────────────────────────────────
    if data.loan_amnt < 200 or data.loan_amnt > 5000:
        return {
            "status": "rejected",
            "rejection_reason": "Loan amount must be between $200 and $5,000.",
            "pd_score": None, "interest_rate": None,
            "collateral_required": None, "monthly_payment": None,
            "total_repayment": None, "total_interest": None,
            "repayment_schedule": []
        }

    # ── Derived values ────────────────────────────────────────────────────────
    int_rate  = credit_score_to_int_rate(data.credit_score)
    grade     = credit_score_to_grade(data.credit_score)
    sub_grade = grade_to_subgrade(grade, data.credit_score)

    home_map = {"OWN": 4, "MORTGAGE": 3, "RENT": 2, "OTHER": 1}
    home_enc = home_map.get(data.home_ownership.upper(), 2)

    dti = (data.monthly_debt * 12 / data.annual_inc * 100) if data.annual_inc > 0 else 0

    # Model term: training data only had 36/60 months — use 36 as closest proxy
    # Actual repayment term (12/18/24/36/48) is used for the repayment schedule below
    model_term = 36
    r_m = int_rate / 100 / 12
    installment = (data.loan_amnt * r_m * (1 + r_m)**model_term /
                   ((1 + r_m)**model_term - 1)) if r_m > 0 else data.loan_amnt / model_term

    bc_util = (data.revol_bal / data.total_bc_limit * 100) if data.total_bc_limit > 0 else 0

    # Purpose one-hot
    all_purposes = ["car", "credit_card", "debt_consolidation", "educational",
                    "home_improvement", "house", "major_purchase", "medical",
                    "moving", "other", "renewable_energy", "small_business",
                    "vacation", "wedding"]
    purpose_features = {f"purpose_{p}": (1 if p == data.purpose else 0)
                        for p in all_purposes}

    # ── Build full 74-feature vector ─────────────────────────────────────────
    features = {
        # Delinquency signals
        "acc_now_delinq":               0,
        "delinq_2yrs":                  data.delinq_2yrs,
        "delinq_amnt":                  0,
        "mths_since_last_delinq":       999 if data.delinq_2yrs == 0 else 18,
        "mths_since_last_major_derog":  999,
        "mths_since_last_record":       999 if data.pub_rec == 0 else 36,
        "mths_since_recent_bc_dlq":     999 if data.delinq_2yrs == 0 else 18,
        "mths_since_recent_revol_delinq": 999 if data.delinq_2yrs == 0 else 18,
        "mths_since_recent_inq":        3,
        "num_accts_ever_120_pd":        0,
        "num_tl_30dpd":                 0,
        "num_tl_90g_dpd_24m":           0,
        "num_tl_120dpd_2m":             0,
        "pct_tl_nvr_dlq":              100 if data.delinq_2yrs == 0 else 85,
        "chargeoff_within_12_mths":     0,
        "collections_12_mths_ex_med":   0,
        "tot_coll_amt":                 0,
        "pub_rec":                      data.pub_rec,
        "pub_rec_bankruptcies":         data.pub_rec_bankruptcies,
        "tax_liens":                    0,
        # Borrower profile
        "annual_inc":                   data.annual_inc,
        "dti":                          dti,
        "emp_length":                   data.emp_length,
        "home_ownership":               home_enc,
        "verification_status":          1,
        "credit_history_months":        data.mo_sin_old_il_acct,
        # Loan details
        "loan_amnt":                    data.loan_amnt,
        "term":                         model_term,
        "int_rate":                     int_rate,
        "installment":                  installment,
        "grade":                        grade,
        "sub_grade":                    sub_grade,
        # Credit utilisation
        "revol_util":                   data.revol_util,
        "revol_bal":                    data.revol_bal,
        "open_acc":                     max(1, int(data.total_acc * 0.7)),
        "open_acc_6m":                  1,
        "total_acc":                    data.total_acc,
        "inq_last_6mths":               1,
        "inq_last_12m":                 2,
        "avg_cur_bal":                  data.avg_cur_bal,
        "bc_util":                      bc_util,
        "all_util":                     data.revol_util,
        "il_util":                      50.0,
        "mort_acc":                     1 if data.home_ownership.upper() == "MORTGAGE" else 0,
        "tot_cur_bal":                  data.avg_cur_bal * data.total_acc,
        "tot_hi_cred_lim":              data.tot_hi_cred_lim,
        "num_rev_accts":                max(1, int(data.total_acc * 0.4)),
        "num_bc_sats":                  2,
        "num_actv_bc_tl":               2,
        "num_actv_rev_tl":              3,
        "num_bc_tl":                    3,
        "num_il_tl":                    max(1, int(data.total_acc * 0.3)),
        "num_sats":                     data.total_acc,
        "percent_bc_gt_75":             75.0 if bc_util > 75 else 10.0,
        "acc_open_past_24mths":         2,
        "mo_sin_old_il_acct":           data.mo_sin_old_il_acct,
        "mo_sin_rcnt_tl":               6,
        "total_bal_ex_mort":            data.total_bal_ex_mort,
        "total_bc_limit":               data.total_bc_limit,
        "total_bal_il":                 data.total_bal_ex_mort * 0.3,
    }
    features.update(purpose_features)

    df_input = pd.DataFrame([features])

    # Reorder columns to match model's expected feature order
    try:
        df_input = df_input[model.feature_name_]
    except Exception:
        pass  # LightGBM will match by name

    # ── Predict PD ───────────────────────────────────────────────────────────
    pd_score = float(model.predict_proba(df_input)[0][1])

    # ── Collateral ────────────────────────────────────────────────────────────
    high_risk = data.credit_score < 500 or pd_score > 0.80

    if high_risk:
        collateral = data.loan_amnt  # 100% collateral for very high risk
    else:
        credit_history_years = data.mo_sin_old_il_acct / 12
        if (data.credit_score > 700 and dti < 40 and credit_history_years >= 2):
            collateral = 0.0
        else:
            LGD = 0.70
            expected_loss = pd_score * LGD * data.loan_amnt
            guarantor_buffer = {
                "spouse": 0.05, "family": 0.10,
                "friend": 0.15, "none": 0.25, None: 0.25
            }
            buffer_pct = guarantor_buffer.get(
                data.guarantor_relationship.lower() if data.guarantor_relationship else None, 0.25
            )
            collateral = expected_loss + (buffer_pct * data.loan_amnt)

    # ── Interest rate (12–16%, continuous, rounded to nearest 0.5%) ──────────
    raw_rate = 12.0 + (pd_score * 4.0)
    interest_rate = round(min(16.0, max(12.0, raw_rate)) * 2) / 2

    # ── Repayment schedule ────────────────────────────────────────────────────
    monthly_payment = calc_monthly_payment(data.loan_amnt, interest_rate, data.term)
    r = interest_rate / 100 / 12
    balance = data.loan_amnt
    schedule = []
    for month in range(1, data.term + 1):
        interest_pmt  = balance * r
        principal_pmt = monthly_payment - interest_pmt
        balance       = max(0.0, balance - principal_pmt)
        schedule.append({
            "month":     month,
            "payment":   round(monthly_payment, 2),
            "principal": round(principal_pmt, 2),
            "interest":  round(interest_pmt, 2),
            "balance":   round(balance, 2),
        })

    total_repayment = monthly_payment * data.term
    total_interest  = total_repayment - data.loan_amnt

    return {
        "status":             "approved",
        "rejection_reason":   None,
        "pd_score":           round(pd_score, 4),
        "interest_rate":      interest_rate,
        "collateral_required": round(collateral, 2),
        "monthly_payment":    round(monthly_payment, 2),
        "total_repayment":    round(total_repayment, 2),
        "total_interest":     round(total_interest, 2),
        "repayment_schedule": schedule,
    }

# Serve frontend — must come after all API routes
app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="static")
