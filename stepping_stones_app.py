import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import sqlite3
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Stepping Stones | Loan Assessment",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0D1B2A;
    color: #E8E0D0;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif;
}

.stApp {
    background-color: #0D1B2A;
}

/* Force all Streamlit inputs dark */
input, textarea, [data-baseweb="input"] input, [data-baseweb="textarea"] textarea {
    background-color: #1C2B3A !important;
    color: #E8E0D0 !important;
    border-color: #2E4057 !important;
}

[data-baseweb="select"] > div {
    background-color: #1C2B3A !important;
    color: #E8E0D0 !important;
    border-color: #2E4057 !important;
}

label, .stTextInput label, .stNumberInput label, .stSelectbox label {
    color: #A8B8C8 !important;
}

.stCaption, small {
    color: #7A9AAF !important;
}

/* Header banner */
.header-banner {
    background: #1C2B3A;
    color: #F5F2EC;
    padding: 2.5rem 3rem;
    border-radius: 16px;
    margin-bottom: 2rem;
}

.header-banner h1 {
    font-size: 2.4rem;
    margin: 0;
    color: #E8C97A;
}

.header-banner p {
    margin: 0.4rem 0 0;
    font-size: 1rem;
    opacity: 0.75;
    font-weight: 300;
}

/* Section divider */
.section-divider {
    border-top: 1px solid #2E4057;
    margin: 1.8rem 0 1.4rem 0;
}

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    color: #E8E0D0;
    border-bottom: 2px solid #E8C97A;
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
}

/* Result boxes */
.result-highlight {
    background: #E8C97A;
    color: #0D1B2A;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1rem;
}

.result-highlight .label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4A3800;
    margin-bottom: 0.3rem;
}

.result-highlight .value {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #0D1B2A;
}

.result-sub {
    background: #1C2B3A;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.8rem;
    border-left: 4px solid #E8C97A;
}

.result-sub .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7A9AAF;
    margin-bottom: 0.2rem;
}

.result-sub .value {
    font-size: 1.2rem;
    font-weight: 500;
    color: #E8E0D0;
}

/* Risk badge */
.badge-high {
    background: #4A1515;
    color: #FF8A8A;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}

.badge-low {
    background: #0F3320;
    color: #6FCF97;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}

/* Table styling */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}

.styled-table th {
    background: #0D1B2A;
    color: #E8C97A;
    padding: 0.7rem 1rem;
    text-align: left;
    font-weight: 500;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.styled-table td {
    padding: 0.6rem 1rem;
    border-bottom: 1px solid #2E4057;
    color: #E8E0D0;
}

.styled-table tr:last-child td {
    border-bottom: none;
    font-weight: 600;
    background: #0D1B2A;
}

/* Error */
.error-box {
    background: #3D0F0F;
    border-left: 4px solid #C0392B;
    padding: 1rem 1.4rem;
    border-radius: 8px;
    color: #FF8A8A;
    margin-top: 0.5rem;
}

/* Submit button */
div.stButton > button {
    background: #E8C97A;
    color: #0D1B2A;
    border: none;
    padding: 0.75rem 2.5rem;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    width: 100%;
    transition: background 0.2s;
}

div.stButton > button:hover {
    background: #F5D98A;
}

/* Clear button */
div[data-testid="column"] div.stButton > button.clear-btn,
div.stButton > button[kind="secondary"] {
    background: transparent;
    color: #FF6B6B;
    border: 1.5px solid #FF6B6B;
    font-weight: 500;
}

div.stButton > button[kind="secondary"]:hover {
    background: #3D0F0F;
    color: #FF8A8A;
    border-color: #FF8A8A;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid #2E4057;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
LGD = 0.70

LOAN_LIMITS = {
    "Prime (660–850)": 5000,
    "Subprime (580–659)": 4000,
    "Deep Subprime (< 580)": 3000,
}

RELATIONSHIP_ADJ = {
    "Spouse": -0.03,
    "Parent / Child": -0.02,
    "Sibling": -0.01,
    "Other Relative": 0.00,
    "Friend": 0.02,
    "Other": 0.03,
}

INTEREST_RATES = [2.00, 2.25, 2.50, 2.75, 3.00]

# ─────────────────────────────────────────────
# HIGH RISK ZIP CODES (placeholder set)
# Replace with real zip→default_rate mapping once data arrives
# ─────────────────────────────────────────────
HIGH_RISK_ZIPS = {
    "10001", "10002", "10003", "60601", "60602", "90001", "90002",
    "77001", "77002", "30301", "30302", "85001", "85002",
}  # Extend with real data


def classify_zip(zip_code: str) -> tuple[str, float]:
    """Return (risk_label, zip_fee)."""
    if zip_code.strip() in HIGH_RISK_ZIPS:
        return "High Risk", 0.05
    return "Low Risk", 0.03


# ─────────────────────────────────────────────
# CREDIT BAND HELPER
# ─────────────────────────────────────────────
def get_credit_band(score: int) -> tuple[str, int]:
    if score >= 660:
        return "Prime (660–850)", 5000
    elif score >= 580:
        return "Subprime (580–659)", 4000
    else:
        return "Deep Subprime (< 580)", 3000


# ─────────────────────────────────────────────
# MODEL LOADER (graceful fallback)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return None


def predict_pd(model, credit_score: int, annual_income: float, zip_risk: str) -> float:
    """Return probability of default. Falls back to heuristic if no model."""
    if model is not None:
        zip_encoded = 1 if zip_risk == "High Risk" else 0
        features = np.array([[credit_score, annual_income, zip_encoded]])
        return float(model.predict_proba(features)[0][1])

    # Heuristic fallback (used until real model is trained)
    base = 0.30
    score_adj = (700 - credit_score) * 0.001
    income_adj = max(0, (30000 - annual_income) / 200000)
    zip_adj = 0.05 if zip_risk == "High Risk" else 0.0
    pd_val = base + score_adj + income_adj + zip_adj
    return float(np.clip(pd_val, 0.01, 0.95))


# ─────────────────────────────────────────────
# AFFORDABILITY / DTI CHECK
# ─────────────────────────────────────────────
MAX_DTI = 0.43  # 43% max debt-to-income ratio

def check_affordability(annual_income: float, loan_amount: float, annual_rate_pct: float, term_months: int):
    """
    Returns (approved: bool, reason: str, dti: float)
    Calculates estimated monthly payment on full loan amount and checks against monthly income.
    """
    monthly_income = annual_income / 12

    if monthly_income <= 0:
        return False, "Annual income must be greater than zero.", None

    # Estimate monthly payment on full loan amount (worst case, before collateral reduction)
    monthly_rate = (annual_rate_pct / 100) / 12
    if term_months == 1:
        estimated_payment = loan_amount * (1 + monthly_rate)
    elif monthly_rate == 0:
        estimated_payment = loan_amount / term_months
    else:
        estimated_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** term_months) / \
                            ((1 + monthly_rate) ** term_months - 1)

    dti = estimated_payment / monthly_income

    if dti > MAX_DTI:
        return False, (
            f"DTI ratio is **{dti*100:.1f}%**, which exceeds the maximum allowed **{MAX_DTI*100:.0f}%**. "
            f"Monthly payment of **${estimated_payment:,.2f}** is too high relative to monthly income of **${monthly_income:,.2f}**."
        ), dti

    return True, "Affordability check passed.", dti


# ─────────────────────────────────────────────
# COLLATERAL FORMULA
# ─────────────────────────────────────────────
def compute_collateral(pd_val: float, zip_fee: float, rel_adj: float, loan_amount: float):
    collateral_pct = (pd_val * LGD) + zip_fee + rel_adj
    collateral_pct = max(0.0, collateral_pct)
    collateral_dollar = collateral_pct * loan_amount
    repayment_base = loan_amount - collateral_dollar
    return collateral_pct, collateral_dollar, repayment_base


# ─────────────────────────────────────────────
# REPAYMENT SCHEDULE
# ─────────────────────────────────────────────
def build_schedule(repayment_base: float, annual_rate_pct: float, term_months: int) -> pd.DataFrame:
    monthly_rate = (annual_rate_pct / 100) / 12
    if term_months == 1:
        interest = repayment_base * monthly_rate
        rows = [{
            "Month": 1,
            "Payment": repayment_base + interest,
            "Principal": repayment_base,
            "Interest": interest,
            "Balance": 0.00,
        }]
        return pd.DataFrame(rows)

    if monthly_rate == 0:
        payment = repayment_base / term_months
    else:
        payment = repayment_base * (monthly_rate * (1 + monthly_rate) ** term_months) / \
                  ((1 + monthly_rate) ** term_months - 1)

    rows = []
    balance = repayment_base
    for m in range(1, term_months + 1):
        interest = balance * monthly_rate
        principal = payment - interest
        balance = max(0, balance - principal)
        rows.append({
            "Month": m,
            "Payment": round(payment, 2),
            "Principal": round(principal, 2),
            "Interest": round(interest, 2),
            "Balance": round(balance, 2),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# DATABASE LOGGING
# ─────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect("applications.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            borrower_name TEXT,
            zip_code TEXT,
            annual_income REAL,
            credit_score INTEGER,
            credit_band TEXT,
            loan_amount REAL,
            term_months INTEGER,
            interest_rate REAL,
            co_depositor_name TEXT,
            relationship TEXT,
            zip_risk TEXT,
            pd_value REAL,
            collateral_pct REAL,
            collateral_dollar REAL,
            repayment_base REAL
        )
    """)
    conn.commit()
    conn.close()


def log_application(data: dict):
    try:
        conn = sqlite3.connect("applications.db")
        conn.execute("""
            INSERT INTO applications (
                timestamp, borrower_name, zip_code, annual_income, credit_score,
                credit_band, loan_amount, term_months, interest_rate,
                co_depositor_name, relationship, zip_risk, pd_value,
                collateral_pct, collateral_dollar, repayment_base
            ) VALUES (
                :timestamp, :borrower_name, :zip_code, :annual_income, :credit_score,
                :credit_band, :loan_amount, :term_months, :interest_rate,
                :co_depositor_name, :relationship, :zip_risk, :pd_value,
                :collateral_pct, :collateral_dollar, :repayment_base
            )
        """, data)
        conn.commit()
        conn.close()
    except Exception:
        pass  # Don't crash the app if DB fails


# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────
init_db()
model = load_model()

# ─────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────
defaults = {
    "borrower_name": "",
    "credit_score": 620,
    "annual_income": 42000,
    "zip_code": "",
    "loan_amount": 2500,
    "term_months": 12,
    "interest_rate": 2.50,
    "co_name": "",
    "relationship": "Spouse",
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def clear_form():
    for k, v in defaults.items():
        st.session_state[k] = v

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>🪨 Stepping Stones</h1>
    <p>Collateral-Backed Loan Assessment Tool &nbsp;·&nbsp; Credit Union Pilot</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.info("⚠️ No trained model found (`model.pkl`). Running with heuristic fallback — train and save your model to activate ML predictions.", icon="ℹ️")

# ─────────────────────────────────────────────
# LAYOUT: two columns
# ─────────────────────────────────────────────
left, right = st.columns([1.1, 1], gap="large")

# ─────────────────────────────────────────────
# LEFT: INPUT FORM
# ─────────────────────────────────────────────
with left:

    # --- Borrower Info ---
    st.markdown('<div class="section-title">👤 Borrower Information</div>', unsafe_allow_html=True)
    borrower_name = st.text_input("Borrower Full Name", placeholder="e.g. Maria Garcia", key="borrower_name")
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=1, key="credit_score")
    with col2:
        annual_income = st.number_input("Annual Income ($)", min_value=0, max_value=500000, step=500, key="annual_income")
    zip_code = st.text_input("ZIP Code", placeholder="e.g. 10001", max_chars=5, key="zip_code")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Loan Details ---
    st.markdown('<div class="section-title">💰 Loan Details</div>', unsafe_allow_html=True)
    band, max_loan = get_credit_band(credit_score)
    st.caption(f"Credit Band: **{band}** — Max loan: **${max_loan:,}**")
    loan_amount = st.number_input("Loan Amount ($)", min_value=100, max_value=max_loan, step=100, key="loan_amount")
    col3, col4 = st.columns(2)
    with col3:
        term_months = st.selectbox("Repayment Term", options=[1, 6, 12, 18],
                                   index=[1, 6, 12, 18].index(st.session_state.term_months),
                                   format_func=lambda x: f"{x} month{'s' if x > 1 else ''}",
                                   key="term_months")
    with col4:
        interest_rate = st.selectbox("Interest Rate (APR)", options=INTEREST_RATES,
                                     index=INTEREST_RATES.index(st.session_state.interest_rate),
                                     format_func=lambda x: f"{x:.2f}%",
                                     key="interest_rate")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Co-Depositor ---
    st.markdown('<div class="section-title">🤝 Co-Depositor (Pledgor)</div>', unsafe_allow_html=True)
    co_name = st.text_input("Co-Depositor Full Name", placeholder="e.g. Rosa Garcia", key="co_name")
    relationship = st.selectbox("Relationship to Borrower",
                                options=list(RELATIONSHIP_ADJ.keys()),
                                index=list(RELATIONSHIP_ADJ.keys()).index(st.session_state.relationship),
                                key="relationship")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Submit + Clear buttons ---
    btn_col1, btn_col2 = st.columns([3, 1])
    with btn_col1:
        submitted = st.button("Run Assessment →")
    with btn_col2:
        cleared = st.button("✕ Clear", on_click=clear_form, type="secondary")


# ─────────────────────────────────────────────
# RIGHT: RESULTS
# ─────────────────────────────────────────────
with right:
    if submitted:

        # Validation
        errors = []
        if not borrower_name.strip():
            errors.append("Borrower name is required.")
        if not zip_code.strip() or len(zip_code.strip()) != 5 or not zip_code.strip().isdigit():
            errors.append("Enter a valid 5-digit ZIP code.")
        if loan_amount > max_loan:
            errors.append(f"Loan amount ${loan_amount:,} exceeds the ${max_loan:,} limit for {band}.")

        if errors:
            for e in errors:
                st.markdown(f'<div class="error-box">⚠️ {e}</div>', unsafe_allow_html=True)
        else:
            # ── Affordability Check ──
            affordable, reason, dti = check_affordability(annual_income, loan_amount, interest_rate, term_months)

            if not affordable:
                st.markdown(f"""
                <div style="background:#3D0F0F; border-left:4px solid #C0392B; border-radius:10px; padding:1.5rem 1.8rem; margin-bottom:1rem;">
                    <div style="font-size:1.1rem; font-weight:700; color:#FF6B6B; margin-bottom:0.5rem;">❌ Loan Not Approved</div>
                    <div style="color:#FFB3B3; font-size:0.95rem; line-height:1.6;">{reason}</div>
                    <div style="margin-top:1rem; color:#FF8A8A; font-size:0.85rem;">
                        To qualify, the borrower would need a higher income, a smaller loan amount, or a longer repayment term.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Show DTI as a passing info badge
                st.markdown(f"""
                <div style="background:#0F3320; border-left:4px solid #6FCF97; border-radius:8px; padding:0.8rem 1.2rem; margin-bottom:1rem;">
                    <span style="color:#6FCF97; font-size:0.85rem;">✓ Affordability check passed &nbsp;·&nbsp; DTI: <strong>{dti*100:.1f}%</strong> (max {MAX_DTI*100:.0f}%)</span>
                </div>
                """, unsafe_allow_html=True)

                # ── Compute ──
                zip_risk, zip_fee = classify_zip(zip_code)
                rel_adj = RELATIONSHIP_ADJ[relationship]
                pd_val = predict_pd(model, credit_score, annual_income, zip_risk)
                collateral_pct, collateral_dollar, repayment_base = compute_collateral(pd_val, zip_fee, rel_adj, loan_amount)
                schedule = build_schedule(repayment_base, interest_rate, term_months)
                total_interest = schedule["Interest"].sum()
                total_repaid = schedule["Payment"].sum()

                # ── Log ──
                log_application({
                    "timestamp": datetime.now().isoformat(),
                    "borrower_name": borrower_name,
                    "zip_code": zip_code,
                    "annual_income": annual_income,
                    "credit_score": credit_score,
                    "credit_band": band,
                    "loan_amount": loan_amount,
                    "term_months": term_months,
                    "interest_rate": interest_rate,
                    "co_depositor_name": co_name,
                    "relationship": relationship,
                    "zip_risk": zip_risk,
                    "pd_value": pd_val,
                    "collateral_pct": collateral_pct,
                    "collateral_dollar": collateral_dollar,
                    "repayment_base": repayment_base,
                })

                # ── Display ──
                badge = "badge-high" if zip_risk == "High Risk" else "badge-low"

                st.markdown(f"""
                <div class="result-highlight">
                    <div class="label">Collateral to Pledge</div>
                    <div class="value">${collateral_dollar:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class="result-sub">
                        <div class="label">Repayment Base</div>
                        <div class="value">${repayment_base:,.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="result-sub">
                        <div class="label">Collateral %</div>
                        <div class="value">{collateral_pct*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                c3, c4 = st.columns(2)
                with c3:
                    st.markdown(f"""
                    <div class="result-sub">
                        <div class="label">Est. Default Probability</div>
                        <div class="value">{pd_val*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c4:
                    st.markdown(f"""
                    <div class="result-sub">
                        <div class="label">ZIP Risk</div>
                        <div class="value"><span class="{badge}">{zip_risk}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                # Adjustments breakdown
                st.markdown('<hr>', unsafe_allow_html=True)
                st.markdown('<div class="section-title" style="font-size:1rem;">Formula Breakdown</div>', unsafe_allow_html=True)
                breakdown_df = pd.DataFrame([
                    {"Component": "PD × LGD", "Value": f"{pd_val*LGD*100:.2f}%"},
                    {"Component": f"ZIP Fee ({zip_risk})", "Value": f"{zip_fee*100:.0f}%"},
                    {"Component": f"Relationship Adj. ({relationship})", "Value": f"{rel_adj*100:+.0f}%"},
                    {"Component": "Total Collateral %", "Value": f"{collateral_pct*100:.2f}%"},
                ])
                st.dataframe(breakdown_df, hide_index=True, use_container_width=True)

                # Repayment schedule
                st.markdown('<hr>', unsafe_allow_html=True)
                st.markdown('<div class="section-title" style="font-size:1rem;">Repayment Schedule</div>', unsafe_allow_html=True)
                st.caption(f"On repayment base of **${repayment_base:,.2f}** at **{interest_rate:.2f}% APR** over **{term_months} month(s)**")

                display_schedule = schedule.copy()
                display_schedule["Payment"] = display_schedule["Payment"].apply(lambda x: f"${x:,.2f}")
                display_schedule["Principal"] = display_schedule["Principal"].apply(lambda x: f"${x:,.2f}")
                display_schedule["Interest"] = display_schedule["Interest"].apply(lambda x: f"${x:,.2f}")
                display_schedule["Balance"] = display_schedule["Balance"].apply(lambda x: f"${x:,.2f}")
                st.dataframe(display_schedule, hide_index=True, use_container_width=True)

                c5, c6 = st.columns(2)
                with c5:
                    st.markdown(f"""
                    <div class="result-sub">
                        <div class="label">Total Interest Paid</div>
                        <div class="value">${total_interest:,.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c6:
                    st.markdown(f"""
                    <div class="result-sub">
                        <div class="label">Total Repaid</div>
                        <div class="value">${total_repaid:,.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 5rem 2rem; color: #4A6070;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🪨</div>
            <div style="font-family:'Playfair Display',serif; font-size:1.3rem; color:#E8C97A; margin-bottom:0.5rem;">
                Fill in the form to run an assessment
            </div>
            <div style="font-size:0.9rem; line-height:1.6; color:#7A9AAF;">
                Enter borrower info, loan details, and co-depositor<br>
                to calculate collateral and repayment schedule.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; font-size:0.78rem; color:#4A6070; padding-bottom:1rem;">
    Stepping Stones Pilot · For internal credit union use only · Not financial advice
</div>
""", unsafe_allow_html=True)