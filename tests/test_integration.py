"""
Integration / end-to-end tests for the Stepping Stones FastAPI backend.

Tests the full request → model → response pipeline via TestClient,
which starts the app in-process without needing a running server.

Run with:
    pytest tests/test_integration.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "website"))

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

# ─────────────────────────────────────────────────────────────────────────────
# Reusable base payload — a realistic mid-range applicant
# ─────────────────────────────────────────────────────────────────────────────

BASE_PAYLOAD = {
    "first_name":            "Test",
    "last_name":             "User",
    "home_ownership":        "RENT",
    "emp_length":            3,
    "loan_amnt":             2000,
    "purpose":               "debt_consolidation",
    "term":                  36,
    "annual_inc":            48000,
    "credit_score":          650,
    "monthly_debt":          400,
    "revol_util":            40.0,
    "revol_bal":             5000,
    "total_bc_limit":        12500,
    "total_acc":             8,
    "avg_cur_bal":           2500,
    "total_bal_ex_mort":     8000,
    "tot_hi_cred_lim":       20000,
    "mo_sin_old_il_acct":    60,
    "delinq_2yrs":           0,
    "pub_rec":               0,
    "pub_rec_bankruptcies":  0,
    "has_guarantor":         False,
    "guarantor_relationship": "none",
}


# ─────────────────────────────────────────────────────────────────────────────
# Happy path — end-to-end workflow
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndApproval:

    def test_valid_application_returns_200(self):
        response = client.post("/api/predict", json=BASE_PAYLOAD)
        assert response.status_code == 200

    def test_valid_application_approved(self):
        body = client.post("/api/predict", json=BASE_PAYLOAD).json()
        assert body["status"] == "approved"

    def test_response_contains_all_required_fields(self):
        body = client.post("/api/predict", json=BASE_PAYLOAD).json()
        required = [
            "status", "pd_score", "interest_rate",
            "collateral_required", "monthly_payment",
            "total_repayment", "total_interest", "repayment_schedule",
        ]
        for field in required:
            assert field in body, f"Missing field: {field}"

    def test_pd_score_is_valid_probability(self):
        body = client.post("/api/predict", json=BASE_PAYLOAD).json()
        assert 0.0 <= body["pd_score"] <= 1.0

    def test_interest_rate_within_product_range(self):
        body = client.post("/api/predict", json=BASE_PAYLOAD).json()
        assert 12.0 <= body["interest_rate"] <= 16.0

    def test_interest_rate_is_half_percent_increment(self):
        body = client.post("/api/predict", json=BASE_PAYLOAD).json()
        assert (body["interest_rate"] * 2) % 1 == 0

    def test_collateral_is_non_negative(self):
        body = client.post("/api/predict", json=BASE_PAYLOAD).json()
        assert body["collateral_required"] >= 0.0

    def test_monthly_payment_is_positive(self):
        body = client.post("/api/predict", json=BASE_PAYLOAD).json()
        assert body["monthly_payment"] > 0

    def test_total_repayment_equals_payment_times_term(self):
        body = client.post("/api/predict", json=BASE_PAYLOAD).json()
        expected = round(body["monthly_payment"] * BASE_PAYLOAD["term"], 2)
        assert abs(body["total_repayment"] - expected) < 0.50

    def test_total_interest_equals_repayment_minus_principal(self):
        body = client.post("/api/predict", json=BASE_PAYLOAD).json()
        expected = body["total_repayment"] - BASE_PAYLOAD["loan_amnt"]
        assert abs(body["total_interest"] - expected) < 0.10

    def test_repayment_schedule_has_correct_number_of_months(self):
        body = client.post("/api/predict", json=BASE_PAYLOAD).json()
        assert len(body["repayment_schedule"]) == BASE_PAYLOAD["term"]

    def test_repayment_schedule_last_balance_near_zero(self):
        body = client.post("/api/predict", json=BASE_PAYLOAD).json()
        last_balance = body["repayment_schedule"][-1]["balance"]
        assert last_balance < 1.0, f"Final balance should be ~0 but got {last_balance}"

    def test_repayment_schedule_month_numbers_sequential(self):
        body = client.post("/api/predict", json=BASE_PAYLOAD).json()
        months = [row["month"] for row in body["repayment_schedule"]]
        assert months == list(range(1, BASE_PAYLOAD["term"] + 1))


# ─────────────────────────────────────────────────────────────────────────────
# Loan amount validation — rejection edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestLoanAmountValidation:

    def test_loan_below_200_rejected(self):
        payload = {**BASE_PAYLOAD, "loan_amnt": 100}
        body = client.post("/api/predict", json=payload).json()
        assert body["status"] == "rejected"
        assert body["rejection_reason"] is not None

    def test_loan_above_5000_rejected(self):
        payload = {**BASE_PAYLOAD, "loan_amnt": 6000}
        body = client.post("/api/predict", json=payload).json()
        assert body["status"] == "rejected"

    def test_loan_exactly_200_approved(self):
        payload = {**BASE_PAYLOAD, "loan_amnt": 200}
        body = client.post("/api/predict", json=payload).json()
        assert body["status"] == "approved"

    def test_loan_exactly_5000_approved(self):
        payload = {**BASE_PAYLOAD, "loan_amnt": 5000}
        body = client.post("/api/predict", json=payload).json()
        assert body["status"] == "approved"

    def test_loan_199_rejected(self):
        payload = {**BASE_PAYLOAD, "loan_amnt": 199}
        body = client.post("/api/predict", json=payload).json()
        assert body["status"] == "rejected"

    def test_loan_5001_rejected(self):
        payload = {**BASE_PAYLOAD, "loan_amnt": 5001}
        body = client.post("/api/predict", json=payload).json()
        assert body["status"] == "rejected"

    def test_rejected_response_has_null_financials(self):
        payload = {**BASE_PAYLOAD, "loan_amnt": 50}
        body = client.post("/api/predict", json=payload).json()
        assert body["pd_score"] is None
        assert body["interest_rate"] is None
        assert body["collateral_required"] is None
        assert body["monthly_payment"] is None


# ─────────────────────────────────────────────────────────────────────────────
# High risk rules
# ─────────────────────────────────────────────────────────────────────────────

class TestHighRiskRules:

    def test_credit_score_below_500_triggers_full_collateral(self):
        payload = {**BASE_PAYLOAD, "credit_score": 450}
        body = client.post("/api/predict", json=payload).json()
        assert body["status"] == "approved"
        assert body["collateral_required"] == BASE_PAYLOAD["loan_amnt"]

    def test_credit_score_499_full_collateral(self):
        payload = {**BASE_PAYLOAD, "credit_score": 499}
        body = client.post("/api/predict", json=payload).json()
        assert body["collateral_required"] == BASE_PAYLOAD["loan_amnt"]


# ─────────────────────────────────────────────────────────────────────────────
# Zero collateral rule
# ─────────────────────────────────────────────────────────────────────────────

class TestZeroCollateralRule:

    def test_excellent_profile_gets_zero_collateral(self):
        # cs > 700, DTI < 40 (100/mo on 120k income = 1%), history >= 2 years
        payload = {
            **BASE_PAYLOAD,
            "credit_score":       750,
            "annual_inc":         120000,
            "monthly_debt":       100,
            "mo_sin_old_il_acct": 36,   # 3 years
        }
        body = client.post("/api/predict", json=payload).json()
        assert body["status"] == "approved"
        assert body["collateral_required"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Guarantor effect
# ─────────────────────────────────────────────────────────────────────────────

class TestGuarantorEffect:

    def _collateral_for(self, relationship):
        payload = {**BASE_PAYLOAD, "guarantor_relationship": relationship}
        return client.post("/api/predict", json=payload).json()["collateral_required"]

    def test_spouse_lower_collateral_than_no_guarantor(self):
        assert self._collateral_for("spouse") < self._collateral_for("none")

    def test_guarantor_buffer_order(self):
        spouse = self._collateral_for("spouse")
        family = self._collateral_for("family")
        friend = self._collateral_for("friend")
        none   = self._collateral_for("none")
        assert spouse < family < friend < none


# ─────────────────────────────────────────────────────────────────────────────
# Different loan terms
# ─────────────────────────────────────────────────────────────────────────────

class TestLoanTerms:

    def test_all_valid_terms_return_approved(self):
        for term in [12, 18, 24, 36, 48]:
            payload = {**BASE_PAYLOAD, "term": term}
            body = client.post("/api/predict", json=payload).json()
            assert body["status"] == "approved", f"Term {term} months failed"
            assert len(body["repayment_schedule"]) == term

    def test_longer_term_gives_lower_monthly_payment(self):
        payments = {}
        for term in [12, 24, 48]:
            payload = {**BASE_PAYLOAD, "term": term}
            payments[term] = client.post("/api/predict", json=payload).json()["monthly_payment"]
        assert payments[12] > payments[24] > payments[48]
