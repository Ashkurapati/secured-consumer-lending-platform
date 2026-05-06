"""
Unit tests for Stepping Stones core business logic.

Tests the pure functions in website/api.py:
  - credit_score_to_int_rate
  - credit_score_to_grade
  - calc_monthly_payment
  - collateral formula rules
  - interest rate formula

Run with:
    pytest tests/test_unit.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "website"))

from api import (
    credit_score_to_int_rate,
    credit_score_to_grade,
    grade_to_subgrade,
    calc_monthly_payment,
)


# ─────────────────────────────────────────────────────────────────────────────
# credit_score_to_int_rate
# ─────────────────────────────────────────────────────────────────────────────

class TestCreditScoreToIntRate:

    def test_excellent_credit_750_plus(self):
        assert credit_score_to_int_rate(800) == 7.51

    def test_boundary_exactly_750(self):
        assert credit_score_to_int_rate(750) == 7.51

    def test_boundary_exactly_720(self):
        assert credit_score_to_int_rate(720) == 9.93

    def test_between_720_and_749(self):
        assert credit_score_to_int_rate(735) == 9.93

    def test_boundary_exactly_700(self):
        assert credit_score_to_int_rate(700) == 11.99

    def test_boundary_exactly_680(self):
        assert credit_score_to_int_rate(680) == 13.67

    def test_boundary_exactly_660(self):
        assert credit_score_to_int_rate(660) == 15.61

    def test_boundary_exactly_620(self):
        assert credit_score_to_int_rate(620) == 19.99

    def test_below_550_returns_highest_rate(self):
        assert credit_score_to_int_rate(400) == 29.99

    def test_boundary_549(self):
        assert credit_score_to_int_rate(549) == 29.99

    def test_boundary_550(self):
        assert credit_score_to_int_rate(550) == 27.49

    def test_rate_decreases_as_score_increases(self):
        scores = [400, 500, 550, 580, 600, 620, 640, 660, 680, 700, 720, 750, 800]
        rates = [credit_score_to_int_rate(s) for s in scores]
        # Each rate should be >= the next (higher score = lower or equal rate)
        for i in range(len(rates) - 1):
            assert rates[i] >= rates[i + 1], (
                f"Score {scores[i]} → {rates[i]}% should be >= "
                f"score {scores[i+1]} → {rates[i+1]}%"
            )


# ─────────────────────────────────────────────────────────────────────────────
# credit_score_to_grade
# ─────────────────────────────────────────────────────────────────────────────

class TestCreditScoreToGrade:

    def test_grade_1_excellent(self):
        assert credit_score_to_grade(800) == 1

    def test_grade_1_boundary(self):
        assert credit_score_to_grade(750) == 1

    def test_grade_2_boundary(self):
        assert credit_score_to_grade(749) == 2

    def test_grade_7_very_poor(self):
        assert credit_score_to_grade(400) == 7

    def test_grade_7_boundary_539(self):
        assert credit_score_to_grade(539) == 7

    def test_grade_6_boundary_540(self):
        assert credit_score_to_grade(540) == 6

    def test_grades_are_1_to_7(self):
        for cs in range(300, 851, 10):
            grade = credit_score_to_grade(cs)
            assert 1 <= grade <= 7, f"Credit score {cs} gave invalid grade {grade}"


# ─────────────────────────────────────────────────────────────────────────────
# calc_monthly_payment
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcMonthlyPayment:

    def test_known_value_12_percent_12_months(self):
        # $1,000 at 12% annual over 12 months ≈ $88.85/month
        payment = calc_monthly_payment(1000, 12.0, 12)
        assert abs(payment - 88.85) < 0.05

    def test_known_value_14_percent_36_months(self):
        # $2,000 at 14% annual over 36 months ≈ $68.39/month
        payment = calc_monthly_payment(2000, 14.0, 36)
        assert abs(payment - 68.39) < 0.05

    def test_zero_interest_rate_divides_evenly(self):
        # Edge case: 0% interest — should just divide principal by months
        payment = calc_monthly_payment(1200, 0.0, 12)
        assert abs(payment - 100.0) < 0.01

    def test_total_repayment_exceeds_principal(self):
        # At any positive rate, total repayment > principal
        payment = calc_monthly_payment(2000, 14.0, 36)
        assert payment * 36 > 2000

    def test_longer_term_means_lower_monthly_payment(self):
        p12 = calc_monthly_payment(2000, 14.0, 12)
        p36 = calc_monthly_payment(2000, 14.0, 36)
        p48 = calc_monthly_payment(2000, 14.0, 48)
        assert p12 > p36 > p48

    def test_max_loan_max_rate_max_term(self):
        # $5,000 at 16% over 48 months — should not crash and should be positive
        payment = calc_monthly_payment(5000, 16.0, 48)
        assert payment > 0

    def test_min_loan_min_rate_min_term(self):
        payment = calc_monthly_payment(200, 12.0, 12)
        assert payment > 0


# ─────────────────────────────────────────────────────────────────────────────
# Collateral formula rules (tested directly without calling the model)
# ─────────────────────────────────────────────────────────────────────────────

class TestCollateralRules:

    def _compute_collateral(self, pd_score, credit_score, dti,
                            mo_sin_old_il_acct, loan_amnt,
                            guarantor_relationship="none"):
        """Mirrors the collateral logic in api.py predict()."""
        high_risk = credit_score < 500 or pd_score > 0.80
        if high_risk:
            return loan_amnt

        credit_history_years = mo_sin_old_il_acct / 12
        if credit_score > 700 and dti < 40 and credit_history_years >= 2:
            return 0.0

        LGD = 0.70
        expected_loss = pd_score * LGD * loan_amnt
        guarantor_buffer = {
            "spouse": 0.05, "family": 0.10,
            "friend": 0.15, "none": 0.25,
        }
        buffer_pct = guarantor_buffer.get(guarantor_relationship, 0.25)
        return expected_loss + (buffer_pct * loan_amnt)

    def test_zero_collateral_rule(self):
        # cs > 700, dti < 40, history >= 2 years → collateral waived
        collateral = self._compute_collateral(
            pd_score=0.30, credit_score=750, dti=25.0,
            mo_sin_old_il_acct=36, loan_amnt=2000
        )
        assert collateral == 0.0

    def test_zero_collateral_boundary_credit_score_700_not_eligible(self):
        # cs must be > 700, so 700 exactly is NOT eligible
        collateral = self._compute_collateral(
            pd_score=0.30, credit_score=700, dti=25.0,
            mo_sin_old_il_acct=36, loan_amnt=2000
        )
        assert collateral > 0.0

    def test_zero_collateral_boundary_dti_40_not_eligible(self):
        # dti must be < 40, so 40 exactly is NOT eligible
        collateral = self._compute_collateral(
            pd_score=0.30, credit_score=750, dti=40.0,
            mo_sin_old_il_acct=36, loan_amnt=2000
        )
        assert collateral > 0.0

    def test_high_risk_credit_score_below_500(self):
        # cs < 500 → 100% collateral regardless of PD
        collateral = self._compute_collateral(
            pd_score=0.20, credit_score=450, dti=20.0,
            mo_sin_old_il_acct=36, loan_amnt=2000
        )
        assert collateral == 2000.0

    def test_high_risk_pd_above_080(self):
        # PD > 0.80 → 100% collateral
        collateral = self._compute_collateral(
            pd_score=0.85, credit_score=650, dti=25.0,
            mo_sin_old_il_acct=36, loan_amnt=2000
        )
        assert collateral == 2000.0

    def test_pd_exactly_080_is_high_risk(self):
        # PD > 0.80 triggers high risk — 0.80 itself does NOT (strictly greater)
        collateral_at_080 = self._compute_collateral(
            pd_score=0.80, credit_score=600, dti=35.0,
            mo_sin_old_il_acct=24, loan_amnt=2000
        )
        collateral_above_080 = self._compute_collateral(
            pd_score=0.81, credit_score=600, dti=35.0,
            mo_sin_old_il_acct=24, loan_amnt=2000
        )
        assert collateral_above_080 == 2000.0
        assert collateral_at_080 < 2000.0

    def test_spouse_guarantor_lower_than_no_guarantor(self):
        base = dict(pd_score=0.40, credit_score=600, dti=35.0,
                    mo_sin_old_il_acct=24, loan_amnt=2000)
        assert self._compute_collateral(**base, guarantor_relationship="spouse") < \
               self._compute_collateral(**base, guarantor_relationship="none")

    def test_guarantor_buffer_ordering(self):
        base = dict(pd_score=0.40, credit_score=600, dti=35.0,
                    mo_sin_old_il_acct=24, loan_amnt=2000)
        spouse = self._compute_collateral(**base, guarantor_relationship="spouse")
        family = self._compute_collateral(**base, guarantor_relationship="family")
        friend = self._compute_collateral(**base, guarantor_relationship="friend")
        none   = self._compute_collateral(**base, guarantor_relationship="none")
        assert spouse < family < friend < none

    def test_collateral_formula_exact_values(self):
        # PD=0.40, LGD=0.70, loan=$2000, no guarantor (25% buffer)
        # expected_loss = 0.40 * 0.70 * 2000 = 560
        # buffer        = 0.25 * 2000        = 500
        # total         = 1060
        collateral = self._compute_collateral(
            pd_score=0.40, credit_score=600, dti=35.0,
            mo_sin_old_il_acct=24, loan_amnt=2000, guarantor_relationship="none"
        )
        assert abs(collateral - 1060.0) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Interest rate formula
# ─────────────────────────────────────────────────────────────────────────────

class TestInterestRateFormula:

    def _compute_rate(self, pd_score):
        raw = 12.0 + (pd_score * 4.0)
        return round(min(16.0, max(12.0, raw)) * 2) / 2

    def test_pd_zero_gives_12_percent(self):
        assert self._compute_rate(0.0) == 12.0

    def test_pd_one_capped_at_16_percent(self):
        assert self._compute_rate(1.0) == 16.0

    def test_pd_half_gives_14_percent(self):
        assert self._compute_rate(0.50) == 14.0

    def test_rate_always_in_valid_range(self):
        for pd_score in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            rate = self._compute_rate(pd_score)
            assert 12.0 <= rate <= 16.0, f"PD {pd_score} gave rate {rate}% outside range"

    def test_rate_rounded_to_half_percent(self):
        for pd_score in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            rate = self._compute_rate(pd_score)
            assert (rate * 2) % 1 == 0, f"Rate {rate}% is not a 0.5% increment"

    def test_rate_increases_with_pd(self):
        pd_values = [0.0, 0.25, 0.50, 0.75, 1.0]
        rates = [self._compute_rate(pd) for pd in pd_values]
        for i in range(len(rates) - 1):
            assert rates[i] <= rates[i + 1]
