"""explain.py - Advanced Rule-based Explanation Engine with Feature Contribution Analysis"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Thresholds
CREDIT_EXCELLENT = 750
CREDIT_GOOD = 670
CREDIT_FAIR = 580

INCOME_HIGH = 100000
INCOME_MEDIUM = 50000
INCOME_LOW = 25000

DEBT_RATIO_SAFE = 3.0
DEBT_RATIO_WARNING = 6.0

LOAN_TO_INCOME_SAFE = 5.0
LOAN_TO_INCOME_HIGH = 10.0


def _credit_band(score: int) -> str:
    if score >= CREDIT_EXCELLENT:
        return "excellent"
    elif score >= CREDIT_GOOD:
        return "good"
    elif score >= CREDIT_FAIR:
        return "fair"
    else:
        return "poor"


def _income_band(income: float) -> str:
    if income >= INCOME_HIGH:
        return "high"
    elif income >= INCOME_MEDIUM:
        return "moderate"
    elif income >= INCOME_LOW:
        return "low"
    else:
        return "very low"


def _compute_feature_contribution(applicant: dict) -> List[Tuple[str, float]]:
    """
    Compute normalized feature contribution scores (0-1) indicating how much each feature
    influenced the decision. Features are scored based on deviation from safe thresholds.
    """
    credit_score = applicant["Credit_Score"]
    income = applicant["Annual_Income"]
    loan_amount = applicant["Loan_Amount"]
    existing_loans = applicant["Existing_Loans"]
    employment = applicant["Employment_Type"]
    age = applicant["Age"]

    debt_ratio = (existing_loans * 50000 + loan_amount) / max(income, 1)
    loan_to_income = loan_amount / max(income, 1)

    contributions = {}

    # Credit Score Contribution (0-1, normalized)
    # Poor(<=580): 1.0, Fair(580-670): 0.6, Good(670-750): 0.3, Excellent(>=750): 0.0
    if credit_score <= 580:
        contributions["Credit_Score"] = 1.0
    elif credit_score <= 670:
        contributions["Credit_Score"] = 0.6
    elif credit_score <= 750:
        contributions["Credit_Score"] = 0.3
    else:
        contributions["Credit_Score"] = 0.0

    # Income Contribution (0-1)
    # Very Low (<25K): 1.0, Low (25-50K): 0.6, Medium (50-100K): 0.3, High (>100K): 0.0
    if income < 25000:
        contributions["Annual_Income"] = 1.0
    elif income < 50000:
        contributions["Annual_Income"] = 0.6
    elif income < 100000:
        contributions["Annual_Income"] = 0.3
    else:
        contributions["Annual_Income"] = 0.0

    # Loan Amount Contribution (0-1, based on loan-to-income)
    # > 20x: 1.0, 10-20x: 0.8, 5-10x: 0.4, < 5x: 0.0
    if loan_to_income > 20:
        contributions["Loan_Amount"] = 1.0
    elif loan_to_income > 10:
        contributions["Loan_Amount"] = 0.8
    elif loan_to_income > 5:
        contributions["Loan_Amount"] = 0.4
    else:
        contributions["Loan_Amount"] = 0.0

    # Existing Loans Contribution (0-1)
    # 4+: 1.0, 3: 0.7, 2: 0.4, 1: 0.2, 0: 0.0
    contributions["Existing_Loans"] = min(1.0, existing_loans * 0.25)

    # Debt Ratio Contribution (0-1)
    # > 8x: 1.0, 5-8x: 0.6, 3-5x: 0.3, < 3x: 0.0
    if debt_ratio > 8:
        contributions["Debt_Ratio"] = 1.0
    elif debt_ratio > 5:
        contributions["Debt_Ratio"] = 0.6
    elif debt_ratio > 3:
        contributions["Debt_Ratio"] = 0.3
    else:
        contributions["Debt_Ratio"] = 0.0

    # Employment Type Contribution (0-1)
    emp_contribution = {
        "Salaried": 0.0,
        "Business Owner": 0.2,
        "Self-Employed": 0.5,
        "Freelancer": 0.7,
        "Unemployed": 1.0,
    }
    contributions["Employment_Type"] = emp_contribution.get(employment, 0.5)

    # Age Contribution (0-1)
    # Outside 25-55: higher contribution; within: lower
    if 25 <= age <= 55:
        contributions["Age"] = 0.0
    elif age < 25 or age > 58:
        contributions["Age"] = 0.4
    else:
        contributions["Age"] = 0.2

    # Sort by contribution descending
    sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    return sorted_contrib


def generate_explanation(applicant: dict, prediction: int, probability: float) -> Dict:
    credit_score = applicant["Credit_Score"]
    income = applicant["Annual_Income"]
    loan_amount = applicant["Loan_Amount"]
    existing_loans = applicant["Existing_Loans"]
    employment = applicant["Employment_Type"]
    purpose = applicant["Loan_Purpose"]
    age = applicant["Age"]

    debt_ratio = (existing_loans * 50000 + loan_amount) / max(income, 1)
    loan_to_income = loan_amount / max(income, 1)

    positive_factors: List[str] = []
    negative_factors: List[str] = []

    # Credit Score Analysis
    cb = _credit_band(credit_score)
    if credit_score >= CREDIT_EXCELLENT:
        positive_factors.append(f"Excellent credit score ({credit_score}) — top-tier creditworthiness")
    elif credit_score >= CREDIT_GOOD:
        positive_factors.append(f"Good credit score ({credit_score}) — demonstrates responsible credit usage")
    elif credit_score >= CREDIT_FAIR:
        negative_factors.append(f"Fair credit score ({credit_score}) — below preferred threshold of {CREDIT_GOOD}")
    else:
        negative_factors.append(f"Poor credit score ({credit_score}) — significantly below acceptable range")

    # Income Analysis
    ib = _income_band(income)
    if income >= INCOME_HIGH:
        positive_factors.append(f"High annual income (₹{income:,}) — strong repayment capacity")
    elif income >= INCOME_MEDIUM:
        positive_factors.append(f"Moderate income (₹{income:,}) — adequate for loan servicing")
    elif income >= INCOME_LOW:
        negative_factors.append(f"Low income (₹{income:,}) — may strain repayment capacity")
    else:
        negative_factors.append(f"Very low income (₹{income:,}) — insufficient for requested loan amount")

    # Employment Type
    if employment in ["Salaried", "Business Owner"]:
        positive_factors.append(f"Stable employment ({employment}) — consistent income source")
    elif employment == "Self-Employed":
        negative_factors.append("Self-employed status — income may be irregular or unverifiable")
    elif employment == "Freelancer":
        negative_factors.append("Freelance employment — variable income stream increases repayment risk")
    elif employment == "Unemployed":
        negative_factors.append("Currently unemployed — no active income source identified")

    # Debt Ratio
    if debt_ratio <= DEBT_RATIO_SAFE:
        positive_factors.append(f"Healthy debt-to-income ratio ({debt_ratio:.2f}x) — manageable debt load")
    elif debt_ratio <= DEBT_RATIO_WARNING:
        negative_factors.append(f"Elevated debt ratio ({debt_ratio:.2f}x) — existing obligations may reduce capacity")
    else:
        negative_factors.append(f"High debt burden ({debt_ratio:.2f}x) — total obligations exceed safe threshold")

    # Loan Amount vs Income
    if loan_to_income <= LOAN_TO_INCOME_SAFE:
        positive_factors.append(f"Loan amount (₹{loan_amount:,}) is proportionate to income")
    elif loan_to_income <= LOAN_TO_INCOME_HIGH:
        negative_factors.append(f"Loan amount (₹{loan_amount:,}) is high relative to annual income")
    else:
        negative_factors.append(f"Requested loan amount (₹{loan_amount:,}) far exceeds income capacity")

    # Existing Loans
    if existing_loans == 0:
        positive_factors.append("No existing loan obligations — clean repayment history expected")
    elif existing_loans <= 2:
        positive_factors.append(f"{existing_loans} existing loan(s) — within acceptable range")
    else:
        negative_factors.append(f"{existing_loans} active loans — high concurrent debt obligations")

    # Age
    if 25 <= age <= 55:
        positive_factors.append(f"Age {age} — within optimal earning years")
    elif age < 25:
        negative_factors.append(f"Age {age} — limited credit and employment history expected")
    elif age > 58:
        negative_factors.append(f"Age {age} — approaching retirement may affect long-term repayment")

    # Purpose-based note
    purpose_notes = {
        "Home": "Home loan purpose aligns with priority lending category",
        "Education": "Education loan — considered productive investment",
        "Medical": "Medical emergency — humanitarian consideration applicable",
        "Vehicle": "Vehicle loan — asset-backed purpose provides collateral",
        "Business": "Business loan — higher risk; depends on business viability",
        "Personal": "Personal loan — unsecured; scrutinized more strictly",
    }
    if purpose in ["Home", "Education", "Medical", "Vehicle"]:
        positive_factors.append(purpose_notes.get(purpose, ""))
    else:
        negative_factors.append(purpose_notes.get(purpose, ""))

    # Generate summary sentence
    if prediction == 1:
        if not negative_factors:
            summary = f"Application APPROVED. Strong profile with {len(positive_factors)} favorable indicators and an approval confidence of {probability*100:.1f}%."
        else:
            summary = f"Application APPROVED despite minor risk factors. Dominant strengths — particularly {positive_factors[0].lower()} — justify approval at {probability*100:.1f}% confidence."
    else:
        if not positive_factors:
            summary = f"Application REJECTED. Profile presents {len(negative_factors)} critical risk factors with only {probability*100:.1f}% approval probability."
        else:
            summary = f"Application REJECTED. While the applicant shows some strengths, key disqualifiers — particularly {negative_factors[0].lower()} — outweigh positive indicators."

    # Compute feature contributions
    feature_contributions = _compute_feature_contribution(applicant)

    return {
        "summary": summary,
        "positive_factors": positive_factors,
        "negative_factors": negative_factors,
        "feature_contributions": feature_contributions,  # Ranked list of (feature_name, contribution_score)
        "debt_ratio": round(debt_ratio, 3),
        "loan_to_income_ratio": round(loan_to_income, 3),
        "credit_band": cb,
        "income_band": ib,
    }
