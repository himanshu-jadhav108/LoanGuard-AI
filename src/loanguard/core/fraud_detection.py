"""fraud_detection.py - Fraud Score System (0-1) with Multi-Level Risk Classification"""

from typing import Dict, List
import numpy as np


def detect_fraud(applicant: dict) -> Dict:
    """
    Compute fraud score (0-1) and classify into risk levels: Low, Medium, High, Critical
    """
    credit_score = applicant["Credit_Score"]
    income = applicant["Annual_Income"]
    loan_amount = applicant["Loan_Amount"]
    existing_loans = applicant["Existing_Loans"]
    employment = applicant["Employment_Type"]
    age = applicant["Age"]

    debt_ratio = (existing_loans * 50000 + loan_amount) / max(income, 1)
    loan_to_income = loan_amount / max(income, 1)

    flags: List[str] = []
    risk_scores = []  # Individual risk contributions

    # RULE 1: Income Anomaly Score (0.0-0.25)
    income_anomaly = 0.0
    if income > 150000 and credit_score < 550:
        flags.append("High reported income (₹{:,}) paired with very low credit score ({}) — possible income inflation".format(income, credit_score))
        income_anomaly = 0.20
    elif employment == "Salaried" and income < 10000:
        flags.append("Salaried employment declared but income ₹{:,}/yr is implausibly low — possible misreporting".format(income))
        income_anomaly = 0.15
    risk_scores.append(income_anomaly)

    # RULE 2: Loan Overload Score (0.0-0.30)
    loan_overload = 0.0
    if existing_loans >= 3 and loan_amount > 500000:
        flags.append("{} existing active loans + ₹{:,} new request — potential debt trap / stacking pattern".format(existing_loans, loan_amount))
        loan_overload = 0.25
    elif income > 200000 and existing_loans >= 4:
        flags.append("Despite high income, {} concurrent loans suggest loan stacking behavior".format(existing_loans))
        loan_overload = 0.18
    elif employment == "Unemployed" and loan_amount > 100000:
        flags.append("Unemployed applicant requesting large loan of ₹{:,} — no repayment source identified".format(loan_amount))
        loan_overload = 0.30
    risk_scores.append(loan_overload)

    # RULE 3: Credit Inconsistency Score (0.0-0.25)
    credit_inconsistency = 0.0
    if debt_ratio > 12:
        flags.append("Debt-to-income ratio of {:.1f}x — grossly exceeds safe lending threshold of 6x".format(debt_ratio))
        credit_inconsistency = 0.22
    elif loan_to_income > 20:
        flags.append("Loan amount is {:.0f}x annual income — extreme over-leverage request".format(loan_to_income))
        credit_inconsistency = 0.25
    elif debt_ratio > 8:
        credit_inconsistency = 0.15
    risk_scores.append(credit_inconsistency)

    # RULE 4: Age-Loan Mismatch Score (0.0-0.20)
    age_mismatch = 0.0
    if age < 23 and loan_amount > 500000:
        flags.append("Applicant aged {} requesting ₹{:,} — disproportionate to typical early-career profile".format(age, loan_amount))
        age_mismatch = 0.18
    risk_scores.append(age_mismatch)

    # Compute overall fraud score as weighted average
    fraud_score = min(1.0, np.mean(risk_scores) if risk_scores else 0.0) if risk_scores else 0.0
    
    # Risk level classification based on fraud score
    if fraud_score >= 0.75:
        fraud_level = "🚨 CRITICAL"
        fraud_level_short = "CRITICAL"
        color = "#dc2626"  # Red
    elif fraud_score >= 0.50:
        fraud_level = "🔴 HIGH"
        fraud_level_short = "HIGH"
        color = "#ef4444"  # Bright Red
    elif fraud_score >= 0.25:
        fraud_level = "🟡 MEDIUM"
        fraud_level_short = "MEDIUM"
        color = "#f59e0b"  # Orange
    else:
        fraud_level = "🟢 LOW"
        fraud_level_short = "LOW"
        color = "#22c55e"  # Green
    
    is_fraud_risk = fraud_score >= 0.25

    return {
        "fraud_score": round(fraud_score, 3),
        "is_fraud_risk": is_fraud_risk,
        "fraud_level": fraud_level,
        "fraud_level_short": fraud_level_short,
        "color": color,
        "flags": flags,
        "flag_count": len(flags),
        "income_anomaly": round(income_anomaly, 3),
        "loan_overload": round(loan_overload, 3),
        "credit_inconsistency": round(credit_inconsistency, 3),
        "age_mismatch": round(age_mismatch, 3),
        "recommendation": (
            "🚨 ESCALATE: Immediate fraud investigation required. Reject application pending inquiry."
            if fraud_score >= 0.75 else
            "⚠️ REVIEW: Flag for senior loan officer manual review. Possible fraud pattern."
            if fraud_score >= 0.50 else
            "⚡ VERIFY: Enhanced verification recommended before approval."
            if fraud_score >= 0.25 else
            "✅ CLEAR: No anomalies detected. Standard processing applicable."
        )
    }
