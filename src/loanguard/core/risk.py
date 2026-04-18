"""risk.py - Multi-factor risk classification system"""

from typing import Dict


def classify_risk(applicant: dict, approval_probability: float) -> Dict:
    credit_score = applicant["Credit_Score"]
    income = applicant["Annual_Income"]
    loan_amount = applicant["Loan_Amount"]
    existing_loans = applicant["Existing_Loans"]
    employment = applicant["Employment_Type"]

    debt_ratio = (existing_loans * 50000 + loan_amount) / max(income, 1)

    # Score-based risk engine (0 = safest, higher = riskier)
    risk_score = 0

    # Probability contribution (most weighted)
    if approval_probability < 0.35:
        risk_score += 40
    elif approval_probability < 0.60:
        risk_score += 20
    elif approval_probability < 0.80:
        risk_score += 8
    else:
        risk_score += 0

    # Credit Score
    if credit_score < 580:
        risk_score += 30
    elif credit_score < 670:
        risk_score += 15
    elif credit_score < 740:
        risk_score += 5
    else:
        risk_score += 0

    # Debt Ratio
    if debt_ratio > 8:
        risk_score += 20
    elif debt_ratio > 5:
        risk_score += 10
    elif debt_ratio > 3:
        risk_score += 4
    else:
        risk_score += 0

    # Employment
    emp_risk = {
        "Salaried": 0,
        "Business Owner": 3,
        "Self-Employed": 5,
        "Freelancer": 8,
        "Unemployed": 20,
    }
    risk_score += emp_risk.get(employment, 5)

    # Existing Loans
    risk_score += existing_loans * 3

    # Final classification
    if risk_score <= 15:
        level = "Low Risk"
        color = "#22c55e"
        icon = "🟢"
        description = "Applicant presents a low-risk profile. Minimal monitoring required post-disbursement."
    elif risk_score <= 35:
        level = "Medium Risk"
        color = "#f59e0b"
        icon = "🟡"
        description = "Moderate risk indicators present. Recommend standard monitoring and possible collateral documentation."
    else:
        level = "High Risk"
        color = "#ef4444"
        icon = "🔴"
        description = "Multiple high-risk factors identified. Enhanced due diligence and stricter terms recommended."

    return {
        "level": level,
        "color": color,
        "icon": icon,
        "score": risk_score,
        "description": description,
        "debt_ratio": round(debt_ratio, 3),
        "components": {
            "probability_risk": min(40, max(0, 40 - int(approval_probability * 40))),
            "credit_risk": max(0, 30 - int((credit_score - 300) / 18.5)),
            "debt_risk": min(20, max(0, int(debt_ratio * 2.5))),
            "employment_risk": emp_risk.get(employment, 5),
            "loan_count_risk": existing_loans * 3,
        }
    }
