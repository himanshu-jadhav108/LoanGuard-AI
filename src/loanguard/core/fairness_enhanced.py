"""fairness_enhanced.py - Bias Detection, Fair Lending Analysis, and Regulatory Compliance"""

import pandas as pd
import numpy as np
from typing import Dict, List


def compute_fairness_metrics_enhanced(df: pd.DataFrame, threshold: float = 0.15) -> Dict:
    """
    Advanced fairness analysis with bias detection using 4/5 rule and disparate impact analysis.
    
    Args:
        df: Application log dataframe
        threshold: Alert threshold for approval rate gap (15% = 0.15)
    
    Returns:
        Dictionary with fairness metrics, violations, and recommendations
    """
    results = {
        "has_bias_violations": False,
        "violations": [],
        "metrics_by_group": {},
        "warnings": [],
        "recommendations": [],
        "compliance_status": "[COMPLIANT]",
        "compliance_summary": "No data available for fairness evaluation."
    }
    
    if len(df) == 0:
        return results

    # ─────────────────────────────────────────
    # ANALYSIS 1: Gender-Based Fairness
    # ─────────────────────────────────────────
    if "Gender" in df.columns and "Prediction" in df.columns:
        gender_stats = []
        
        for gender in df["Gender"].unique():
            if pd.isna(gender):
                continue
            
            gender_subset = df[df["Gender"] == gender]
            approval_rate = (gender_subset["Prediction"] == "Approved").mean()
            count = len(gender_subset)
            
            gender_stats.append({
                "group": gender,
                "approval_rate": approval_rate,
                "approved_count": (gender_subset["Prediction"] == "Approved").sum(),
                "total_count": count,
                "avg_credit": gender_subset["Credit_Score"].mean() if "Credit_Score" in df.columns else None,
                "avg_income": gender_subset["Annual_Income"].mean() if "Annual_Income" in df.columns else None,
            })
        
        if len(gender_stats) > 1:
            # 4/5 Rule: protected group approval rate should be ≥ 80% of non-protected group
            gender_stats_sorted = sorted(gender_stats, key=lambda x: x["approval_rate"], reverse=True)
            highest = gender_stats_sorted[0]
            lowest = gender_stats_sorted[-1]
            
            disparate_impact_ratio = lowest["approval_rate"] / max(highest["approval_rate"], 0.01)
            gap = highest["approval_rate"] - lowest["approval_rate"]
            
            results["metrics_by_group"]["Gender"] = {
                "groups": gender_stats,
                "disparate_impact_ratio": disparate_impact_ratio,
                "gap": gap,
                "violation": disparate_impact_ratio < 0.80 or gap > threshold
            }
            
            if disparate_impact_ratio < 0.80:
                results["has_bias_violations"] = True
                results["violations"].append({
                    "type": "DISPARATE IMPACT - Gender",
                    "severity": "CRITICAL",
                    "icon": "[BIAS_GENDER]",
                    "detail": "Gender disparate impact: {:.1f}%. {} approval rate ({:.1f}%) is <80% of {} rate ({:.1f}%). FCRA violation risk.".format(disparate_impact_ratio*100, lowest['group'], lowest['approval_rate']*100, highest['group'], highest['approval_rate']*100),
                    "affected_group": lowest["group"],
                    "ratio": disparate_impact_ratio
                })
                results["recommendations"].append(
                    "Audit underwriting criteria for gender bias. {} applicants show {:.1f}% lower approval rate.".format(lowest['group'], gap*100)
                )

    # ─────────────────────────────────────────
    # ANALYSIS 2: Employment-Based Fairness
    # ─────────────────────────────────────────
    if "Employment_Type" in df.columns:
        emp_stats = []
        
        for emp_type in df["Employment_Type"].unique():
            if pd.isna(emp_type):
                continue
            
            emp_subset = df[df["Employment_Type"] == emp_type]
            approval_rate = (emp_subset["Prediction"] == "Approved").mean()
            count = len(emp_subset)
            
            emp_stats.append({
                "group": emp_type,
                "approval_rate": approval_rate,
                "approved_count": (emp_subset["Prediction"] == "Approved").sum(),
                "total_count": count,
                "avg_risk": emp_subset["Risk_Level"].value_counts().to_dict() if "Risk_Level" in df.columns else {},
            })
        
        if len(emp_stats) > 1:
            emp_stats_sorted = sorted(emp_stats, key=lambda x: x["approval_rate"], reverse=True)
            highest = emp_stats_sorted[0]
            lowest = emp_stats_sorted[-1]
            
            gap = highest["approval_rate"] - lowest["approval_rate"]
            
            results["metrics_by_group"]["Employment_Type"] = {
                "groups": emp_stats,
                "gap": gap,
                "violation": gap > threshold
            }
            
            if gap > threshold:
                results["warnings"].append({
                    "type": "Employment-Type Gap Alert",
                    "severity": "HIGH",
                    "icon": "[EMP]",
                    "detail": "Approval rate gap: {:.1f}% between {} ({:.1f}%) and {} ({:.1f}%). Requires policy review.".format(gap*100, highest['group'], highest['approval_rate']*100, lowest['group'], lowest['approval_rate']*100),
                    "affected_group": lowest["group"]
                })

    # ─────────────────────────────────────────
    # ANALYSIS 3: Income Bracket Fairness
    # ─────────────────────────────────────────
    if "Annual_Income" in df.columns:
        income_brackets = [
            {"name": "Very Low (< ₹25K)", "min": 0, "max": 25000},
            {"name": "Low (₹25-50K)", "min": 25000, "max": 50000},
            {"name": "Medium (₹50-100K)", "min": 50000, "max": 100000},
            {"name": "High (₹100-200K)", "min": 100000, "max": 200000},
            {"name": "Very High (> ₹200K)", "min": 200000, "max": float('inf')},
        ]
        
        income_stats = []
        for bracket in income_brackets:
            bracket_subset = df[(df["Annual_Income"] >= bracket["min"]) & (df["Annual_Income"] < bracket["max"])]
            
            if len(bracket_subset) > 0:
                approval_rate = (bracket_subset["Prediction"] == "Approved").mean()
                income_stats.append({
                    "bracket": bracket["name"],
                    "approval_rate": approval_rate,
                    "count": len(bracket_subset),
                    "approved": (bracket_subset["Prediction"] == "Approved").sum(),
                })
        
        if len(income_stats) > 1:
            results["metrics_by_group"]["Income_Bracket"] = {"groups": income_stats}
            
            # Check monotonicity: approval rate should increase with income
            approval_rates = [s["approval_rate"] for s in income_stats]
            if approval_rates != sorted(approval_rates):
                results["warnings"].append({
                    "type": "Income Non-Monotonicity Alert",
                    "severity": "MEDIUM",
                    "icon": "[INCOME]",
                    "detail": "Approval rate does not strictly increase with income. May indicate criteria inconsistency.",
                })

    # ─────────────────────────────────────────
    # ANALYSIS 4: Age-Based Fairness (ADEA)
    # ─────────────────────────────────────────
    if "Age" in df.columns:
        age_groups = [
            {"name": "Young (< 25)", "min": 0, "max": 25},
            {"name": "Prime (25-55)", "min": 25, "max": 55},
            {"name": "Senior (55+)", "min": 55, "max": 150},
        ]
        
        age_stats = []
        for group in age_groups:
            age_subset = df[(df["Age"] >= group["min"]) & (df["Age"] < group["max"])]
            
            if len(age_subset) > 0:
                approval_rate = (age_subset["Prediction"] == "Approved").mean()
                age_stats.append({
                    "group": group["name"],
                    "approval_rate": approval_rate,
                    "count": len(age_subset),
                    "avg_age": age_subset["Age"].mean(),
                })
        
        if len(age_stats) > 1:
            results["metrics_by_group"]["Age"] = {"groups": age_stats}
            
            # ADEA: Protected age 55+ should not face materially adverse impact
            senior_rate = next((s["approval_rate"] for s in age_stats if "55+" in s["group"]), None)
            prime_rate = next((s["approval_rate"] for s in age_stats if "25-55" in s["group"]), None)
            
            if senior_rate is not None and prime_rate is not None:
                age_gap = prime_rate - senior_rate
                if age_gap > 0.15:
                    results["warnings"].append({
                        "type": "ADEA Concern - Senior Applicants",
                        "severity": "HIGH",
                        "icon": "[SENIOR]",
                        "detail": "Senior applicants (55+) {:.1f}% less likely to be approved. May violate Age Discrimination in Employment Act.".format(age_gap*100)
                    })
    # ─────────────────────────────────────────
    compliance_status = "[COMPLIANT]" if not results["has_bias_violations"] else "[VIOLATIONS_DETECTED]"
    results["compliance_status"] = compliance_status
    
    if not results["has_bias_violations"] and results["warnings"]:
        results["compliance_summary"] = "No critical violations, but monitor listed warnings."
    elif results["has_bias_violations"]:
        results["compliance_summary"] = "WARNING: {} potential fair lending violation(s) detected. Immediate legal review recommended.".format(len(results['violations']))
    else:
        results["compliance_summary"] = "All fairness checks passed. Fair lending principles observed."
    
    return results


def get_fairness_dashboard_metrics(df: pd.DataFrame) -> Dict:
    """
    Prepare simplified fairness metrics for dashboard display.
    """
    metrics = {
        "total_applications": len(df),
        "overall_approval_rate": (df["Prediction"] == "Approved").mean() if len(df) > 0 else 0,
    }
    
    if "Gender" in df.columns and len(df) > 0:
        gender_rates = df.groupby("Gender")["Prediction"].apply(lambda x: (x == "Approved").mean())
        metrics["gender_fairness"] = {
            "rates": gender_rates.to_dict(),
            "max_gap": gender_rates.max() - gender_rates.min()
        }
    
    if "Employment_Type" in df.columns and len(df) > 0:
        emp_rates = df.groupby("Employment_Type")["Prediction"].apply(lambda x: (x == "Approved").mean())
        metrics["employment_fairness"] = {
            "rates": emp_rates.to_dict(),
            "max_gap": emp_rates.max() - emp_rates.min()
        }
    
    return metrics
