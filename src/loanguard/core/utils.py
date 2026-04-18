"""utils.py - Shared utilities: data storage, ID generation, fairness analysis"""

import pandas as pd
import numpy as np
import os
import uuid
from datetime import datetime
from typing import Dict

STORAGE_PATH = "data/applications_log.csv"

COLUMNS = [
    "Applicant_ID", "Timestamp", "Age", "Gender", "Employment_Type",
    "Annual_Income", "Credit_Score", "Existing_Loans",
    "Loan_Amount", "Loan_Purpose",
    "Prediction", "Approval_Probability", "Risk_Level", "Fraud_Flags"
]


def generate_applicant_id() -> str:
    return "APP-" + str(uuid.uuid4())[:8].upper()


def save_application(applicant: dict, prediction_result: dict, risk_result: dict, fraud_result: dict):
    record = {
        "Applicant_ID": generate_applicant_id(),
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Age": applicant["Age"],
        "Gender": applicant.get("Gender", "Not Specified"),
        "Employment_Type": applicant["Employment_Type"],
        "Annual_Income": applicant["Annual_Income"],
        "Credit_Score": applicant["Credit_Score"],
        "Existing_Loans": applicant["Existing_Loans"],
        "Loan_Amount": applicant["Loan_Amount"],
        "Loan_Purpose": applicant["Loan_Purpose"],
        "Prediction": prediction_result["label"],
        "Approval_Probability": round(prediction_result["approval_probability"], 4),
        "Risk_Level": risk_result["level"],
        "Fraud_Flags": fraud_result["flag_count"],
    }

    if os.path.exists(STORAGE_PATH):
        df = pd.read_csv(STORAGE_PATH)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])

    df.to_csv(STORAGE_PATH, index=False)
    return record["Applicant_ID"]


def load_applications() -> pd.DataFrame:
    if os.path.exists(STORAGE_PATH):
        return pd.read_csv(STORAGE_PATH)
    return pd.DataFrame(columns=COLUMNS)


def load_training_data() -> pd.DataFrame:
    return pd.read_csv("data/loan_dataset.csv")


def compute_fairness_metrics(df: pd.DataFrame) -> Dict:
    results = {}

    if "Gender" in df.columns and "Prediction" in df.columns:
        gender_rates = (
            df.groupby("Gender")["Prediction"]
            .apply(lambda x: (x == "Approved").mean())
            .reset_index()
        )
        gender_rates.columns = ["Group", "Approval_Rate"]
        gender_rates["Attribute"] = "Gender"
        results["gender"] = gender_rates

    if "Employment_Type" in df.columns:
        emp_rates = (
            df.groupby("Employment_Type")["Prediction"]
            .apply(lambda x: (x == "Approved").mean())
            .reset_index()
        )
        emp_rates.columns = ["Group", "Approval_Rate"]
        emp_rates["Attribute"] = "Employment_Type"
        results["employment"] = emp_rates

    return results


def format_inr(amount: float) -> str:
    return f"₹{amount:,.0f}"
