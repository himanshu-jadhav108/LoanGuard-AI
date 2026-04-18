"""preprocessing.py - Feature engineering and data preprocessing pipeline"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

FEATURE_COLS = [
    "Age", "Annual_Income", "Credit_Score",
    "Existing_Loans", "Loan_Amount",
    "Employment_Type_enc", "Loan_Purpose_enc",
    "Debt_Ratio", "Income_Per_Loan", "Credit_Score_Band"
]

EMPLOYMENT_MAP = {
    "Salaried": 4,
    "Business Owner": 3,
    "Self-Employed": 2,
    "Freelancer": 1,
    "Unemployed": 0,
}
PURPOSE_MAP = {
    "Home": 5,
    "Education": 4,
    "Vehicle": 3,
    "Business": 2,
    "Medical": 1,
    "Personal": 0,
}

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Employment_Type_enc"] = df["Employment_Type"].map(EMPLOYMENT_MAP).fillna(1)
    df["Loan_Purpose_enc"] = df["Loan_Purpose"].map(PURPOSE_MAP).fillna(0)
    df["Debt_Ratio"] = (df["Existing_Loans"] * 50000 + df["Loan_Amount"]) / df["Annual_Income"].replace(0, 1)
    df["Income_Per_Loan"] = df["Annual_Income"] / (df["Loan_Amount"] + 1)
    df["Credit_Score_Band"] = pd.cut(
        df["Credit_Score"],
        bins=[0, 580, 670, 740, 800, 851],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)
    return df

def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = engineer_features(df)
    return df[FEATURE_COLS]

def build_scaler(df: pd.DataFrame) -> StandardScaler:
    X = get_feature_matrix(df)
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

def transform(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    X = get_feature_matrix(df)
    return scaler.transform(X)

def preprocess_single(applicant: dict, scaler: StandardScaler) -> np.ndarray:
    df = pd.DataFrame([applicant])
    return transform(df, scaler)
