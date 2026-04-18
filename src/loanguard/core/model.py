"""model.py - ML model training, saving, and prediction logic"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix
)
from loanguard.core.preprocessing import get_feature_matrix, build_scaler, transform, preprocess_single, FEATURE_COLS

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_models(data_path: str = "data/loan_dataset.csv"):
    df = pd.read_csv(data_path)
    y = df["Eligibility"].values
    scaler = build_scaler(df)
    X = transform(df, scaler)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_prob = lr.predict_proba(X_test)[:, 1]

    lr_metrics = {
        "accuracy": accuracy_score(y_test, lr_pred),
        "auc": roc_auc_score(y_test, lr_prob),
        "report": classification_report(y_test, lr_pred),
        "cm": confusion_matrix(y_test, lr_pred),
        "cv_scores": cross_val_score(lr, X, y, cv=5, scoring="accuracy"),
    }

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced")
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_prob = dt.predict_proba(X_test)[:, 1]

    dt_metrics = {
        "accuracy": accuracy_score(y_test, dt_pred),
        "auc": roc_auc_score(y_test, dt_prob),
        "report": classification_report(y_test, dt_pred),
        "cm": confusion_matrix(y_test, dt_pred),
        "cv_scores": cross_val_score(dt, X, y, cv=5, scoring="accuracy"),
    }

    # Save artifacts
    joblib.dump(lr, f"{MODEL_DIR}/logistic_regression.pkl")
    joblib.dump(dt, f"{MODEL_DIR}/decision_tree.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(FEATURE_COLS, f"{MODEL_DIR}/feature_cols.pkl")

    print("=== Logistic Regression ===")
    print(f"Accuracy: {lr_metrics['accuracy']:.4f} | AUC: {lr_metrics['auc']:.4f}")
    print(f"CV Mean: {lr_metrics['cv_scores'].mean():.4f}")
    print(lr_metrics["report"])
    print("\n=== Decision Tree ===")
    print(f"Accuracy: {dt_metrics['accuracy']:.4f} | AUC: {dt_metrics['auc']:.4f}")
    print(f"CV Mean: {dt_metrics['cv_scores'].mean():.4f}")

    return lr, dt, scaler, lr_metrics, dt_metrics


def load_model(model_name: str = "logistic_regression"):
    model = joblib.load(f"{MODEL_DIR}/{model_name}.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    return model, scaler


def predict(applicant: dict, model_name: str = "logistic_regression"):
    model, scaler = load_model(model_name)
    X = preprocess_single(applicant, scaler)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    return {
        "prediction": int(pred),
        "label": "Approved" if pred == 1 else "Rejected",
        "approval_probability": float(prob[1]),
        "rejection_probability": float(prob[0]),
    }


if __name__ == "__main__":
    train_models()
