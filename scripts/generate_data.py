import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

ages = np.random.randint(21, 65, n)
genders = np.random.choice(["Male", "Female", "Other"], n, p=[0.52, 0.45, 0.03])
employment_types = np.random.choice(
    ["Salaried", "Self-Employed", "Business Owner", "Freelancer", "Unemployed"],
    n, p=[0.45, 0.25, 0.15, 0.10, 0.05]
)
incomes = np.where(
    employment_types == "Unemployed",
    np.random.randint(0, 15000, n),
    np.random.randint(25000, 300000, n)
)
credit_scores = np.clip(np.random.normal(680, 80, n).astype(int), 300, 850)
existing_loans = np.random.randint(0, 5, n)
loan_amounts = np.random.randint(10000, 1500000, n)
loan_purposes = np.random.choice(
    ["Home", "Education", "Vehicle", "Business", "Medical", "Personal"],
    n, p=[0.30, 0.15, 0.20, 0.15, 0.10, 0.10]
)

debt_ratio = (existing_loans * 50000 + loan_amounts) / np.maximum(incomes, 1)

# Eligibility logic
prob_approved = (
    0.35 * (credit_scores / 850) +
    0.30 * np.clip(incomes / 200000, 0, 1) +
    0.15 * (1 - np.clip(debt_ratio / 10, 0, 1)) +
    0.10 * (1 - existing_loans / 5) +
    0.10 * (employment_types == "Salaried").astype(float)
)
prob_approved += np.random.normal(0, 0.05, n)
prob_approved = np.clip(prob_approved, 0, 1)
eligibility = (prob_approved > 0.50).astype(int)

df = pd.DataFrame({
    "Applicant_ID": [f"APP{str(i).zfill(5)}" for i in range(1, n+1)],
    "Age": ages,
    "Gender": genders,
    "Employment_Type": employment_types,
    "Annual_Income": incomes,
    "Credit_Score": credit_scores,
    "Existing_Loans": existing_loans,
    "Loan_Amount": loan_amounts,
    "Loan_Purpose": loan_purposes,
    "Eligibility": eligibility,
    "Approval_Probability": prob_approved.round(4),
})

df.to_csv("data/loan_dataset.csv", index=False)
print(f"Dataset created: {len(df)} records")
print(df["Eligibility"].value_counts())
