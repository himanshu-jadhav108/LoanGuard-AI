# 🏦 LoanGuard AI

**Intelligent Real-Time Loan Decision Engine with Explainability, Risk Classification & Fairness Compliance**

> Production-grade fintech platform that combines machine learning with business guardrails to make transparent, auditable, and fair lending decisions at scale.

---

## 🎯 Overview

LoanGuard AI is a comprehensive decision support system designed for financial institutions to evaluate loan applications with confidence. Unlike traditional black-box ML models, this system integrates:

- **Intelligent ML predictions** (92.25% accuracy)
- **Business guardrails** (policy-driven overrides for high-risk cases)
- **Multi-factor fraud detection** (5-point rule engine)
- **Regulatory compliance** (disparate impact analysis, ADEA adherence)
- **Full explainability** (decision reasoning at every stage)

The platform is built for **production deployment**, handling real-time single applications, scenario simulation, and batch processing—all with built-in compliance monitoring.

---

## ✨ Core Features

### 🤖 **Real-Time Loan Eligibility Prediction**
- Dual-model ensemble (Logistic Regression + Decision Tree)
- Sub-second inference on structured applicant data
- Probability-based confidence scoring
- Staged processing pipeline with 6+ intermediate decision points

### 📊 **Intelligent Risk Classification**
Multi-factor risk engine scoring applicants 0–100 across:
- **Approval Probability** (40 points) — Model confidence signal
- **Credit Profile** (30 points) — Credit score bands & history strength
- **Debt Burden** (20 points) — Debt-to-income & total obligation ratio
- **Employment Stability** (10 points) — Employment type reliability score
- **Loan Velocity** (variable) — Count and frequency of existing loans

Risk decisions: **Low Risk** (≤15), **Medium Risk** (≤35), **High Risk** (>35)

### 🚨 **Advanced Fraud Detection**
Rules-based system with 5 independent fraud signals (0–1 score, critical threshold 0.50):

1. **Income Anomalies** — Detects misreported income paired with contradictory credit signals
2. **Loan Overload** — Identifies debt stacking patterns and unrealistic multiple loan requests
3. **Credit Inconsistencies** — Flags impossible debt-to-income or loan-to-income ratios
4. **Employment Behavioral Signals** — Rejects unemployed applicants requesting large loans
5. **Age-Loan Mismatch** — Detects age-inappropriate requested loan durations

### 💡 **Explainable AI (XAI) Engine**
Every decision includes:
- **Feature-level explanations** — Why each factor contributed positively/negatively
- **Decision reasoning** — Human-readable narrative of approval/rejection logic
- **Risk factor breakdown** — Quantified impact of each risk component
- **Fraud signal details** — Specific rule violations and evidence
- **Policy override transparency** — When business rules override ML output

### ⚖️ **Fairness & Bias Compliance**
Comprehensive regulatory monitoring:

- **4/5 Rule (Disparate Impact)** — Detects systematic approval rate gaps by protected class
- **ADEA Compliance** — Age Discrimination in Employment Act checks (55+ applicant parity)
- **Income Bracket Fairness** — Ensures equitable approval across income segments
- **Employment-Type Equity** — Monitors bias in Salaried vs. Self-Employed vs. Unemployed groups
- **Gender Parity** — Gender-based approval rate monitoring with impact scoring
- **Actionable Recommendations** — Suggests process adjustments if violations detected

### 🔄 **What-If Scenario Simulator**
Interactive tool for:
- Adjusting applicant attributes (income, loan amount, employment, credit score)
- Simulating decision changes in real-time
- Visualizing approval probability as parameters change
- Understanding decision boundary sensitivity
- Training relationship managers on edge cases

### 📦 **Batch Processing**
- Upload CSV files with 100s–1000s of applications
- Parallel scoring pipeline with consistent feature engineering
- Bulk export of decisions, explanations, and compliance flags
- Summary statistics on batch approval rates, risk distribution, fraud signals

### 📈 **Interactive Analytics Dashboard**
- **KPI Widgets** — Approval rates, avg risk scores, fraud detection efficiency
- **Trend Charts** — Application volume, approval distribution by risk band
- **Cohort Analysis** — Demographic breakdowns, employment-type comparisons
- **Performance Benchmarks** — Model accuracy, AUC, feature importance rankings
- **System Pipeline View** — Flowchart of decision stages with timing

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LOAN APPLICANT DATA                      │
│        (Demographics, Income, Credit, Employment)           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │   DATA VALIDATION & PREPROCESSING │
         │  (Feature Engineering, Scaling)   │
         └────────────┬────────────────────┘
                      │
              ┌───────┴────────┐
              │                │
              ▼                ▼
      ┌───────────────┐   ┌──────────────────┐
      │  LOG. REG.    │   │ DECISION TREE    │
      │  (Primary)    │   │ (Comparison)     │
      │ 92.25% Acc    │   │ 89.25% Acc       │
      └───────┬───────┘   └────────┬─────────┘
              │                    │
              └───────────┬────────┘
                          ▼
             ┌────────────────────────┐
             │  MULTI-FACTOR RISK     │
             │  CLASSIFICATION        │
             │  (0-100 Score)         │
             └───────────┬────────────┘
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
    ┌──────────────────┐   ┌──────────────────┐
    │  FRAUD DETECTION │   │ BUSINESS POLICY  │
    │  (5-Rule Engine) │   │ GUARDRAILS       │
    │  0-1 Score       │   │ (Override Rules) │
    └────┬─────────────┘   └────────┬─────────┘
         │                          │
         └──────────────┬───────────┘
                        ▼
       ┌──────────────────────────────────┐
       │  EXPLAINABILITY ENGINE           │
       │  (Feature Attribution Analysis)  │
       └─────────────┬────────────────────┘
                     │
   ┌─────────────────┼─────────────────┐
   ▼                 ▼                 ▼
┌──────────┐  ┌─────────────────┐  ┌─────────────┐
│ DECISION │  │ FAIRNESS &      │  │ APPLICATION │
│ DELIVER  │  │ BIAS ANALYSIS   │  │ LOGGING &   │
│ (YES/NO) │  │ (Compliance)    │  │ AUDIT TRAIL │
└──────────┘  └─────────────────┘  └─────────────┘
```

**Flow Stages:**
1. **Data Input** — Structured applicant form or batch CSV
2. **Validation** — Required field checks, data type validation
3. **Preprocessing** — Feature engineering, standardization, encoding
4. **ML Prediction** — Dual-model ensemble predicts approval probability
5. **Risk Analysis** — Multi-factor scoring (0–100 scale)
6. **Fraud Detection** — 5-rule engine detects anomalies
7. **Policy Application** — Business guardrails override high-risk/high-fraud cases
8. **Explanation** — XAI engine generates decision narratives
9. **Fairness Check** — Compliance monitoring & regulatory flags
10. **Output & Storage** — Decision logged, audit trail maintained

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit 1.32+ | Interactive 6-page UI dashboard |
| **Backend Engine** | Python 3.8+ | Core decision logic, orchestration |
| **ML Framework** | Scikit-learn 1.3+ | Logistic Regression, Decision Tree models |
| **Data Processing** | Pandas 2.0+, NumPy 1.24+ | Feature engineering, aggregation, analytics |
| **Visualization** | Plotly 5.18+, Matplotlib 3.7+, Seaborn 0.12+ | Interactive charts, fairness heatmaps |
| **Model Persistence** | Joblib 1.3+ | Model serialization, fast loading |
| **Graph Rendering** | Graphviz 0.20+ | Architecture & pipeline visualization |

---

## 📁 Project Structure

```text
loan_system/
├── 📄 app.py                          # Root Streamlit entrypoint (compatibility launcher)
├── 📄 requirements.txt                # Python dependencies
├── 📄 pyproject.toml                  # Project metadata
│
├── 📁 src/loanguard/                  # Main package (professional layout)
│   ├── 📁 apps/
│   │   └── 📄 app.py                  # 6-page production interface
│   │       ├── Loan Application page (real-time prediction)
│   │       ├── Scenario Simulator (what-if analysis)
│   │       ├── Batch Processing (CSV scoring)
│   │       ├── Analytics Dashboard (KPIs & trends)
│   │       ├── Fairness & Bias (compliance monitoring)
│   │       └── System Pipeline (architecture view)
│   │
│   └── 📁 core/                       # ML & decision engine
│       ├── 📄 model.py                # Logistic Regression + Decision Tree training/inference
│       ├── 📄 preprocessing.py        # Feature engineering, scaling, encoding pipeline
│       ├── 📄 risk.py                 # Multi-factor risk classification (0-100 score)
│       ├── 📄 fraud_detection.py      # 5-rule fraud detection engine (0-1 score)
│       ├── 📄 explain.py              # XAI engine with feature-level explanations
│       ├── 📄 fairness_enhanced.py    # Bias detection, 4/5 rule, ADEA, compliance checks
│       ├── 📄 analytics.py            # Dashboard KPIs, batch summaries, benchmarks
│       ├── 📄 utils.py                # Utility functions (data I/O, formatting)
│       └── 📄 __init__.py             # Package initialization
│
├── 📁 scripts/                        # Data & model utilities
│   ├── 📄 generate_data.py            # Synthetic dataset creation (2000 loan applications)
│   └── 📄 train_model.py              # Model training pipeline
│
├── 📁 data/                           # Dataset storage
│   ├── 📄 loan_dataset.csv            # Training data (2000 applications)
│   └── 📄 applications_log.csv        # Historical applications & predictions
│
└── 📁 models/                         # Model artifacts
    ├── 📄 scaler.pkl                  # StandardScaler for feature normalization
    ├── 📄 logistic_regression.pkl     # Primary model (92.25% accuracy)
    └── 📄 decision_tree.pkl           # Comparison model (89.25% accuracy)
```

### **Key File Descriptions**

| File | Role | Key Responsibility |
|------|------|-------------------|
| `app.py` | Entry Point | Routes `streamlit run app.py` to production UI |
| `src/loanguard/apps/app.py` | Main UI | 6-page Streamlit interface with all user interactions |
| `model.py` | ML Engine | Train logistic regression & decision tree; inference pipeline |
| `preprocessing.py` | Feature Prep | Encode categories, compute debt ratios, standardize features |
| `risk.py` | Risk Scoring | Multi-factor classification (0–100 scale, risk bands) |
| `fraud_detection.py` | Fraud Rules | 5-point anomaly detection; critical threshold 0.50 |
| `explain.py` | Explainability | Feature contributions, decision narratives for each decision |
| `fairness_enhanced.py` | Compliance | 4/5 rule disparate impact, ADEA, income fairness analysis |
| `analytics.py` | Dashboard | KPI computation, batch summaries, performance benchmarks |
| `utils.py` | Helpers | CSV I/O, INR formatting, application logging |

---

## 🚀 Quick Start

### **1. Clone & Setup**
```bash
git clone https://github.com/yourusername/loanguard-ai.git
cd loan_system
```

### **2. Create Virtual Environment** (Optional but recommended)
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Generate Synthetic Dataset** (One-time)
```bash
python scripts/generate_data.py
```
Generates `data/loan_dataset.csv` with 2,000 loan applications across diverse demographics, credit profiles, and loan amounts.

### **5. Train ML Models** (One-time)
```bash
python scripts/train_model.py
```
Produces:
- `models/logistic_regression.pkl` — Primary model (92.25% accuracy, AUC 0.977)
- `models/decision_tree.pkl` — Comparison model (89.25% accuracy, AUC 0.913)
- `models/scaler.pkl` — Feature standardization

### **6. Launch Application**
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. The interface starts on the **Loan Application** page.

---

## 💼 How It Works: Decision Pipeline

### **Stage 1: Applicant Input**
User provides structured data:
- **Demographics:** Age, Gender, Employment Type
- **Financial:** Annual Income, Loan Amount, Existing Loans
- **Credit:** Credit Score, Loan Purpose
- **Requested:** Loan tenor, specific amount needed

### **Stage 2: Feature Engineering**
- Employment type encoded (Salaried→4, Unemployed→0)
- Loan purpose encoded (Home→5, Personal→0)
- Derived ratios: Debt-to-Income, Income-Per-Loan, Credit Score Band

### **Stage 3: ML Prediction**
Logistic Regression model outputs:
- **Approval Probability** (0.0–1.0) — Confidence the applicant qualifies
- **Predicted Label** (Approved/Rejected) — Default decision if no overrides

### **Stage 4: Risk Classification**
Multi-factor engine scores applicant on 0–100 scale:
- Low Risk (0–15): Safe lending candidate
- Medium Risk (15–35): Monitor for additional conditions
- **High Risk (>35): Policy override → Always rejected**

### **Stage 5: Fraud Detection**
5-rule engine assigns fraud_score (0–1):
- Income inflation checks
- Loan stacking pattern detection
- Debt-to-income impossibilities
- Employment-loan mismatch detection
- Age-inappropriate loan duration requests

**If fraud_score ≥ 0.50 → Policy override → Always rejected**

### **Stage 6: Policy Application**
Business guardrails override ML output when:
- ✅ Risk Level = "High Risk" → Change decision to "Rejected"
- ✅ Fraud Score ≥ 0.50 → Change decision to "Rejected"
- ✅ Otherwise → Keep ML decision (Approved/Rejected)

### **Stage 7: Explainability Layer**
For every decision, system generates:
- **Positive Factors:** Why applicant qualifies (good credit, stable employment, low debt ratio)
- **Negative Factors:** Risk signals or fraud red flags
- **Policy Notes:** When/why business rules overrode ML output
- **Risk Components:** Breakdown of risk score across 5 factors

### **Stage 8: Fairness & Compliance Check**
Background analysis of decision pool:
- Approval rate by gender, age, employment type, income bracket
- Flags disparate impact if any group < 80% of comparison group approval rate
- ADEA checks for age discrimination signals
- Logs all decisions for audit trail

### **Stage 9: Output & Audit Log**
Decision stored with:
- Prediction (Approved/Rejected)
- Confidence score
- Risk level
- Fraud signals detected
- Policy overrides applied
- Explanation narrative
- Timestamp & applicant ID

---

## 📊 Example Use Case

**Applicant Profile:**
- Name: Rajesh Kumar
- Age: 32
- Annual Income: ₹50,000
- Credit Score: 620
- Employment: Salaried (Government)
- Existing Loans: 2
- Loan Amount: ₹300,000
- Purpose: Home

**System Processing:**

| Stage | Result | Notes |
|-------|--------|-------|
| **1. ML Prediction** | 68% Approval Probability | Model confidence moderate |
| **2. Risk Score** | 28 (Medium Risk) | Decent credit + stable job offset by debt load |
| **3. Fraud Detection** | 0.15 (Low) | No anomalies detected |
| **4. Risk Classification** | Medium Risk | Safe to proceed, no override needed |
| **5. Policy Application** | No override | Keep ML decision |
| **Final Decision** | **✅ APPROVED** | — |
| **Decision Reasoning** | Salaried employment with stable ₹50K income, 2 existing loans managed responsibly, credit score in Fair range. Debt-to-income ratio of 3.2x is within acceptable limits. Risk profile supports approval. | — |

**Contrast: High-Risk Rejection**

| Stage | Result | Notes |
|-------|--------|-------|
| **1. ML Prediction** | 72% Approval Probability | Model wants to approve |
| **2. Risk Score** | 48 (High Risk) | Low credit (540), high debt ratio (6.8x), 5 existing loans |
| **3. Fraud Detection** | 0.25 (Medium) | Loan stacking pattern detected |
| **4. Risk Classification** | **High Risk** | **Policy Override Triggered** |
| **Final Decision** | **❌ REJECTED** | — |
| **Override Reason** | High aggregate risk score | Risk level exceeds policy threshold; manual review recommended |

---

## ⚖️ Fairness & Bias Monitoring

LoanGuard AI includes real-time compliance checks:

### **Disparate Impact (4/5 Rule)**
Monitors approval rate parity across demographics:
- If Female approval rate < 80% × Male approval rate → **CRITICAL violation flagged**
- If any protected class significantly disadvantaged → **Compliance warning**

### **ADEA (Age Discrimination) Checks**
- Applicants 55+ must not face 15%+ approval rate reduction vs. 25–55 age band
- Detects age-based rejection patterns

### **Income Fairness**
- Low-income (<₹30K) applicants analyzed separately
- Medium-income (₹30K–₹100K) approval rates monitored
- High-income (>₹100K) group benchmarked

### **Employment-Type Equity**
- Salaried vs. Self-Employed vs. Unemployed approval rates tracked
- Unemployed applicants scrutinized for extra restrictions

### **Recommendations Engine**
If violations detected, system suggests:
- Adjust approval thresholds to equalize outcomes
- Add human review layer for borderline cases
- Retrain model with fairness constraints
- Document business justification for disparities

---

## 🧪 Testing & Validation

### **Data Generation**
```bash
python scripts/generate_data.py
```
Creates 2,000-row synthetic dataset with realistic distributions:
- Age: 25–65 years
- Income: ₹10K–₹250K/year
- Credit Score: 450–800
- Existing Loans: 0–6
- Employment: Salaried, Self-Employed, Business Owner, Freelancer, Unemployed (realistic ratios)

### **Model Training & Evaluation**
```bash
python scripts/train_model.py
```
Output metrics:
```
Logistic Regression:
  Accuracy: 92.25%
  AUC-ROC: 0.977
  Precision: 91.50%
  Recall: 93.10%

Decision Tree:
  Accuracy: 89.25%
  AUC-ROC: 0.913
```

### **Manual Testing in Streamlit**
1. Navigate to **Loan Application** page
2. Enter applicant details (form auto-validates)
3. Click "Evaluate Loan Application"
4. View decision, risk score, fraud score, and full explanation
5. Navigate to **Fairness & Bias** page to view compliance status

---

## 📈 Dashboard & Analytics

The **Analytics Dashboard** presents:

- **KPI Cards:** Total applications, approval rate, avg risk score, fraud detection efficiency
- **Approval Trend:** Line chart of approval decisions over time
- **Risk Distribution:** Bar chart showing Low/Medium/High Risk breakdowns
- **Model Performance:** Confusion matrix, AUC curve, feature importance
- **Cohort Breakdown:** Approval rates by age band, employment type, income bracket
- **Processing Metrics:** Average inference time, batch throughput

---

## 🔄 Deployment & Production Use

### **Single Application Mode**
- Use **Loan Application** page for real-time decisions
- Decision latency: <500ms
- Suitable for relationship manager tools

### **Batch Processing Mode**
- Upload CSV files (single or multiple)
- Process 100s of applications in parallel
- Export results with full explainability
- Compliance report automatically generated

### **API Deployment** (Future)
```python
# Planned FastAPI wrapper for REST access
POST /api/v1/predict
{
  "applicant": {...},
  "mode": "explain|risk|fraud|fairness"
}
→ Returns: decision, explanation, compliance flags
```

### **Database Integration** (Future)
- Connect to bank's core CRM system
- Real-time application pulling
- Decision history storage
- Audit log export to compliance team

---

## 🚦 Decision Policy Guardrails

To prevent harmful lending practices, LoanGuard AI applies business rules **on top of ML predictions**:

| Condition | Action | Rationale |
|-----------|--------|-----------|
| Risk Level = High (>35) | Override to Rejected | High borrower risk |
| Fraud Score ≥ 0.50 | Override to Rejected | Suspicious patterns |
| ML Prediction contradicts policy | Apply policy | Business rules supersede models |

These ensure:
- ✅ Compliance with risk appetite
- ✅ Fraud prevention
- ✅ Legal defensibility (decisions have explicit business reasons)
- ✅ Explainability (never approve due to model alone if high risk)

---

## 🎓 Key Insights & Learnings

### **Why Explainability Matters**
Traditional "black-box" ML models cannot defend lending decisions to regulators or applicants. LoanGuard AI provides narratives for every decision, enabling:
- Regulatory audit readiness
- Applicant appeals handling
- Relationship manager training
- Model debugging & improvement

### **Why Policy Guardrails Matter**
ML models optimize accuracy but may miss edge cases or follow unintended patterns. Business overlays ensure:
- High-risk applicants never slip through
- Fraud signals get immediate escalation
- Risk appetite is respected
- No unfair systematic approvals by accident

### **Why Fairness Checks Matter**
Unintentional bias in lending causes real harm and legal liability. Continuous monitoring ensures:
- Protected classes treated equitably
- Disparate impact detected early
- Corrective actions possible before harm scales

---

## 📦 Dependencies & Requirements

See `requirements.txt`:

```
streamlit>=1.32.0         # Interactive UI
pandas>=2.0.0             # Data manipulation
numpy>=1.24.0             # Numerical computing
scikit-learn>=1.3.0       # ML models (LogReg, DecisionTree)
plotly>=5.18.0            # Interactive charts
matplotlib>=3.7.0         # Static charts
seaborn>=0.12.0           # Statistical visualizations
joblib>=1.3.0             # Model serialization
graphviz>=0.20.0          # Flowchart rendering
```

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:

1. **Advanced ML Models**
   - Gradient Boosting (XGBoost, LightGBM) for higher accuracy
   - Neural Networks for deep decision boundary learning
   - Ensemble voting across 3+ models

2. **Feature Additions**
   - SHAP-based true feature importance ranking
   - Applicant appeal workflow with manual override UI
   - Real-time model retraining on new applications
   - A/B testing framework for policy changes

3. **Production Hardening**
   - REST API wrapper (FastAPI)
   - Database storage for audit logs
   - Multi-tenant support for multiple banks
   - Real-time alerting for fraud/fairness violations

4. **Fairness Enhancements**
   - Causal fairness analysis (not just correlational)
   - Multi-dimensional equity (intersectional fairness)
   - Fairness-constrained model retraining
   - Transparent trade-off controls (accuracy vs. fairness sliders)

---

## 📝 License

This project is provided as-is for educational and demonstration purposes.

---

## 👨‍💼 Author & Contact

**LoanGuard AI Development Team**

For questions, suggestions, or partnership inquiries:
- 📧 Email: team@loanguard.example
- 🐙 GitHub: [github.com/yourorg/loanguard-ai](https://github.com)
- 💼 LinkedIn: Coming soon

---

## 🎯 Vision & Roadmap

**Current State (v1.0):**
- ✅ Real-time single application scoring
- ✅ Batch CSV processing
- ✅ Fairness & bias monitoring
- ✅ Full decision explainability
- ✅ Interactive what-if simulator

**Planned (v2.0):**
- 🔄 REST API for bank core systems
- 🔄 Real-time model retraining with data drift detection
- 🔄 Advanced ML ensembles (XGBoost, LightGBM)
- 🔄 SHAP-based true feature importance
- 🔄 Causal fairness analysis

**Future (v3.0+):**
- 🌐 Multi-bank, multi-product platform (mortgages, credit cards, business loans)
- 🌐 Blockchain-based audit trail for immutable compliance records
- 🌐 Federated learning for privacy-preserving model sharing across institutions
- 🌐 Real-time regulatory filing automation (ECOA, FHA, FCRA compliance)

---

**Built with ❤️ for transparent, fair, and intelligent lending.**
