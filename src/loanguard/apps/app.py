"""
app.py - Production-Grade Loan Intelligence System
Real-Time Decision Engine with Advanced Analytics, Fairness Monitoring & Scenario Simulation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
import time
import io
from datetime import datetime
from loanguard.core.model import predict, load_model
from loanguard.core.explain import generate_explanation
from loanguard.core.risk import classify_risk
from loanguard.core.fraud_detection import detect_fraud
from loanguard.core.utils import save_application, load_applications, load_training_data, compute_fairness_metrics, format_inr
from loanguard.core.analytics import generate_insights, compute_batch_summary, get_pipeline_stages, get_processing_comparison, get_performance_benchmark
from loanguard.core.fairness_enhanced import compute_fairness_metrics_enhanced, get_fairness_dashboard_metrics


def apply_decision_policy(prediction: dict, risk: dict, fraud: dict) -> dict:
    """Apply business guardrails on top of model output for final approval decision."""
    final = prediction.copy()
    override_reasons = []

    if risk.get("level") == "High Risk":
        override_reasons.append("High aggregate risk score")

    # High/critical fraud risk should always escalate to rejection.
    if fraud.get("fraud_score", 0.0) >= 0.50:
        override_reasons.append("High fraud risk score")

    if override_reasons:
        final["prediction"] = 0
        final["label"] = "Rejected"

    final["override_reasons"] = override_reasons
    final["is_overridden"] = bool(override_reasons)
    return final

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="LoanGuard AI 🏦",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ADVANCED STYLING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* { font-family: 'Space Grotesk', sans-serif; }

.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 100%); color: #e2e8f0; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #0a0e1a 100%);
    border-right: 2px solid #1e293b;
}

/* Cards & Containers */
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

.result-approved {
    background: linear-gradient(135deg, #052e16 0%, #064e3b 100%);
    border: 2px solid #22c55e;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 8px 24px rgba(34, 197, 94, 0.2);
}

.result-rejected {
    background: linear-gradient(135deg, #1c0404 0%, #3b0a0a 100%);
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 8px 24px rgba(239, 68, 68, 0.2);
}

.insight-card {
    background: #111827;
    border-left: 4px solid;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}

.insight-high { border-left-color: #ef4444; }
.insight-medium { border-left-color: #f59e0b; }
.insight-info { border-left-color: #3b82f6; }

.feature-importance-bar { background: linear-gradient(90deg, #3b82f6, #2563eb); }

.fraud-score-display {
    font-size: 2.5rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    text-align: center;
}

.pipeline-stage {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    margin: 0.5rem;
}

/* Streamlit overrides */
.stSelectbox label, .stNumberInput label, .stSlider label, .stButton label {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(37, 99, 235, 0.35) !important;
}

div[data-testid="stMetricValue"] { color: #60a5fa !important; font-family: 'JetBrains Mono', monospace !important; }

.stTabs [data-baseweb="tab-list"] { background: #111827 !important; border-radius: 10px !important; gap: 4px !important; }

.stTabs [data-baseweb="tab"] { color: #94a3b8 !important; border-radius: 8px !important; }

.stTabs [aria-selected="true"] { background: #1e3a8a !important; color: #93c5fd !important; }

hr { border-color: #1e293b !important; }
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0;">
        <div style="font-size:3rem;">🏦</div>
        <div style="font-size:1.5rem; font-weight:700; color:#60a5fa; letter-spacing:0.05em;">LoanGuard AI</div>
        <div style="font-size:0.75rem; color:#475569; margin-top:0.5rem; text-transform:uppercase; letter-spacing:0.1em;">⚡ Production Decision Engine v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📋 Loan Application", "🧮 Scenario Simulator", "📂 Batch Processing", "📊 Analytics Dashboard", "⚖️ Fairness & Bias", "🔧 System Pipeline"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    model_choice = st.selectbox(
        "🤖 Active Model",
        ["logistic_regression", "decision_tree"],
        format_func=lambda x: "Logistic Regression ⭐ (Preferred)" if x == "logistic_regression" else "Decision Tree"
    )

    st.markdown("---")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Processing Speed", "0.8s", help="Average decision time")
    with col_s2:
        st.metric("Uptime", "99.9%", help="System availability")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.7rem; color:#64748b; text-align:center; line-height:2;">
        <div style="color:#94a3b8; margin-bottom:0.3rem; font-weight:600;">System Status</div>
        🟢 Model Online | 🟢 Fraud Engine | 🟢 Storage | 🟢 Explainer
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 1: LOAN APPLICATION (with Real-Time Stages)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if page == "📋 Loan Application":

    st.markdown("""
    <div style="margin-bottom:2rem;">
        <h1 style="color:#f1f5f9; font-size:2.2rem; font-weight:700; margin:0;">⚡ Instant Loan Assessment</h1>
        <p style="color:#64748b; margin-top:0.5rem; font-size:0.95rem;">AI-powered decision in real-time with complete transparency</p>
    </div>
    """, unsafe_allow_html=True)

    # Form
    with st.form("loan_form", clear_on_submit=False):
        st.markdown('<h3 style="color:#93c5fd; margin-bottom:1rem;">👤 Applicant Profile</h3>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", min_value=18, max_value=75, value=32)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        with col3:
            employment = st.selectbox("Employment", ["Salaried", "Self-Employed", "Business Owner", "Freelancer", "Unemployed"])

        st.markdown('<h3 style="color:#93c5fd; margin-top:1.5rem; margin-bottom:1rem;">💰 Financial Profile</h3>', unsafe_allow_html=True)

        col4, col5 = st.columns(2)
        with col4:
            income = st.number_input("Annual Income (₹)", min_value=0, max_value=10_000_000, value=600000, step=50000)
        with col5:
            credit_score = st.slider("Credit Score (CIBIL)", min_value=300, max_value=850, value=720, step=5)

        col6, col7 = st.columns(2)
        with col6:
            existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=1)
        with col7:
            loan_amount = st.number_input("Loan Amount (₹)", min_value=10000, max_value=50_000_000, value=500000, step=50000)

        loan_purpose = st.selectbox("Purpose", ["Home", "Education", "Vehicle", "Business", "Medical", "Personal"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡ PROCESS APPLICATION", use_container_width=True)

    # Process
    if submitted:
        applicant = {
            "Age": age, "Gender": gender, "Employment_Type": employment,
            "Annual_Income": income, "Credit_Score": credit_score,
            "Existing_Loans": existing_loans, "Loan_Amount": loan_amount, "Loan_Purpose": loan_purpose,
        }

        # Real-time processing stages
        st.markdown("---")
        stages_container = st.container()
        
        with stages_container:
            col_stages = st.columns(6)
            stages = [
                ("Validating", "✅"),
                ("ML Model", "🧮"),
                ("Risk Check", "⚖️"),
                ("Fraud Scan", "🚨"),
                ("Explaining", "🧠"),
                ("Finalizing", "📋"),
            ]
            
            for idx, (stage, icon) in enumerate(stages):
                with col_stages[idx]:
                    st.markdown(f"<div style='text-align:center; color:#60a5fa;'>{icon}<br><small>{stage}</small></div>", unsafe_allow_html=True)

        # Run pipeline
        time_start = time.time()
        
        # Stage 1: Validate
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Stage 1/6: Validating input data...")
        time.sleep(0.3)
        progress_bar.progress(15)

        # Stage 2: Predict
        status_text.text("Stage 2/6: Running ML prediction model...")
        prediction = predict(applicant, model_name=model_choice)
        time.sleep(0.2)
        progress_bar.progress(35)

        # Stage 3: Risk
        status_text.text("Stage 3/6: Computing risk classification...")
        risk = classify_risk(applicant, prediction["approval_probability"])
        time.sleep(0.15)
        progress_bar.progress(50)

        # Stage 4: Fraud
        status_text.text("Stage 4/6: Scanning for fraud patterns...")
        fraud = detect_fraud(applicant)
        time.sleep(0.2)
        progress_bar.progress(70)

        # Stage 4.5: Policy decision guardrails
        prediction = apply_decision_policy(prediction, risk, fraud)

        # Stage 5: Explain
        status_text.text("Stage 5/6: Generating explanation...")
        explanation = generate_explanation(applicant, prediction["prediction"], prediction["approval_probability"])
        time.sleep(0.15)
        progress_bar.progress(85)

        # Stage 6: Store
        status_text.text("Stage 6/6: Storing application record...")
        app_id = save_application(applicant, prediction, risk, fraud)
        time.sleep(0.1)
        progress_bar.progress(100)
        time.sleep(0.3)
        
        status_text.empty()
        progress_bar.empty()

        total_time = time.time() - time_start

        st.markdown("---")

        # RESULT
        col_r1, col_r2 = st.columns([2.5, 1])

        with col_r1:
            if prediction["label"] == "Approved":
                st.markdown(f"""
                <div class="result-approved">
                    <h2 style="color:#4ade80; margin:0;">✅ APPROVED</h2>
                    <p style="color:#86efac; font-size:1rem; margin:0.8rem 0 0;">{explanation['summary']}</p>
                    <div style="margin-top:1rem; font-family:'JetBrains Mono'; font-size:0.9rem; color:#4ade80;">
                        📊 Approval Confidence: <strong>{prediction['approval_probability']*100:.1f}%</strong>
                    </div>
                    <div style="color:#64748b; font-size:0.75rem; margin-top:0.5rem;">ID: {app_id} | Processed in {total_time:.2f}s</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                override_html = ""
                if prediction.get("is_overridden"):
                    reasons = " | ".join(prediction.get("override_reasons", []))
                    override_html = f"<div style='margin-top:0.8rem; color:#fca5a5; font-size:0.85rem;'>Policy override applied: {reasons}</div>"
                st.markdown(f"""
                <div class="result-rejected">
                    <h2 style="color:#f87171; margin:0;">❌ REJECTED</h2>
                    <p style="color:#fca5a5; font-size:1rem; margin:0.8rem 0 0;">{explanation['summary']}</p>
                    <div style="margin-top:1rem; font-family:'JetBrains Mono'; font-size:0.9rem; color:#f87171;">
                        📊 Approval Probability: <strong>{prediction['approval_probability']*100:.1f}%</strong>
                    </div>
                    {override_html}
                    <div style="color:#64748b; font-size:0.75rem; margin-top:0.5rem;">ID: {app_id} | Processed in {total_time:.2f}s</div>
                </div>
                """, unsafe_allow_html=True)

        with col_r2:
            # Confidence gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction["approval_probability"] * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Confidence Score", "font": {"color": "#94a3b8"}},
                number={"suffix": "%", "font": {"color": "#60a5fa", "size": 32}},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"color": "#64748b"}},
                    "bar": {"color": "#22c55e" if prediction["prediction"] == 1 else "#ef4444"},
                    "bgcolor": "#1e293b",
                    "steps": [
                        {"range": [0, 33], "color": "#1c0404"},
                        {"range": [33, 67], "color": "#1c1005"},
                        {"range": [67, 100], "color": "#052e16"},
                    ]
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e2e8f0"}, height=250, margin=dict(t=30, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Risk level
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="display:inline-block; padding:0.5rem 1.2rem; border-radius:50px; background:rgba(0,0,0,0.3); border:2px solid {risk['color']}; color:{risk['color']}; font-weight:600;">
                    {risk['icon']} {risk['level']}
                </div>
                <div style="color:#64748b; font-size:0.8rem; margin-top:0.5rem;">Risk Score: {risk['score']}/100</div>
            </div>
            """, unsafe_allow_html=True)

        # TABS
        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🧠 Decision Factors", "📊 Feature Importance", "🚨 Fraud Analysis", "💰 Financial Ratios", "⏱️ Performance"])

        with tab1:
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.markdown('<h4 style="color:#4ade80;">✅ Supporting Factors</h4>', unsafe_allow_html=True)
                if explanation["positive_factors"]:
                    for f in explanation["positive_factors"]:
                        st.markdown(f'<div style="background:rgba(34,197,94,0.1); border-left:3px solid #22c55e; padding:0.7rem; margin:0.4rem 0; border-radius:4px; color:#86efac;">✓ {f}</div>', unsafe_allow_html=True)
                else:
                    st.info("No significant positive factors")

            with col_e2:
                st.markdown('<h4 style="color:#f87171;">❌ Risk Factors</h4>', unsafe_allow_html=True)
                if explanation["negative_factors"]:
                    for f in explanation["negative_factors"]:
                        st.markdown(f'<div style="background:rgba(239,68,68,0.1); border-left:3px solid #ef4444; padding:0.7rem; margin:0.4rem 0; border-radius:4px; color:#fca5a5;">✗ {f}</div>', unsafe_allow_html=True)
                else:
                    st.info("No significant risk factors")

        with tab2:
            st.markdown('<h4>Feature Contribution to Decision</h4>', unsafe_allow_html=True)
            st.text("How much each factor influenced the decision (higher = more influential):")
            
            feature_contrib = explanation.get("feature_contributions", [])
            if feature_contrib:
                fig_features = go.Figure()
                features_names = [f[0].replace("_", " ") for f in feature_contrib]
                features_scores = [f[1] for f in feature_contrib]
                
                fig_features.add_trace(go.Bar(
                    y=features_names,
                    x=features_scores,
                    orientation='h',
                    marker=dict(
                        color=features_scores,
                        colorscale='Reds',
                        showscale=False
                    ),
                    text=[f"{s:.2f}" for s in features_scores],
                    textposition="auto",
                ))
                
                fig_features.update_layout(
                    xaxis=dict(range=[0, 1], title="Contribution Score", gridcolor="#1e293b"),
                    yaxis_title="Feature",
                    plot_bgcolor="#111827",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"),
                    height=300,
                    margin=dict(l=150, r=20)
                )
                st.plotly_chart(fig_features, use_container_width=True)

        with tab3:
            col_f1, col_f2 = st.columns([1, 2])
            
            with col_f1:
                st.markdown(f'<div class="fraud-score-display" style="color:{fraud["color"]};">{fraud["fraud_score"]:.2f}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="text-align:center; color:{fraud["color"]}; font-weight:600; font-size:1.1rem;">{fraud["fraud_level"]}</div>', unsafe_allow_html=True)

            with col_f2:
                st.markdown(f'<div style="background:{fraud["color"]}15; border: 2px solid {fraud["color"]}; border-radius:8px; padding:1rem; color:#e2e8f0;">{fraud["recommendation"]}</div>', unsafe_allow_html=True)

            if fraud["flags"]:
                st.markdown('<h4 style="margin-top:1.5rem;">🚩 Detected Anomalies:</h4>', unsafe_allow_html=True)
                for i, flag in enumerate(fraud["flags"], 1):
                    st.markdown(f'<div style="background:#1c0a00; border-left:3px solid #f59e0b; padding:0.7rem; margin:0.4rem 0; border-radius:4px; color:#fcd34d;">{i}. {flag}</div>', unsafe_allow_html=True)

            # Fraud sub-scores
            col_fs1, col_fs2, col_fs3, col_fs4 = st.columns(4)
            with col_fs1:
                st.metric("Income Anomaly", f"{fraud['income_anomaly']:.2f}", help="0-1 scale")
            with col_fs2:
                st.metric("Loan Overload", f"{fraud['loan_overload']:.2f}", help="0-1 scale")
            with col_fs3:
                st.metric("Credit Inconsistency", f"{fraud['credit_inconsistency']:.2f}", help="0-1 scale")
            with col_fs4:
                st.metric("Age Mismatch", f"{fraud['age_mismatch']:.2f}", help="0-1 scale")

        with tab4:
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.metric("Debt-to-Income", f"{explanation['debt_ratio']:.2f}x", 
                         delta="Safe" if explanation['debt_ratio'] <= 3 else "Elevated" if explanation['debt_ratio'] <= 6 else "High")
            with col_r2:
                st.metric("Loan-to-Income", f"{explanation['loan_to_income_ratio']:.2f}x",
                         delta="Good" if explanation['loan_to_income_ratio'] <= 5 else "Caution")

        with tab5:
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("Processing Time", f"{total_time:.3f}s", "⚡ Real-time")
            with col_p2:
                vs_traditional = 48 * 3600 / total_time
                st.metric("vs Traditional", f"{vs_traditional:,.0f}x", "faster")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 2: SCENARIO SIMULATOR (What-If Analysis)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "🧮 Scenario Simulator":

    st.markdown("""
    <div style="margin-bottom:2rem;">
        <h1 style="color:#f1f5f9; font-size:2.2rem; font-weight:700; margin:0;">🧮 Interactive What-If Analysis</h1>
        <p style="color:#64748b; margin-top:0.5rem; font-size:0.95rem;">Adjust parameters to see real-time impact on loan decisions</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 style="color:#93c5fd; margin-bottom:1.5rem;">Base Applicant Profile</h3>', unsafe_allow_html=True)

    # Base values
    col1, col2, col3 = st.columns(3)
    with col1:
        age_base = st.slider("Age", 18, 75, 35, key="age_sim")
    with col2:
        employment_base = st.selectbox("Employment", ["Salaried", "Self-Employed", "Business Owner", "Freelancer", "Unemployed"], key="emp_sim")
    with col3:
        gender_base = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender_sim")

    st.markdown("---")
    st.markdown('<h3 style="color:#93c5fd; margin-bottom:1.5rem;">📊 Simulation Parameters (Adjust & Watch Impact)</h3>', unsafe_allow_html=True)

    col_sim1, col_sim2 = st.columns(2)

    with col_sim1:
        st.markdown("**💰 Income Range**")
        income_sim = st.slider("Annual Income (₹)", 0, 2_000_000, 600000, 50000, key="income_sim", help="Drag to see real-time impact")
        
    with col_sim2:
        st.markdown("**📈 Credit Score Range**")
        credit_sim = st.slider("Credit Score", 300, 850, 720, 10, key="credit_sim", help="Drag to see real-time impact")

    col_sim3, col_sim4 = st.columns(2)

    with col_sim3:
        st.markdown("**🏦 Loan Amount**")
        loan_sim = st.slider("Requested Amount (₹)", 10000, 5_000_000, 500000, 50000, key="loan_sim", help="Drag to see real-time impact")

    with col_sim4:
        st.markdown("**🔗 Existing Loans**")
        existing_sim = st.slider("Number of Active Loans", 0, 10, 1, 1, key="existing_sim", help="Drag to see real-time impact")

    loan_purpose_sim = st.selectbox("Loan Purpose", ["Home", "Education", "Vehicle", "Business", "Medical", "Personal"], key="purpose_sim")

    # Compute scenarios
    st.markdown("---")
    st.markdown('<h3 style="color:#93c5fd; margin-bottom:1.5rem;">📊 Scenario Analysis</h3>', unsafe_allow_html=True)

    applicant_sim = {
        "Age": age_base, "Gender": gender_base, "Employment_Type": employment_base,
        "Annual_Income": income_sim, "Credit_Score": credit_sim,
        "Existing_Loans": existing_sim, "Loan_Amount": loan_sim, "Loan_Purpose": loan_purpose_sim,
    }

    pred_sim = predict(applicant_sim, model_name=model_choice)
    risk_sim = classify_risk(applicant_sim, pred_sim["approval_probability"])
    fraud_sim = detect_fraud(applicant_sim)
    pred_sim = apply_decision_policy(pred_sim, risk_sim, fraud_sim)

    col_out1, col_out2, col_out3 = st.columns(3)

    with col_out1:
        result_text = "✅ APPROVED" if pred_sim["label"] == "Approved" else "❌ REJECTED"
        result_color = "#22c55e" if pred_sim["label"] == "Approved" else "#ef4444"
        st.markdown(f'<div style="background:rgba(0,0,0,0.3); border:2px solid {result_color}; border-radius:10px; padding:1.5rem; text-align:center;"><div style="color:{result_color}; font-weight:700; font-size:1.3rem;">{result_text}</div><div style="color:#94a3b8; font-size:0.9rem; margin-top:0.5rem;">Probability: {pred_sim["approval_probability"]*100:.1f}%</div></div>', unsafe_allow_html=True)

    with col_out2:
        st.markdown(f'<div style="background:rgba(0,0,0,0.3); border:2px solid {risk_sim["color"]}; border-radius:10px; padding:1.5rem; text-align:center;"><div style="color:{risk_sim["color"]}; font-weight:700; font-size:1.1rem;">{risk_sim["icon"]} {risk_sim["level"]}</div><div style="color:#94a3b8; font-size:0.9rem; margin-top:0.5rem;">Risk Score: {risk_sim["score"]}/100</div></div>', unsafe_allow_html=True)

    with col_out3:
        st.markdown(f'<div style="background:rgba(0,0,0,0.3); border:2px solid {fraud_sim["color"]}; border-radius:10px; padding:1.5rem; text-align:center;"><div style="color:{fraud_sim["color"]}; font-weight:700; font-size:1.1rem;">{fraud_sim["fraud_level"]}</div><div style="color:#94a3b8; font-size:0.9rem; margin-top:0.5rem;">Fraud Score: {fraud_sim["fraud_score"]:.2f}/1.0</div></div>', unsafe_allow_html=True)

    # Sensitivity analysis: show how approval changes with each variable
    st.markdown("---")
    st.markdown('<h4>📈 Sensitivity Analysis - Impact on Approval Probability</h4>', unsafe_allow_html=True)

    sensitivity_data = []

    # Income sensitivity
    for inc in [300000, 500000, 700000, 900000, 1100000]:
        app_test = applicant_sim.copy()
        app_test["Annual_Income"] = inc
        pred_test = predict(app_test, model_name=model_choice)
        sensitivity_data.append({"Variable": "Income (₹)", "Value": f"₹{inc/100000:.0f}L", "Approval %": pred_test["approval_probability"] * 100})

    # Credit score sensitivity
    for cs in [550, 650, 750, 800]:
        app_test = applicant_sim.copy()
        app_test["Credit_Score"] = cs
        pred_test = predict(app_test, model_name=model_choice)
        sensitivity_data.append({"Variable": "Credit Score", "Value": cs, "Approval %": pred_test["approval_probability"] * 100})

    # Loan amount sensitivity
    for la in [200000, 400000, 600000, 800000]:
        app_test = applicant_sim.copy()
        app_test["Loan_Amount"] = la
        pred_test = predict(app_test, model_name=model_choice)
        sensitivity_data.append({"Variable": "Loan Amount (₹)", "Value": f"₹{la/100000:.0f}L", "Approval %": pred_test["approval_probability"] * 100})

    sens_df = pd.DataFrame(sensitivity_data)

    fig_sens = go.Figure()
    for var in sens_df["Variable"].unique():
        var_data = sens_df[sens_df["Variable"] == var]
        fig_sens.add_trace(go.Scatter(
            x=var_data["Value"].astype(str),
            y=var_data["Approval %"],
            mode="lines+markers",
            name=var,
            hovertemplate="<b>%{x}</b><br>Approval: %{y:.1f}%<extra></extra>"
        ))

    fig_sens.update_layout(
        title="How Parameters Affect Approval Probability",
        xaxis_title="Parameter Value",
        yaxis_title="Approval Probability (%)",
        plot_bgcolor="#111827",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        hovermode="x unified",
        height=350,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_sens, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 3: BATCH PROCESSING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "📂 Batch Processing":

    st.markdown("""
    <div style="margin-bottom:2rem;">
        <h1 style="color:#f1f5f9; font-size:2.2rem; font-weight:700; margin:0;">📂 Bulk Processing</h1>
        <p style="color:#64748b; margin-top:0.5rem; font-size:0.95rem;">Process multiple applications from CSV in seconds</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"], help="Required columns: Age, Gender, Employment_Type, Annual_Income, Credit_Score, Existing_Loans, Loan_Amount, Loan_Purpose")

    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        st.markdown(f"**📊 Loaded {len(batch_df)} applications**")
        
        # Preview
        with st.expander("👀 Preview Data", expanded=False):
            st.dataframe(batch_df.head())

        if st.button("⚡ Process All Applications", use_container_width=True):
            progress_bar = st.progress(0)
            status = st.empty()

            results = []
            for idx, row in batch_df.iterrows():
                status.text(f"Processing {idx+1}/{len(batch_df)}...")
                
                try:
                    applicant_batch = row.to_dict()
                    pred_batch = predict(applicant_batch, model_name=model_choice)
                    risk_batch = classify_risk(applicant_batch, pred_batch["approval_probability"])
                    fraud_batch = detect_fraud(applicant_batch)
                    pred_batch = apply_decision_policy(pred_batch, risk_batch, fraud_batch)
                    
                    results.append({
                        "Index": idx + 1,
                        "Prediction": pred_batch["label"],
                        "Approval_Probability": round(pred_batch["approval_probability"], 3),
                        "Risk_Level": risk_batch["level"],
                        "Fraud_Flags": fraud_batch["flag_count"],
                        "Fraud_Score": fraud_batch["fraud_score"],
                    })
                except Exception as e:
                    st.warning(f"Row {idx+1} error: {str(e)}")

                progress_bar.progress((idx + 1) / len(batch_df))

            status.empty()
            progress_bar.empty()

            # Results
            results_df = pd.DataFrame(results)
            st.markdown("---")
            st.markdown('<h3 style="color:#93c5fd;">📊 Batch Processing Results</h3>', unsafe_allow_html=True)

            col_b1, col_b2, col_b3, col_b4 = st.columns(4)

            summary = compute_batch_summary(results_df)
            with col_b1:
                st.metric("Total Processed", summary["total_applications"])
            with col_b2:
                st.metric("Approved", summary["approved"], f"{summary['approval_rate']*100:.1f}%")
            with col_b3:
                st.metric("Rejected", summary["rejected"], f"{(1-summary['approval_rate'])*100:.1f}%")
            with col_b4:
                st.metric("Avg Confidence", f"{summary['avg_approval_probability']*100:.1f}%")

            # Risk distribution
            if "risk_distribution" in summary:
                fig_risk_dist = go.Figure(data=[
                    go.Bar(x=list(summary["risk_distribution"].keys()), y=list(summary["risk_distribution"].values()),
                           marker=dict(color=["#22c55e", "#f59e0b", "#ef4444"]))
                ])
                fig_risk_dist.update_layout(
                    title="Risk Distribution", plot_bgcolor="#111827", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"), height=300, showlegend=False
                )
                st.plotly_chart(fig_risk_dist, use_container_width=True)

            # Detailed results
            st.dataframe(results_df, use_container_width=True)

            # Download
            csv = results_df.to_csv(index=False)
            st.download_button("⬇️ Download Results (CSV)", csv, file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 4: ANALYTICS DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "📊 Analytics Dashboard":

    st.markdown("""
    <div style="margin-bottom:2rem;">
        <h1 style="color:#f1f5f9; font-size:2.2rem; font-weight:700; margin:0;">📊 Portfolio Analytics & Insights</h1>
        <p style="color:#64748b; margin-top:0.5rem; font-size:0.95rem;">AI-generated business intelligence from historical applications</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    apps_df = load_applications()

    if len(apps_df) == 0:
        st.info("📭 No applications processed yet. Process some loans first to see analytics.")
    else:
        # Insights
        insights = generate_insights(apps_df)

        st.markdown('<h3 style="color:#93c5fd; margin-bottom:1rem;">💡 AI-Generated Insights</h3>', unsafe_allow_html=True)

        if insights:
            for insight in insights:
                icon = "🔴" if insight["severity"] == "high" else "🟡" if insight["severity"] == "medium" else "ℹ️"
                color = "#ef4444" if insight["severity"] == "high" else "#f59e0b" if insight["severity"] == "medium" else "#3b82f6"
                
                st.markdown(f"""
                <div style="background:rgba(0,0,0,0.3); border-left:4px solid {color}; border-radius:8px; padding:1rem; margin:0.5rem 0;">
                    <div style="color:{color}; font-weight:600; font-size:1rem;">{icon} {insight['title']}</div>
                    <div style="color:#cbd5e1; font-size:0.9rem; margin-top:0.3rem;">{insight['description']}</div>
                    <div style="color:#64748b; font-size:0.8rem; margin-top:0.3rem;">Category: {insight['category']} | Metric: {insight['metric']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # KPI Cards
        st.markdown('<h3 style="color:#93c5fd; margin-bottom:1rem;">📈 Key Performance Indicators</h3>', unsafe_allow_html=True)

        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

        with col_kpi1:
            total_apps = len(apps_df)
            st.metric("Total Applications", total_apps)

        with col_kpi2:
            approval_rate = (apps_df["Prediction"] == "Approved").mean() if "Prediction" in apps_df.columns else 0
            st.metric("Approval Rate", f"{approval_rate*100:.1f}%", f"{int(approval_rate*total_apps)} approved")

        with col_kpi3:
            if "Fraud_Flags" in apps_df.columns:
                fraud_count = (apps_df["Fraud_Flags"] > 0).sum()
                st.metric("Fraud Flagged", fraud_count, f"{fraud_count/max(total_apps,1)*100:.1f}%")

        with col_kpi4:
            if "Risk_Level" in apps_df.columns:
                high_risk = (apps_df["Risk_Level"] == "High Risk").sum()
                st.metric("High Risk", high_risk, f"{high_risk/max(total_apps,1)*100:.1f}%")

        st.markdown("---")

        # Visualizations
        st.markdown('<h3 style="color:#93c5fd; margin-bottom:1rem;">📊 Portfolio Composition</h3>', unsafe_allow_html=True)

        col_viz1, col_viz2 = st.columns(2)

        with col_viz1:
            if "Prediction" in apps_df.columns:
                prediction_counts = apps_df["Prediction"].value_counts()
                fig_pred = go.Figure(go.Pie(labels=prediction_counts.index, values=prediction_counts.values,
                                            marker=dict(colors=["#22c55e", "#ef4444"])))
                fig_pred.update_layout(title="Approval vs Rejection", plot_bgcolor="#111827", paper_bgcolor="rgba(0,0,0,0)",
                                      font=dict(color="#e2e8f0"), height=300)
                st.plotly_chart(fig_pred, use_container_width=True)

        with col_viz2:
            if "Risk_Level" in apps_df.columns:
                risk_counts = apps_df["Risk_Level"].value_counts()
                fig_risk = go.Figure(go.Pie(labels=risk_counts.index, values=risk_counts.values,
                                           marker=dict(colors=["#22c55e", "#f59e0b", "#ef4444"])))
                fig_risk.update_layout(title="Risk Distribution", plot_bgcolor="#111827", paper_bgcolor="rgba(0,0,0,0)",
                                      font=dict(color="#e2e8f0"), height=300)
                st.plotly_chart(fig_risk, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 5: FAIRNESS & BIAS MONITORING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "⚖️ Fairness & Bias":

    st.markdown("""
    <div style="margin-bottom:2rem;">
        <h1 style="color:#f1f5f9; font-size:2.2rem; font-weight:700; margin:0;">⚖️ Fair Lending & Bias Detection</h1>
        <p style="color:#64748b; margin-top:0.5rem; font-size:0.95rem;">Regulatory compliance monitoring & disparate impact analysis</p>
    </div>
    """, unsafe_allow_html=True)

    apps_df = load_applications()

    if len(apps_df) == 0:
        st.info("📭 No applications yet. Process applications to enable fairness analysis.")
    else:
        fairness_results = compute_fairness_metrics_enhanced(apps_df)

        # Compliance Status
        st.markdown('<h3 style="color:#93c5fd; margin-bottom:1rem;">🏛️ Regulatory Compliance Status</h3>', unsafe_allow_html=True)

        compliance_status = fairness_results.get("compliance_status", "[COMPLIANT]")
        comp_color = "#22c55e" if compliance_status in ["✅ COMPLIANT", "[COMPLIANT]"] else "#ef4444"
        st.markdown(f"""
        <div style="background:rgba(0,0,0,0.3); border:2px solid {comp_color}; border-radius:10px; padding:1.5rem; text-align:center;">
            <div style="color:{comp_color}; font-weight:700; font-size:1.4rem; margin-bottom:0.5rem;">{compliance_status}</div>
            <div style="color:#cbd5e1; font-size:1rem;">{fairness_results.get('compliance_summary', 'Fairness checks completed.')}</div>
        </div>
        """, unsafe_allow_html=True)

        # Violations
        if fairness_results["has_bias_violations"]:
            st.markdown("---")
            st.markdown('<h3 style="color:#ef4444; margin-bottom:1rem;">🚨 Detected Violations</h3>', unsafe_allow_html=True)

            for violation in fairness_results["violations"]:
                st.markdown(f"""
                <div style="background:rgba(239,68,68,0.1); border:2px solid #ef4444; border-radius:8px; padding:1rem; margin:0.5rem 0;">
                    <div style="color:#fca5a5; font-weight:700; font-size:1rem;">{violation['icon']} {violation['type']}</div>
                    <div style="color:#cbd5e1; font-size:0.9rem; margin-top:0.3rem;">{violation['detail']}</div>
                </div>
                """, unsafe_allow_html=True)

        # Warnings
        if fairness_results["warnings"]:
            st.markdown("---")
            st.markdown('<h3 style="color:#f59e0b; margin-bottom:1rem;">⚠️ Monitoring Alerts</h3>', unsafe_allow_html=True)

            for warning in fairness_results["warnings"]:
                st.markdown(f"""
                <div style="background:rgba(245,158,11,0.1); border:2px solid #f59e0b; border-radius:8px; padding:1rem; margin:0.5rem 0;">
                    <div style="color:#fcd34d; font-weight:700; font-size:1rem;">{warning['icon']} {warning['type']}</div>
                    <div style="color:#cbd5e1; font-size:0.9rem; margin-top:0.3rem;">{warning['detail']}</div>
                </div>
                """, unsafe_allow_html=True)

        # Fairness by group
        st.markdown("---")
        st.markdown('<h3 style="color:#93c5fd; margin-bottom:1rem;">📊 Approval Rates by Group</h3>', unsafe_allow_html=True)

        for group_type, group_data in fairness_results.get("metrics_by_group", {}).items():
            if "groups" in group_data:
                groups = group_data["groups"]
                fig_group = go.Figure()
                
                for g in groups:
                    # Support multiple subgroup schemas (group/bracket/name) to avoid runtime KeyError.
                    group_name = g.get("group") or g.get("bracket") or g.get("name") or "Unknown"
                    approval_rate = float(g.get("approval_rate", 0.0) or 0.0)
                    fig_group.add_trace(go.Bar(
                        x=[group_name],
                        y=[approval_rate * 100],
                        name=group_name,
                        text=[f"{approval_rate*100:.1f}%"],
                        textposition="outside",
                    ))
                
                fig_group.update_layout(
                    title=f"Approval Rate by {group_type.replace('_', ' ')}",
                    yaxis_title="Approval Rate (%)",
                    plot_bgcolor="#111827",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"),
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig_group, use_container_width=True)

        # Recommendations
        if fairness_results["recommendations"]:
            st.markdown("---")
            st.markdown('<h3 style="color:#93c5fd; margin-bottom:1rem;">💡 Recommendations</h3>', unsafe_allow_html=True)

            for i, rec in enumerate(fairness_results["recommendations"], 1):
                st.markdown(f"**{i}.** {rec}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 6: SYSTEM PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "🔧 System Pipeline":

    st.markdown("""
    <div style="margin-bottom:2rem;">
        <h1 style="color:#f1f5f9; font-size:2.2rem; font-weight:700; margin:0;">🔧 Decision Pipeline & Performance</h1>
        <p style="color:#64748b; margin-top:0.5rem; font-size:0.95rem;">System architecture, processing stages, and efficiency metrics</p>
    </div>
    """, unsafe_allow_html=True)

    tab_arch, tab_bench = st.tabs(["🏗️ Architecture", "⏱️ Performance Benchmarks"])

    with tab_arch:
        st.markdown('<h3 style="color:#93c5fd; margin-bottom:1rem;">📍 Processing Pipeline</h3>', unsafe_allow_html=True)

        stages = get_pipeline_stages()

        # Pipeline visualization
        col_pipeline = st.columns(6)
        for idx, stage in enumerate(stages):
            with col_pipeline[idx]:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg, {stage['color']}20, {stage['color']}10); border:2px solid {stage['color']}; border-radius:10px; padding:1rem; text-align:center;">
                    <div style="font-size:2rem; margin-bottom:0.5rem;">{stage['icon']}</div>
                    <div style="color:{stage['color']}; font-weight:700; font-size:0.9rem;">{stage['name']}</div>
                    <div style="color:#94a3b8; font-size:0.75rem; margin-top:0.5rem;">{stage['time_ms']}ms</div>
                </div>
                """, unsafe_allow_html=True)

        # Stage details
        st.markdown("---")
        st.markdown('<h4>📋 Stage Details:</h4>', unsafe_allow_html=True)

        for stage in stages:
            with st.expander(f"{stage['icon']} {stage['name']}"):
                col_detail1, col_detail2 = st.columns(2)
                with col_detail1:
                    st.write(f"**Description:** {stage['description']}")
                with col_detail2:
                    st.metric("Processing Time", f"{stage['time_ms']}ms")

    with tab_bench:
        st.markdown('<h3 style="color:#93c5fd; margin-bottom:1rem;">⚡ Performance vs Traditional</h3>', unsafe_allow_html=True)

        comparison = get_processing_comparison()

        col_trad, col_ai = st.columns(2)

        with col_trad:
            st.markdown(f"""
            <div style="background:#1c1c2e; border:2px solid #64748b; border-radius:10px; padding:1.5rem; text-align:center;">
                <div style="font-size:2rem;">🏦</div>
                <div style="color:#e2e8f0; font-weight:700; font-size:1.2rem; margin-top:0.5rem;">{comparison['traditional']['name']}</div>
                <div style="color:#f59e0b; font-size:2rem; font-weight:700; margin:1rem 0; font-family: 'JetBrains Mono';">{comparison['traditional']['time_display']}</div>
                <div style="color:#94a3b8; font-size:0.9rem;">Manual Steps: {comparison['traditional']['steps']}</div>
                <div style="color:#94a3b8; font-size:0.9rem;">Review: {comparison['traditional']['manual_review']}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_ai:
            st.markdown(f"""
            <div style="background:{comparison['your_system']['color']}15; border:2px solid {comparison['your_system']['color']}; border-radius:10px; padding:1.5rem; text-align:center;">
                <div style="font-size:2rem;">⚡</div>
                <div style="color:#e2e8f0; font-weight:700; font-size:1.2rem; margin-top:0.5rem;">{comparison['your_system']['name']}</div>
                <div style="color:{comparison['your_system']['color']}; font-size:2rem; font-weight:700; margin:1rem 0; font-family: 'JetBrains Mono';">{comparison['your_system']['time_display']}</div>
                <div style="color:#94a3b8; font-size:0.9rem;">Automated Steps: {comparison['your_system']['steps']}</div>
                <div style="color:#94a3b8; font-size:0.9rem;">Review: {comparison['your_system']['manual_review']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<h3 style="color:#93c5fd; margin-bottom:1rem;">📊 Efficiency Metrics</h3>', unsafe_allow_html=True)

        benchmark = get_performance_benchmark()

        metrics_df = pd.DataFrame(benchmark["metrics"])
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

        st.markdown("---")

        col_save1, col_save2 = st.columns(2)

        with col_save1:
            st.metric("Time Saved Per Application", f"{benchmark['time_saved_per_app']:.1f} hours", help="Reduced manual review time")

        with col_save2:
            st.metric("Annual Saving (100K apps)", f"{benchmark['annual_savings_100k_apps']:,} hours", help="Potential person-hours saved")

        # ROI calculation
        st.markdown("---")
        st.markdown('<h3 style="color:#93c5fd; margin-bottom:1rem;">💰 Return on Investment</h3>', unsafe_allow_html=True)

        col_roi1, col_roi2, col_roi3 = st.columns(3)

        with col_roi1:
            avg_salary_per_hour = 500  # ₹500/hour assumption
            annual_cost_saved = benchmark["annual_savings_100k_apps"] * avg_salary_per_hour
            st.metric("Annual Cost Savings", f"₹{annual_cost_saved/100000:.1f}L", help="Based on 100K app/year at ₹500/hr")

        with col_roi2:
            approval_rate_improvement = 0.08  # 8% more approvals due to consistency
            revenue_per_approval = 15000  # ₹15K avg revenue/loan
            additional_revenue = 100000 * approval_rate_improvement * revenue_per_approval
            st.metric("Additional Revenue", f"₹{additional_revenue/10000000:.1f}Cr", help="From improved consistency")

        with col_roi3:
            risk_reduction = 0.12  # 12% risk reduction
            avg_loss_per_bad_loan = 500000
            risk_mitigation = 100000 * 0.05 * avg_loss_per_bad_loan * risk_reduction  # 5% default rate
            st.metric("Risk Mitigation", f"₹{risk_mitigation/10000000:.1f}Cr", help="From fraud detection")
