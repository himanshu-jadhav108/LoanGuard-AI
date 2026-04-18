# 🚀 LoanGuard AI - Streamlit Cloud Deployment Guide

Complete step-by-step guide to deploy LoanGuard AI to **Streamlit Cloud** for free global hosting.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Pre-Deployment Setup](#pre-deployment-setup)
3. [Create Streamlit Cloud Account](#create-streamlit-cloud-account)
4. [Deploy to Streamlit Cloud](#deploy-to-streamlit-cloud)
5. [Configure App Settings](#configure-app-settings)
6. [Environment Variables & Secrets](#environment-variables--secrets)
7. [Monitor & Troubleshoot](#monitor--troubleshoot)
8. [Update & Redeploy](#update--redeploy)
9. [Performance Optimization](#performance-optimization)
10. [Production Best Practices](#production-best-practices)

---

## 📦 Prerequisites

Before deploying, ensure you have:

- ✅ **GitHub account** (required for Streamlit Cloud deployment)
- ✅ **Git installed** locally
- ✅ **Python 3.8+** installed
- ✅ **LoanGuard AI repository** pushed to GitHub (public or private)
- ✅ **requirements.txt** file in root directory (already included)
- ✅ **Model artifacts** committed to repo or generated on first run

**GitHub Repository Setup:**
- Repository: https://github.com/himanshu-jadhav108/LoanGuard-AI
- Branch: `main` (or your deployment branch)
- Make repository **public** (recommended) or private (requires authentication)

---

## 🔧 Pre-Deployment Setup

### Step 1: Verify Project Structure

Ensure your GitHub repository has this structure (✅ Already done):

```
LoanGuard-AI/
├── app.py                      # Root Streamlit entry point
├── requirements.txt            # Python dependencies ⭐ CRITICAL
├── README.md                   # Documentation
├── DEPLOY.md                   # This file
├── .gitignore                  # Excluded files
├── pyproject.toml              # Project config
├── src/loanguard/
│   ├── apps/
│   │   └── app.py             # Main application
│   └── core/
│       ├── model.py
│       ├── preprocessing.py
│       ├── risk.py
│       ├── fraud_detection.py
│       ├── explain.py
│       ├── fairness_enhanced.py
│       ├── analytics.py
│       ├── utils.py
│       └── __init__.py
├── scripts/
│   ├── generate_data.py       # Data generation
│   └── train_model.py         # Model training
└── data/
    └── loan_dataset.csv       # Training data
```

### Step 2: Update requirements.txt

**Verify dependencies are correct:**

```bash
cd d:\Projects\loan_system
cat requirements.txt
```

**Expected output:**
```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
graphviz>=0.20.0
```

**Add if missing:**
```bash
pip install -r requirements.txt
```

### Step 3: Create .streamlit Directory (Optional but Recommended)

Create configuration file for Streamlit Cloud:

```bash
mkdir .streamlit
```

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#0a0e1a"
secondaryBackgroundColor = "#111827"
textColor = "#e2e8f0"

[client]
showErrorDetails = true

[logger]
level = "info"

[client.toolbarMode]
mode = "minimal"

[server]
port = 8501
headless = true
runOnSave = true
```

### Step 4: Create .streamlit/secrets.toml (For Secrets)

Create `.streamlit/secrets.toml` **locally** (DO NOT push to GitHub):

```toml
# Database credentials (if using external DB in future)
# DATABASE_URL = "postgresql://user:password@localhost/loanguard"

# API Keys (if using external services)
# API_KEY = "your-api-key-here"

# Model configuration
MODEL_VERSION = "1.0"
FRAUD_THRESHOLD = 0.50
HIGH_RISK_THRESHOLD = 35
```

**Add to .gitignore:**
```bash
.streamlit/secrets.toml
```

### Step 5: Test Locally Before Deploying

**Run the app locally to ensure it works:**

```bash
streamlit run app.py
```

**Access at:** http://localhost:8501

**Test all pages:**
- ✅ Loan Application (real-time prediction)
- ✅ Scenario Simulator (what-if analysis)
- ✅ Batch Processing (CSV upload)
- ✅ Analytics Dashboard (KPIs)
- ✅ Fairness & Bias (compliance)
- ✅ System Pipeline (architecture)

**Fix any local errors before deploying.**

### Step 6: Commit All Changes to GitHub

```bash
git add .
git commit -m "chore: prepare for Streamlit Cloud deployment (add config, secrets)"
git push origin main
```

---

## 🔐 Create Streamlit Cloud Account

### Step 1: Sign Up for Streamlit Cloud

1. Go to: https://share.streamlit.io
2. Click **"Sign Up"** or **"Sign In"**
3. Choose: **"Sign in with GitHub"**
4. Authorize Streamlit to access your GitHub account
5. Accept the terms and complete sign-up

### Step 2: Authorize GitHub Access

When prompted, authorize Streamlit Cloud to:
- Read public/private repositories
- Deploy from GitHub
- Monitor repository changes

**Grant all requested permissions** for seamless deployment.

---

## 🚀 Deploy to Streamlit Cloud

### Step 1: Start New Deployment

1. Visit: https://share.streamlit.io
2. Click **"New app"** button (top right)
3. You'll see: **"Streamlit Community Cloud"** dialog

### Step 2: Select Repository

In the deployment dialog:

**Repository:** 
- Select: `himanshu-jadhav108/LoanGuard-AI`
- (Your GitHub repos will be listed)

**Branch:**
- Select: `main`
- (or your deployment branch)

**Main file path:**
- Enter: `app.py`
- ⚠️ **CRITICAL:** Must be the root `app.py`, not `src/loanguard/apps/app.py`

### Step 3: Configure App Settings

**Advanced Settings (Optional):**
- **Python version:** Python 3.11+ (default)
- **Secrets:** Will add after deployment
- **Resources:** Default (sufficient for demo/MVP)

### Step 4: Deploy

1. Click **"Deploy"** button
2. **Wait for deployment** (2-5 minutes first time)
3. Watch the **build log** for errors:
   ```
   📦 Installing dependencies...
   🔨 Building app...
   ✅ Deployment successful
   ```

### Step 5: Access Your App

After successful deployment, you'll see:

```
✨ Your app is live at:
https://loanguard-ai.streamlit.app
```

**Share this URL with others!**

---

## ⚙️ Configure App Settings

### Step 1: Access App Settings

1. On your app page, click **⋮ (three dots)** → **Settings**
2. Or visit: https://share.streamlit.io/himanshu-jadhav108/LoanGuard-AI

### Step 2: Configure General Settings

| Setting | Recommendation |
|---------|--------------|
| **Visibility** | Public (for portfolio) or Private (if using sensitive data) |
| **Run on Edit** | Enable (auto-refresh on push) |
| **Run on Deploy** | Enable (clear cache on new deployment) |

### Step 3: Set Client Settings

Under **Secrets**:
1. Click **"Secrets"** in settings
2. Add environment variables in TOML format:

```toml
MODEL_VERSION = "1.0"
FRAUD_THRESHOLD = 0.50
HIGH_RISK_THRESHOLD = 35
DEBUG_MODE = "false"
```

---

## 🔒 Environment Variables & Secrets

### Step 1: Add Secrets via Streamlit Cloud UI

1. Go to **App settings** → **Secrets**
2. Add sensitive data:

```toml
# Email configuration (future feature)
EMAIL_SENDER = "noreply@loanguard.app"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Database (if connecting to external DB)
# DATABASE_URL = "postgresql://user:pass@db.example.com/loanguard"

# API Keys
# API_KEY_BANK_INTEGRATION = "your-api-key"

# Model thresholds
FRAUD_CRITICAL_THRESHOLD = 0.75
FRAUD_HIGH_THRESHOLD = 0.50
```

### Step 2: Access Secrets in Code

**Already configured in `src/loanguard/apps/app.py`:**

```python
import streamlit as st

# Access secrets
fraud_threshold = st.secrets.get("FRAUD_THRESHOLD", 0.50)
high_risk_threshold = st.secrets.get("HIGH_RISK_THRESHOLD", 35)
```

### Step 3: Local Testing with Secrets

**Create `.streamlit/secrets.toml` locally:**

```toml
FRAUD_THRESHOLD = 0.50
HIGH_RISK_THRESHOLD = 35
```

**Test locally:**
```bash
streamlit run app.py
```

---

## 📊 Monitor & Troubleshoot

### Step 1: View Deployment Logs

1. Go to your app page
2. Click **⋮ (three dots)** → **View logs**
3. Check for errors during runtime

**Common errors:**

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'loanguard'` | Ensure `src/loanguard/__init__.py` exists and is committed |
| `FileNotFoundError: data/loan_dataset.csv` | Commit data files or generate on app start |
| `sklearn.exceptions.NotFittedError` | Train models or use pre-trained `.pkl` files |
| `MemoryError` | Cache large computations with `@st.cache_resource` |

### Step 2: View Deployment History

1. Settings → **Rerun** (redeploy specific version)
2. Settings → **Deployment history** (see all builds)

### Step 3: Check App Health

**Monitor from the app itself:**
- Check **System Pipeline** page for processing times
- Verify **Analytics Dashboard** loads correctly
- Test all form inputs without errors

### Step 4: Enable Debug Mode

Add to `.streamlit/config.toml`:

```toml
[logger]
level = "debug"

[client]
showErrorDetails = true
```

Then redeploy or commit and push.

---

## 🔄 Update & Redeploy

### Step 1: Make Changes Locally

**Example: Update fraud threshold**

Edit `src/loanguard/core/fraud_detection.py`:

```python
CRITICAL_FRAUD_THRESHOLD = 0.75  # Changed from 0.50
```

### Step 2: Test Locally

```bash
streamlit run app.py
```

Verify the changes work.

### Step 3: Commit and Push to GitHub

```bash
git add src/loanguard/core/fraud_detection.py
git commit -m "feat: increase fraud detection threshold to 0.75 for stricter compliance"
git push origin main
```

### Step 4: Automatic Redeployment

✨ **Streamlit Cloud automatically redeploys** when you push to `main`:

1. GitHub push triggers deployment
2. Streamlit detects changes
3. App rebuilds (2-5 minutes)
4. New version goes live automatically

**Monitor deployment:** Settings → Deployment history

### Step 5: Manual Redeployment (If Needed)

If auto-deploy doesn't trigger:

1. Go to app settings
2. Click **Rerun** button
3. Or delete and redeploy the app

---

## ⚡ Performance Optimization

### Step 1: Use Caching Decorators

**Cache expensive computations:**

```python
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_training_data():
    return pd.read_csv("data/loan_dataset.csv")

@st.cache_resource  # Cache model (persistent)
def load_model():
    return joblib.load("models/logistic_regression.pkl")
```

**Already implemented in:**
- `model.py` - Model loading
- `utils.py` - Data loading

### Step 2: Optimize Data Loading

**Load data once, reuse throughout session:**

```python
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = load_training_data()

df = st.session_state.data_cache
```

### Step 3: Limit Historical Data

**Streamlit Cloud has memory limits (1GB)**

If storing historical applications:

```python
# Keep only last 10,000 applications
if len(application_log) > 10000:
    application_log = application_log.tail(10000)
```

### Step 4: Compress Model Files

**Use pickle protocol 5+ for smaller file sizes:**

```python
import joblib

# Save with compression
joblib.dump(model, "model.pkl", compress=3)

# Load
model = joblib.load("model.pkl")
```

### Step 5: Lazy Load Components

**Load heavy visualizations only when needed:**

```python
if st.checkbox("Show Advanced Analytics"):
    generate_detailed_charts()  # Only if user requests
```

---

## 🏆 Production Best Practices

### Best Practice 1: Use Environment-Specific Configs

**Create `config.py`:**

```python
import os

ENV = os.getenv("ENVIRONMENT", "development")

if ENV == "production":
    MODEL_PATH = "models/logistic_regression.pkl"
    DEBUG = False
    CACHE_TTL = 3600
else:
    MODEL_PATH = "models/logistic_regression.pkl"
    DEBUG = True
    CACHE_TTL = 60
```

### Best Practice 2: Add Error Handling & Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    prediction = predict(applicant_data)
    logger.info(f"Prediction successful for {applicant_id}")
except Exception as e:
    logger.error(f"Prediction failed: {str(e)}")
    st.error("Processing error. Please try again.")
```

### Best Practice 3: Implement Health Checks

**Add status endpoint (future REST API):**

```python
def health_check():
    try:
        model = load_model()
        scaler = load_scaler()
        return {"status": "healthy", "models": "loaded"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### Best Practice 4: Monitor Resource Usage

**Display app performance metrics:**

```python
import psutil

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Memory Usage", f"{psutil.virtual_memory().percent}%")
with col2:
    st.metric("CPU Usage", f"{psutil.cpu_percent()}%")
with col3:
    st.metric("Session Time", f"{time.time() - st.session_state.start_time:.0f}s")
```

### Best Practice 5: Implement Rate Limiting

**Prevent abuse with Streamlit Cloud limits:**

```python
@st.cache_resource
def get_usage_tracker():
    return {}

def check_rate_limit(user_id, max_requests=100):
    tracker = get_usage_tracker()
    today = str(datetime.date.today())
    
    key = f"{user_id}_{today}"
    tracker[key] = tracker.get(key, 0) + 1
    
    if tracker[key] > max_requests:
        st.error("Rate limit exceeded. Try again tomorrow.")
        return False
    return True
```

### Best Practice 6: Secure Sensitive Operations

**Don't expose secrets in logs:**

```python
# ❌ WRONG
st.write(f"API Key: {api_key}")

# ✅ CORRECT
st.write("Connecting to external service...")
# API call happens in backend without logging credentials
```

### Best Practice 7: Document for Users

**Add deployment info to README:**

```markdown
## 🌐 Live Demo

### Access the App
- **URL:** https://loanguard-ai.streamlit.app
- **Status:** ✅ Live
- **Last Updated:** 2026-04-18

### System Status
- ML Model: Logistic Regression (v1.0, 92.25% accuracy)
- Fairness Checks: Enabled (4/5 rule, ADEA)
- Fraud Detection: Active (5-rule engine)
```

---

## 🔗 Useful Links

| Resource | URL |
|----------|-----|
| **Your Live App** | https://loanguard-ai.streamlit.app |
| **Streamlit Cloud Dashboard** | https://share.streamlit.io |
| **Streamlit Documentation** | https://docs.streamlit.io |
| **GitHub Repository** | https://github.com/himanshu-jadhav108/LoanGuard-AI |
| **Streamlit Community Forum** | https://discuss.streamlit.io |
| **Report Issues** | https://github.com/streamlit/streamlit/issues |

---

## 📞 Troubleshooting Guide

### Issue: "No such file or directory: data/loan_dataset.csv"

**Solution:**
```bash
# Generate data on first run
python scripts/generate_data.py
git add data/
git commit -m "feat: add generated dataset"
git push
```

**Or auto-generate on app startup** in `app.py`:
```python
if not os.path.exists("data/loan_dataset.csv"):
    os.system("python scripts/generate_data.py")
```

### Issue: "ModuleNotFoundError: loanguard"

**Solution:** Ensure Python path includes `src/`:

At the top of `app.py`:
```python
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
```

### Issue: App is very slow

**Solution:** Enable caching:
```python
@st.cache_data
def expensive_function():
    return process_data()
```

### Issue: Deployment fails with MemoryError

**Solution:** Use `st.cache_resource` for models:
```python
@st.cache_resource
def load_model():
    return joblib.load("models/logistic_regression.pkl")
```

### Issue: Changes not appearing after push

**Solution:**
1. Go to app settings
2. Click "Rerun" to force redeploy
3. Or delete and redeploy the app

---

## 🎉 Success Indicators

After deployment, verify:

- ✅ App is accessible at `https://loanguard-ai.streamlit.app`
- ✅ All 6 pages load without errors
- ✅ Forms accept input and process predictions
- ✅ Charts and visualizations render correctly
- ✅ Fairness & Bias page shows compliance status
- ✅ Batch processing accepts CSV uploads
- ✅ No 503 Service Unavailable errors
- ✅ Response time is <2 seconds per page

---

## 🚀 Next Steps

After successful deployment:

1. **Share with Friends & Family**
   - Send live app link
   - Gather feedback
   - Collect user insights

2. **Prepare Portfolio**
   - Add to GitHub profile
   - Link to portfolio website
   - Share on LinkedIn

3. **Improve Based on Feedback**
   - Fix bugs
   - Add requested features
   - Optimize performance

4. **Plan v2.0 Enhancements**
   - REST API wrapper
   - Database integration
   - Advanced ML models
   - Real-time monitoring dashboard

---

**🎊 Congratulations! Your LoanGuard AI app is now live on Streamlit Cloud!**

For support, visit:
- 📧 Email: himanshu@loanguard.example
- 💬 GitHub Issues: https://github.com/himanshu-jadhav108/LoanGuard-AI/issues
- 🐙 GitHub Discussions: https://github.com/himanshu-jadhav108/LoanGuard-AI/discussions
