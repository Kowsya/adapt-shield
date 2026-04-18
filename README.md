# 🛡️ ADAPT-SHIELD — Complete Project Guide

> **MLOps Network Intrusion Detection System**  
> CIC-IDS2017 Dataset · 2.3M Records · 8 Attack Classes · Random Forest + XGBoost

---

## 📁 Project Structure

```
adapt-shield/
├── app/
│   ├── main.py              ← FastAPI Security Layer (port 8000)
│   ├── schemas.py           ← Pydantic request/response schemas   ← COPY FROM FILES
│   └── website.py           ← Simulated protected website (port 9000)
├── data/
│   └── attackdata/          ← Put your 8 .parquet files here
├── docker/
│   ├── docker-compose.yml   ← Runs both containers together
│   ├── Dockerfile.security  ← ML model container
│   └── Dockerfile.website   ← Website container
├── models/                  ← Auto-created by training
│   ├── model.pkl            ← Best model (used by API)
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── feature_names.pkl
├── monitoring/
│   ├── drift_detection.py   ← COPY FROM FILES
│   └── drift_log.json
├── notebooks/
│   └── EDA.ipynb
├── reports/                 ← Auto-created
├── src/
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py          ← COPY FROM FILES
│   └── predict.py
├── tests/
│   └── test_pipeline.py
├── .github/
│   └── workflows/
│       └── ci-cd.yml        ← COPY FROM FILES
├── .gitignore
├── requirements.txt         ← COPY FROM FILES (fixed for Python 3.13)
└── README.md
```

---

## ⚙️ FILES TO UPDATE/ADD

Copy these files from the provided code into your project:

| File | Location | Action |
|------|----------|--------|
| `requirements.txt` | root | **Replace** existing file |
| `schemas.py` | `app/` | **Add** (empty now) |
| `evaluate.py` | `src/` | **Add** (empty now) |
| `drift_detection.py` | `monitoring/` | **Replace** existing file |
| `ci-cd.yml` | `.github/workflows/` | **Replace** existing file |

---

## 🚀 STEP-BY-STEP: Running the Project from Scratch

Open **VS Code terminal** (Ctrl + `` ` ``). All commands run from the project root.

---

### STEP 1 — Activate Virtual Environment

```powershell
# Always do this first every time you open VS Code
venv\Scripts\activate
```

You'll see `(venv)` appear at the start of your terminal line.

---

### STEP 2 — Install Dependencies

```powershell
pip install -r requirements.txt
```

> ✅ Already done if you see your packages installed. Skip if done.

---

### STEP 3 — Run Data Ingestion

```powershell
python src/data_ingestion.py
```

**Expected output:**
```
📂 Loading Benign-Monday-no-metadata.parquet...
   Shape: (458831, 78) | Labels: ['Benign']
...
✅ Combined shape: (2313810, 78)
✅ Saved! File size: 261.3 MB
```

---

### STEP 4 — Validate Data

```powershell
python src/data_validation.py
```

**Expected output:**
```
✅ Shape: 2,313,810 rows × 78 columns
✅ No missing values found!
⚠️  Found 82,004 duplicate rows (will be removed in training)
✅ Label column found with 8 classes
```

---

### STEP 5 — Start MLflow UI (open in NEW terminal tab!)

```powershell
# Open a NEW terminal tab (Ctrl+Shift+` in VS Code)
venv\Scripts\activate
mlflow ui
```

**Expected output:**
```
[MLflow] Uvicorn running on http://127.0.0.1:5000
```

> 🌐 Open browser: **http://127.0.0.1:5000**  
> Keep this terminal open the whole time.

---

### STEP 6 — Train Models

```powershell
# Go back to first terminal tab
python src/train.py
```

> ⏱️ Takes 3–10 minutes. Uses 30% of data by default for speed.  
> Change `sample_frac=1.0` in train.py for full training.

**Expected output:**
```
🌲 Training Random Forest...
✅ Random Forest saved → models/random_forest.pkl
   Accuracy: 0.9978 | Recall: 0.9923 | F1: 0.9934

⚡ Training XGBoost...
✅ XGBoost saved → models/xgboost.pkl

🏆 Winner: XGBoost (or RandomForest)
✅ Best model saved as → models/model.pkl
```

> 🌐 Refresh MLflow at http://127.0.0.1:5000 — you'll see both runs!

---

### STEP 7 — Evaluate + Hyperparameter Tuning

```powershell
python src/evaluate.py
```

**Expected output:**
```
  accuracy             0.9978  ████████████████████
  recall_macro         0.9923  ███████████████████
  f1_macro             0.9934  ███████████████████

🔍 Running GridSearchCV (3-fold CV)...
🏆 Best Parameters: {'max_depth': 20, 'min_samples_split': 5, 'n_estimators': 200}
   Best CV F1 (macro): 0.9941
   Logged to MLflow ✅
```

---

### STEP 8 — Start FastAPI Security Layer

```powershell
# In a new terminal tab
venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
```

**Expected output:**
```
🚀 ADAPT-SHIELD Security API starting...
✅ Model ready for inference!
INFO: Uvicorn running on http://0.0.0.0:8000
```

> 🌐 Open browser: **http://localhost:8000/docs** (Swagger UI — interactive!)

---

### STEP 9 — Test the API (in docs or curl)

**Test benign traffic (should ALLOW):**
```powershell
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d "{\"Protocol\": 6, \"Flow_Duration\": 100, \"SYN_Flag_Count\": 1}"
```

**Test DDoS attack (should BLOCK):**
```powershell
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d "{\"SYN_Flag_Count\": 1000, \"Flow_Packets_s\": 50000, \"Protocol\": 6}"
```

**Expected response:**
```json
{
  "prediction": "DDoS",
  "is_malicious": true,
  "confidence": 0.9921,
  "action": "BLOCK",
  "processing_time_ms": 12.4
}
```

---

### STEP 10 — Run Monitoring & Drift Detection

```powershell
python monitoring/drift_detection.py
```

**Expected output:**
```
📈 API HEALTH STATUS
  Status:        healthy
  Model loaded:  True
  Block rate:    15.0%

🔍 DRIFT DETECTION RESULTS (KS Test)
   Drifted features : 5/66

✅ Evidently report saved → reports/drift_report.html
```

> 🌐 Open `reports/drift_report.html` in your browser to see the full drift report!

---

### STEP 11 — Run Docker (Both Containers Together)

```powershell
# Make sure Docker Desktop is open first!
# Then in a terminal (from project root):

cd docker
docker-compose up --build
```

**Expected output:**
```
✅ adapt-shield-security  | ✅ Model ready for inference!
✅ adapt-shield-website   | Started website on port 9000
```

> 🌐 Security Layer: **http://localhost:8000/docs**  
> 🌐 Protected Website: **http://localhost:9000**

**Stop Docker:**
```powershell
docker-compose down
```

---

### STEP 12 — Push to GitHub

```powershell
git add .
git commit -m "feat: Add evaluate.py, schemas.py, updated monitoring"
git push origin main
```

> 🌐 Go to your GitHub repo → **Actions tab** → watch CI/CD pipeline run!  
> You'll see: ✅ Tests passed → ✅ Docker build successful

---

## 🖥️ PRESENTATION DAY — What to Show (in Order)

> Here is the exact sequence to show your professor. Open everything first, then demo.

### Pre-Presentation Setup (15 min before)

```powershell
# Terminal 1: Activate venv
venv\Scripts\activate

# Terminal 2 (new tab): Start MLflow
venv\Scripts\activate
mlflow ui

# Terminal 3 (new tab): Start API
venv\Scripts\activate
uvicorn app.main:app --reload --port 8000

# Terminal 4 (new tab): Start Docker
docker-compose -f docker/docker-compose.yml up
```

**Open these browser tabs:**
1. http://127.0.0.1:5000 — MLflow UI
2. http://localhost:8000/docs — FastAPI Swagger
3. http://localhost:9000 — Protected Website
4. Your GitHub repo → Actions tab
5. `reports/drift_report.html` (open as file in browser)

---

### What to Demo (Script)

#### 1. Show MLflow (2 min)
> "Here is MLflow — our experiment tracking tool. You can see both our Random Forest and XGBoost runs. Each run logged parameters, metrics, and the trained model artifact. We can compare them here — XGBoost had higher F1 score so it became our best model."

- Click the experiment → show both runs
- Click one run → show accuracy, recall, F1, confusion matrix image

#### 2. Show FastAPI (3 min)
> "This is our security layer API at port 8000. It receives network traffic and decides ALLOW or BLOCK."

- Go to http://localhost:8000/docs
- Click `/predict` → Try it out → Paste DDoS test:
```json
{"SYN_Flag_Count": 1000, "Flow_Packets_s": 50000, "Protocol": 6}
```
- Show the BLOCK response
- Then test benign:
```json
{"Protocol": 6, "Flow_Duration": 100}
```
- Show the ALLOW response + forward to website

#### 3. Show Website (1 min)
> "Traffic classified as benign is forwarded to the main website at port 9000."
- Open http://localhost:9000
- Show the protected page

#### 4. Show Docker (2 min)
> "Both services run in Docker containers connected via an internal network — exactly like microservices in production."
- Show terminal with `docker-compose up` running
- Show `docker ps` in another terminal

#### 5. Show GitHub Actions (2 min)
> "We have CI/CD — every time we push code, GitHub automatically runs tests and builds our Docker image."
- Go to GitHub repo → Actions → show green workflow run

#### 6. Show Monitoring (2 min)
> "We use Evidently AI for data drift detection. If incoming traffic differs significantly from training data, the system alerts us to retrain."
- Open `reports/drift_report.html`
- Show drift report with colored dashboard

#### 7. Show Hyperparameter Tuning in MLflow (1 min)
> "We also ran GridSearchCV for hyperparameter tuning and logged the best parameters to MLflow."
- Show the HyperparamTuning run in MLflow

---

## ☁️ AWS Free Tier Deployment

> Only do this if you want to show the live public URL.  
> Your free tier gives **750 hours/month** of t2.micro — enough.

### Option A: Manual EC2 Deployment

```powershell
# 1. Install AWS CLI: https://aws.amazon.com/cli/
# 2. Configure AWS CLI
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-1), Output (json)

# 3. Create ECR repository
aws ecr create-repository --repository-name adapt-shield --region us-east-1

# 4. Login Docker to ECR (replace 123456789012 with your AWS Account ID)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

# 5. Build and tag image
docker build -f docker/Dockerfile.security -t adapt-shield-security .
docker tag adapt-shield-security:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/adapt-shield:latest

# 6. Push to ECR
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/adapt-shield:latest
```

**Then on EC2:**
```bash
# SSH into your EC2 instance
ssh -i "adapt-shield-key.pem" ec2-user@YOUR-EC2-PUBLIC-IP

# Install Docker on EC2
sudo yum update -y
sudo yum install docker -y
sudo systemctl start docker
sudo usermod -aG docker ec2-user

# Pull and run
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
docker pull 123456789012.dkr.ecr.us-east-1.amazonaws.com/adapt-shield:latest
docker run -d -p 8000:8000 123456789012.dkr.ecr.us-east-1.amazonaws.com/adapt-shield:latest

# Access at: http://YOUR-EC2-PUBLIC-IP:8000/docs
```

> ⚠️ **IMPORTANT:** Add inbound rule in EC2 Security Group → Port 8000 → Source: 0.0.0.0/0

---

### Option B: AWS Free Alternative — Render.com (EASIER!)

> No credit card · Auto-deploys from GitHub · Free HTTPS URL

1. Go to **render.com** → Sign Up with GitHub
2. New → Web Service → Connect your repo
3. Settings:
   - **Runtime:** Python 3.11
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Instance:** Free
4. Click **Deploy**
5. Get URL like: `https://adapt-shield.onrender.com/docs`

---

### Option C: AWS CloudWatch for Monitoring (Free Tier)

> No cost — CloudWatch gives you dashboards for EC2 metrics.

```bash
# On your EC2 instance, install CloudWatch agent
sudo yum install amazon-cloudwatch-agent -y

# Create config file
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-config-wizard

# Start agent
sudo systemctl start amazon-cloudwatch-agent
```

> Go to AWS Console → CloudWatch → Dashboards  
> Add widgets for: CPU utilization, network in/out, memory usage

---

## 🐛 TROUBLESHOOTING

### Problem: `mlflow ui` gives port error
```powershell
# MLflow is already running. Kill it:
taskkill /F /IM python.exe
# Then restart: mlflow ui
```

### Problem: `uvicorn app.main:app` — ModuleNotFoundError
```powershell
# Run from the project ROOT, not from src/
cd C:\Users\Student\Desktop\adapt-shield
uvicorn app.main:app --reload --port 8000
```

### Problem: Model not found (503 error from API)
```powershell
# Models haven't been trained yet. Run:
python src/train.py
# Models saved to models/ folder — then restart uvicorn
```

### Problem: Docker build fails
```powershell
# Make sure Docker Desktop is running first (check taskbar)
# Then:
docker-compose -f docker/docker-compose.yml down
docker-compose -f docker/docker-compose.yml up --build
```

### Problem: GitHub Actions failing
- Go to your GitHub repo → Settings → Secrets and Variables → Actions
- Add secrets: `DOCKER_USERNAME` and `DOCKER_PASSWORD` (Docker Hub credentials)
- Re-push to trigger the pipeline

### Problem: `imbalanced-learn` import error
```powershell
pip install imbalanced-learn>=0.12.0 --upgrade
```

---

## 📊 Model Results (Your Actual Training)

Based on your training run (from the terminal logs):

| Metric | Random Forest | XGBoost |
|--------|--------------|---------|
| Accuracy | ~99.7% | ~99.8% |
| Recall (macro) | ~99.2% | ~99.3% |
| F1 (macro) | ~99.3% | ~99.4% |
| Best? | ❌ | ✅ |

**Classes detected:**
- BENIGN (85.5% of traffic) — ALLOWED
- DoS (8.4%) — BLOCKED
- DDoS (5.5%) — BLOCKED
- BruteForce (0.4%) — BLOCKED
- WebAttack (0.1%) — BLOCKED
- PortScan (0.1%) — BLOCKED
- Botnet (0.1%) — BLOCKED
- Infiltration (0.01%) — BLOCKED

---

## 🔄 Complete Run Order Summary

```
1.  venv\Scripts\activate          ← Always first
2.  python src/data_ingestion.py   ← Merge 8 parquet files
3.  python src/data_validation.py  ← Check data quality
4.  mlflow ui                      ← Start tracking UI (new tab)
5.  python src/train.py            ← Train RF + XGBoost (3-10 min)
6.  python src/evaluate.py         ← Evaluate + hyperparameter tune
7.  uvicorn app.main:app --reload --port 8000   ← Start API (new tab)
8.  python monitoring/drift_detection.py        ← Run monitoring
9.  docker-compose -f docker/docker-compose.yml up --build  ← Docker
10. git add . && git commit -m "update" && git push origin main
```

---

## 📌 Important URLs

| Service | URL | What you see |
|---------|-----|-------------|
| MLflow UI | http://127.0.0.1:5000 | Experiment runs, metrics, models |
| FastAPI Docs | http://localhost:8000/docs | Interactive API testing |
| FastAPI Health | http://localhost:8000/health | API status, stats |
| Protected Website | http://localhost:9000 | Simulated website |
| Monitoring Report | Open `reports/drift_report.html` | Drift dashboard |
| GitHub Actions | github.com/YOUR_REPO/actions | CI/CD pipeline |
| AWS EC2 (if deployed) | http://YOUR_IP:8000/docs | Live public API |

---

*ADAPT-SHIELD MLOps · CIC-IDS2017 · Kowsya Mutkundu & Muskan Khatoon*