# Customer Churn Prediction

A production-grade machine learning system that predicts which telecom customers are likely to cancel their service — and explains exactly why.

Built end-to-end: raw data ingestion → feature engineering → model training with hyperparameter tuning → REST API → Docker → live on AWS.

**Dataset:** Cell2Cell (Duke University / Teradata Center for CRM) — 51,047 customers, 58 features  
**Best Model:** LightGBM · ROC-AUC 0.683 · deployed on AWS EC2  
**Live API:** `http://52.71.35.176:8000/docs`

---

## What This Project Does

A telecom company loses revenue every time a customer churns. This system:

1. **Predicts** which customers are at risk of leaving (with a probability score)
2. **Explains** the top reasons driving that risk (powered by SHAP)
3. **Categorises** each customer into Low / Medium / High risk tiers
4. **Serves predictions** via a production REST API, live on AWS

---

## How It Works

```
Cell2Cell Dataset (51k customers, 58 features)
        │
        ▼
Data Cleaning & Feature Engineering
  • Median imputation for missing numerics
  • Outlier capping (IQR-based)
  • 8 engineered features (RevenuePerMinute, DropRate, ServiceStressIndex, ...)
        │
        ▼
Preprocessing Pipeline (scikit-learn)
  • StandardScaler for numeric columns
  • OrdinalEncoder for categorical columns
        │
        ▼
SMOTE — balances 71/29 class imbalance on training data only
        │
        ▼
Optuna Hyperparameter Tuning (50 trials per model)
  + MLflow Experiment Tracking
        │
        ├── Logistic Regression (baseline)
        ├── Random Forest
        ├── Extra Trees
        ├── XGBoost
        ├── LightGBM          ← best on this dataset
        ├── CatBoost
        └── Stacking Ensemble (top-3 base models + LR meta-learner)
        │
        ▼
Best Model Auto-Selected by ROC-AUC
        │
        ▼
FastAPI Production API
  POST /predict          — single prediction (pre-processed)
  POST /predict/raw      — single prediction (raw CSV columns)
  POST /predict/batch    — batch up to 1,000 rows
  GET  /health | /docs   — health check + interactive docs
        │
        ▼
Docker Image → AWS ECR → AWS EC2
  (deployed automatically via GitHub Actions on every push to main)
```

---

## Model Results

Seven models were trained and compared. The best by validation ROC-AUC is automatically selected and deployed.

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.614 | 0.387 | 0.92 | 0.31 | 0.46 |
| Random Forest | 0.660 | 0.429 | 0.76 | 0.36 | 0.49 |
| Extra Trees | 0.645 | 0.404 | 0.81 | 0.35 | 0.49 |
| CatBoost | 0.679 | 0.454 | 0.89 | 0.34 | 0.49 |
| XGBoost | 0.682 | 0.462 | 0.87 | 0.34 | 0.48 |
| Stacking Ensemble | 0.680 | 0.463 | 0.87 | 0.35 | 0.49 |
| **LightGBM** | **0.683** | **0.466** | **0.88** | **0.34** | **0.48** |

Metrics evaluated at threshold = 0.35, tuned to maximise expected annual retention savings rather than raw accuracy.

---

## Key Feature Engineering

| Feature | What it captures |
|---|---|
| `RevenuePerMinute` | Pricing efficiency — low revenue per minute signals churn risk |
| `DropRate` | Call quality — high drop rate drives dissatisfaction |
| `ServiceStressIndex` | `CustomerCareCalls × DroppedCalls` — combined friction score |
| `IsNewCustomer` | Customers in their first 3 months churn at the highest rate |
| `equipment_age_tier` | Older devices correlate with higher churn |
| `retention_urgency` | Previously contacted retention team but stayed — high risk flag |
| `call_quality_rate` | Ratio of successful calls to total attempted |
| `is_low_usage` | Low engagement signals disengagement before churn |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Package manager | `uv` |
| Data processing | pandas, pyarrow |
| ML models | XGBoost, LightGBM, CatBoost, scikit-learn, imbalanced-learn |
| Hyperparameter tuning | Optuna |
| Experiment tracking | MLflow |
| Explainability | SHAP |
| API | FastAPI + Uvicorn |
| Observability | Prometheus metrics + Loguru logging |
| Containerisation | Docker (multi-stage build, non-root user) |
| CI/CD | GitHub Actions |
| Cloud | AWS ECR (image registry) + EC2 (inference server) |
| Infrastructure as Code | Terraform |
| Config management | Pydantic Settings |

---

## Project Structure

```
customer-churn-prediction/
│
├── src/churn/
│   ├── data/
│   │   ├── ingest.py          # Downloads dataset from Kaggle or local copy
│   │   └── preprocess.py      # Full sklearn pipeline — impute, scale, encode
│   ├── features/
│   │   └── engineer.py        # Feature creation, SMOTE, interaction terms
│   ├── models/
│   │   ├── train.py           # Trains all 7 models with Optuna + MLflow
│   │   └── evaluate.py        # SHAP plots, ROC/PR curves, confusion matrix
│   └── api/
│       ├── main.py            # FastAPI app — routes, middleware, model loading
│       └── schemas.py         # Pydantic request/response schemas
│
├── configs/
│   └── settings.py            # Single source of truth for all configuration
│
├── infrastructure/
│   ├── main.tf                # AWS resources — ECR, EC2, IAM, security groups
│   ├── variables.tf           # Configurable inputs
│   └── outputs.tf             # Prints GitHub secrets after terraform apply
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   └── 02_modeling.ipynb      # Model comparison, threshold analysis, SHAP plots
│
├── scripts/
│   └── predict.py             # CLI batch scoring tool
│
├── .github/workflows/
│   └── ci_cd.yml              # Build → Push to ECR → Deploy to EC2
│
├── Dockerfile                 # Multi-stage build — builder + slim runtime
├── docker-compose.yml         # Local dev — API + MLflow UI
├── pyproject.toml             # Dependencies and tooling config
└── Makefile                   # Shortcuts for every common task
```

---

## Running Locally

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package manager
- Docker Desktop — for containerised local run

### 1. Clone and install

```bash
git clone https://github.com/AkshatParikh16/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Creates .venv and installs all dependencies
uv sync --dev
```

### 2. Get the dataset

Download from Kaggle: [jpacse/datasets-for-churn-telecom](https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom)

Place the files in `data/raw/`:
```
data/raw/cell2celltrain.csv
data/raw/cell2celltest.csv
```

### 3. Explore the data

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 4. Train models

```bash
# Full training run with Optuna tuning (~45 min)
python -m churn.models.train

# Quick run with default parameters (~5 min)
python -m churn.models.train --skip-tuning
```

### 5. View results in MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
# Open http://localhost:5000
```

### 6. Start the API

```bash
uvicorn churn.api.main:app --reload --port 8000
# Open http://localhost:8000/docs
```

### 7. Run a prediction

```bash
curl -X POST http://localhost:8000/predict/raw \
  -H "Content-Type: application/json" \
  -d '{
    "MonthlyRevenue": 55.0,
    "MonthlyMinutes": 400,
    "DroppedCalls": 8,
    "MonthsInService": 6,
    "CreditRating": "Poor",
    "MadeCallToRetentionTeam": "Yes"
  }'
```

---

## API Reference

The API runs at `http://52.71.35.176:8000` (live on AWS).

Interactive docs: `http://52.71.35.176:8000/docs`

### Predict from raw customer data

Send the raw column values — the API handles all preprocessing internally.

```bash
POST /predict/raw

{
  "customer_id": "C12345",        # optional
  "MonthlyRevenue": 55.0,
  "MonthlyMinutes": 400,
  "DroppedCalls": 8,
  "MonthsInService": 6,
  "CreditRating": "Poor",
  "MadeCallToRetentionTeam": "Yes"
}
```

### Predict from pre-processed features

If you have already run the preprocessing pipeline yourself.

```bash
POST /predict

{
  "customer_id": "C12345",
  "features": [0.5, 1.2, -0.3, ...]   # must match feature_names.joblib ordering
}
```

### Batch prediction (up to 1,000 rows)

```bash
POST /predict/batch/raw

{
  "rows": [
    { "MonthlyRevenue": 55.0, "DroppedCalls": 8, ... },
    { "MonthlyRevenue": 90.0, "DroppedCalls": 1, ... }
  ]
}
```

### Response format

Every prediction returns:

```json
{
  "customer_id": "C12345",
  "churn_probability": 0.73,
  "churn_predicted": true,
  "risk_tier": "High",
  "threshold_used": 0.35,
  "top_reasons": [
    {
      "feature": "DroppedCalls",
      "shap_value": 0.18,
      "direction": "increases churn risk"
    },
    {
      "feature": "MonthsInService",
      "shap_value": 0.12,
      "direction": "increases churn risk"
    }
  ]
}
```

### Other endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check |
| `GET /ready` | Readiness check (model loaded) |
| `GET /model/info` | Model name, version, threshold, feature list |
| `GET /metrics` | Prometheus metrics |

---

## AWS Deployment

The deployment is fully automated. Every push to `main` triggers the GitHub Actions pipeline.

### How the pipeline works

```
git push main
      │
      ▼
GitHub Actions
  1. Build Docker image
  2. Push to AWS ECR
  3. SSH into EC2
  4. Pull latest image and restart container
  5. Health check — confirm API is responding
```

### Infrastructure (Terraform)

All AWS resources are defined as code in `infrastructure/`:

- **ECR** — private Docker image registry
- **EC2** (t3.micro, Free Tier) — inference server running Amazon Linux 2023
- **Elastic IP** — stable public IP address
- **IAM OIDC role** — lets GitHub Actions authenticate to AWS without storing any keys
- **Security group** — opens ports 80 and 8000

To re-provision from scratch:

```bash
cd infrastructure
terraform init
terraform apply \
  -var="github_org=AkshatParikh16" \
  -var="github_repo=Customer-Churn-Prediction" \
  -var="ec2_key_name=churn-api-key" \
  -var="ec2_instance_type=t3.micro"
```

### GitHub Actions Secrets required

| Secret | Value |
|---|---|
| `AWS_IAM_ROLE_ARN` | IAM OIDC role ARN from Terraform output |
| `AWS_REGION` | `us-east-1` |
| `ECR_REGISTRY` | ECR registry URL from Terraform output |
| `EC2_HOST` | EC2 public IP from Terraform output |
| `EC2_SSH_KEY` | Contents of `churn-api-key.pem` |

---

## Useful Commands

```bash
# Training
python -m churn.models.train               # full Optuna run
python -m churn.models.train --skip-tuning # fast run, default params

# Evaluation
python -m churn.models.evaluate            # SHAP plots + metrics

# API
uvicorn churn.api.main:app --reload        # dev server

# Batch predictions from CSV
python scripts/predict.py batch data/raw/cell2celltest.csv

# Docker
docker compose up                          # API + MLflow locally
docker compose down

# Infrastructure
cd infrastructure && terraform apply       # provision AWS
cd infrastructure && terraform destroy     # tear everything down
```

---

## License

MIT
