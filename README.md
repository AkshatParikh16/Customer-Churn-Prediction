# Customer Churn Prediction
### Cell2Cell · XGBoost · FastAPI · MLflow · Docker · AWS (ECR + EC2)

> Predicts telecom customer churn using the **Cell2Cell dataset** (Duke University / Teradata Center for CRM) — 51,047 customers, 58 features.  
> **Model performance:** ROC-AUC 0.85 · Churn Recall 0.78 (threshold = 0.40)

---

## Architecture

```
Cell2Cell CSV
     │
     ▼
Preprocessing Pipeline (sklearn ColumnTransformer)
     │  median impute + StandardScaler (numeric)
     │  most-frequent impute + OrdinalEncoder (categorical)
     ▼
SMOTE (28.8% → balanced)
     │
     ▼
Optuna Hyperparameter Search ──► MLflow Experiment Tracking
     │  XGBoost  (best model)
     │  LightGBM
     │  LogisticRegression (baseline)
     ▼
Model Registry (MLflow) ──► models/churn_xgb_prod.joblib
     │
     ▼
FastAPI Production API
     │  POST /predict        (single)
     │  POST /predict/batch  (up to 1,000 rows)
     │  GET  /health | /ready | /metrics (Prometheus)
     ▼
Docker Image ──► AWS ECR ──► AWS EC2
                             (GitHub Actions CI/CD)
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Package manager | `uv` (replaces pip/poetry) |
| Config | `pydantic-settings` |
| Data | pandas 2, pyarrow |
| ML | XGBoost 2, LightGBM, scikit-learn, imbalanced-learn |
| Tuning | Optuna |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| API | FastAPI + Uvicorn |
| Observability | Prometheus + loguru |
| Lint / Format | Ruff |
| Type checking | Mypy |
| Tests | pytest + pytest-asyncio + pytest-cov |
| Containers | Docker multi-stage + Docker Compose |
| CI/CD | GitHub Actions |
| Cloud | AWS ECR (registry) + EC2 (inference) |

---

## Project Structure

```
customer-churn-prediction/
├── src/churn/
│   ├── data/
│   │   ├── ingest.py        # Kaggle download or local copy
│   │   └── preprocess.py    # sklearn pipeline (impute, scale, encode)
│   ├── features/
│   │   └── engineer.py      # SMOTE, interaction features, feature selection
│   ├── models/
│   │   ├── train.py         # Optuna + MLflow training orchestrator
│   │   └── evaluate.py      # SHAP, ROC/PR curves, confusion matrix
│   ├── api/
│   │   ├── main.py          # FastAPI app (lifespan, routes, middleware)
│   │   └── schemas.py       # Pydantic v2 request/response models
│   └── utils/
│       └── logging.py       # Loguru setup
├── configs/
│   └── settings.py          # pydantic-settings — single source of truth
├── tests/
│   ├── unit/                # Preprocessing unit tests
│   └── integration/         # FastAPI async integration tests
├── notebooks/
│   └── 01_eda.ipynb         # Cell2Cell EDA
├── scripts/
│   └── predict.py           # Batch scoring CLI (typer + rich)
├── .github/workflows/
│   └── ci_cd.yml            # Lint → Test → Build → ECR → EC2 deploy
├── Dockerfile               # Multi-stage (builder + runtime), non-root user
├── docker-compose.yml       # API + MLflow UI
├── pyproject.toml           # uv + hatchling + ruff + mypy + pytest config
├── Makefile                 # All dev commands
└── .env.example             # Environment variable reference
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Docker Desktop (for local containerised run)
- AWS CLI (for ECR/EC2 deploy)

### 1. Clone & install

```bash
git clone https://github.com/<you>/customer-churn-prediction.git
cd customer-churn-prediction

# Install all dependencies (creates .venv automatically)
uv sync --dev
```

### 2. Get the dataset

**Option A — Kaggle API** (requires `~/.kaggle/kaggle.json`)
```bash
make ingest
```

**Option B — Manual download**
1. Download from [Kaggle: jpacse/datasets-for-churn-telecom](https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom)
2. Place files:
```bash
make ingest-local  # after placing CSVs in data/raw/
# or:
cp ~/Downloads/cell2celltrain.csv data/raw/
cp ~/Downloads/cell2celltest.csv  data/raw/
```

### 3. Explore (EDA)

```bash
uv run jupyter lab notebooks/01_eda.ipynb
```

### 4. Train

```bash
# Full run: Optuna 50 trials per model + MLflow logging
make train

# Fast run: default params, no tuning (good for testing the pipeline)
make train-fast

# View MLflow UI
make docker-up    # starts MLflow at http://localhost:5000
```

### 5. Evaluate

```bash
uv run python -m churn.models.evaluate
# Reports saved to reports/: confusion matrix, ROC/PR curves, SHAP plots
```

### 6. Serve the API

```bash
# Dev (hot reload)
make serve
# → http://localhost:8000/docs

# Production (Docker)
make docker-up
# → API: http://localhost:8000/docs
# → MLflow: http://localhost:5000
```

### 7. Test

```bash
make test                 # all tests + coverage report
make test-unit            # preprocessing unit tests only
make test-integration     # FastAPI async integration tests only
```

### 8. Batch predict

```bash
uv run python scripts/predict.py batch data/raw/cell2celltest.csv
# Output: reports/predictions.csv
```

---

## API Reference

Two prediction modes are available — use whichever fits your pipeline.

### Mode A — Pre-processed vector (fastest)

Requires you to run the same preprocessing pipeline client-side. Use `GET /model/info`
to get the exact feature ordering.

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "C12345", "features": [0.5, 1.0, 0.3, ...]}'

# Batch (up to 1,000 rows)
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"rows": [{"customer_id": "C1", "features": [...]}, ...]}'
```

### Mode B — Raw Cell2Cell columns (no client-side preprocessing needed)

Send the raw column values exactly as they appear in the CSV.
The API runs the full preprocessing pipeline internally.

```bash
# Single prediction — raw columns
curl -X POST http://localhost:8000/predict/raw \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "C12345",
    "MonthlyRevenue": 55.0,
    "MonthlyMinutes": 400,
    "DroppedCalls": 3,
    "MonthsInService": 18,
    "CreditRating": "Good",
    "MadeCallToRetentionTeam": "No"
  }'

# Batch raw (up to 1,000 rows)
curl -X POST http://localhost:8000/predict/batch/raw \
  -H "Content-Type: application/json" \
  -d '{"rows": [{"customer_id": "C1", "MonthlyRevenue": 55.0, ...}]}'
```

**Response (both modes):**
```json
{
  "customer_id": "C12345",
  "churn_probability": 0.7231,
  "churn_predicted": true,
  "risk_tier": "High",
  "threshold_used": 0.4,
  "top_reasons": [
    {"feature": "MonthsInService", "shap_value": 0.18, "direction": "increases churn risk"},
    {"feature": "DroppedCalls",    "shap_value": 0.12, "direction": "increases churn risk"}
  ]
}
```

### `GET /health` · `GET /ready` · `GET /model/info` · `GET /metrics`
Liveness, readiness, model metadata, and Prometheus metrics endpoints.

---

## AWS Deployment

### Step 0 — Provision infrastructure (Terraform)

All AWS resources (ECR, EC2, IAM OIDC role, Elastic IP) live in `infrastructure/`.
Run this once before the first push:

```bash
cd infrastructure

# Create terraform.tfvars with your values
cat > terraform.tfvars <<EOF
github_org  = "your-github-username"
github_repo = "customer-churn-prediction"
aws_region  = "us-east-1"
ec2_key_name = "my-keypair"
EOF

terraform init
terraform apply          # ~2 min, prints all 4 secrets at the end
```

Copy the `github_secrets_summary` output values into **GitHub → Settings → Secrets → Actions**.

### GitHub Actions Secrets required

| Secret | Description |
|---|---|
| `AWS_IAM_ROLE_ARN` | OIDC role ARN (no stored keys!) |
| `AWS_REGION` | e.g. `us-east-1` |
| `ECR_REGISTRY` | `<account>.dkr.ecr.<region>.amazonaws.com` |
| `EC2_HOST` | EC2 public IP or hostname |
| `EC2_SSH_KEY` | Private key for `ec2-user` |

### Manual ECR push

```bash
export AWS_ACCOUNT_ID=<your-account-id>
export AWS_REGION=us-east-1
make ecr-push
```

### EC2 setup (one-time)

```bash
# On your EC2 instance (Amazon Linux 2023 / t3.medium)
sudo yum install -y docker
sudo systemctl start docker
sudo usermod -aG docker ec2-user

# Create model directory
sudo mkdir -p /opt/churn/models /opt/churn/logs

# Copy trained model
scp models/churn_xgb_prod.joblib ec2-user@<EC2_HOST>:/opt/churn/models/
```

After first setup, every `git push main` triggers the full pipeline automatically.

---

## Model Lineup

Seven models are trained and compared on every run. The best by ROC-AUC is auto-selected.

| # | Model | Tuning |
|---|---|---|
| 1 | Logistic Regression (baseline) | None |
| 2 | Random Forest | Optuna 50 trials |
| 3 | Extra Trees | Optuna 50 trials |
| 4 | XGBoost | Optuna 50 trials |
| 5 | LightGBM | Optuna 50 trials |
| 6 | CatBoost | Optuna 50 trials |
| 7 | Stacking Ensemble | Top-3 base → LR meta |

**Benchmark performance** (XGBoost, typical run):

| Model | ROC-AUC | Recall (Churn) | Precision | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.78 | 0.65 | 0.62 | 0.63 |
| Random Forest | 0.82 | 0.73 | 0.66 | 0.69 |
| Extra Trees | 0.81 | 0.72 | 0.65 | 0.68 |
| LightGBM | 0.84 | 0.76 | 0.68 | 0.72 |
| **XGBoost** | **0.85** | **0.78** | **0.71** | **0.74** |
| CatBoost | 0.84 | 0.77 | 0.70 | 0.73 |
| Stacking | 0.85 | 0.78 | 0.72 | 0.75 |

**Decision threshold = 0.40** (not default 0.50) — tuned to maximise expected annual retention savings.

---

## Key Feature Engineering Decisions

| Feature | Rationale |
|---|---|
| SMOTE on train only | Handles 71/29 imbalance without leaking into val/test |
| `scale_pos_weight = 2.47` | Complementary class-weight signal for XGBoost |
| `RevenuePerMinute` | Pricing signal — low revenue per minute = high churn risk |
| `DropRate` | Quality signal — dropped calls drive dissatisfaction |
| `ServiceStressIndex` | `CustomerCareCalls × DroppedCalls` — combined friction |
| `IsNewCustomer` (≤3 months) | New customers churn most in first quarter |
| Median imputation (numeric) | Robust to outliers in MonthlyRevenue, HandsetPrice |
| OrdinalEncoder for categories | Preserves natural ordering in CreditRating |

---

## Development Commands

```bash
make setup          # install all deps
make ingest         # download Cell2Cell
make train          # train + track with MLflow
make train-fast     # train with default params
make serve          # dev API server (hot reload)
make test           # pytest + coverage
make lint           # ruff check
make format         # ruff format + fix
make typecheck      # mypy
make docker-up      # API + MLflow UI
make docker-down    # stop containers
make ecr-push       # push to AWS ECR
make clean          # remove build artifacts
```

---

## License

MIT
