# ─────────────────────────────────────────────────────────────────────────────
# Makefile — Customer Churn Prediction
# Uses: uv (package manager), ruff (lint/format), mypy (types), pytest
# ─────────────────────────────────────────────────────────────────────────────

.DEFAULT_GOAL := help
PYTHON        := uv run python
UV            := uv

.PHONY: help setup lint format typecheck test train evaluate serve docker-build docker-up tf-init tf-plan tf-apply tf-destroy clean

# ── Help ───────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Customer Churn Prediction — Cell2Cell (Duke / Teradata)"
	@echo "  ────────────────────────────────────────────────────────"
	@echo "  make setup        Install all dependencies via uv"
	@echo "  make ingest       Download Cell2Cell from Kaggle"
	@echo "  make train        Train XGBoost + LightGBM with MLflow"
	@echo "  make train-fast   Train with default params (no Optuna)"
	@echo "  make serve        Start FastAPI dev server"
	@echo "  make lint         Ruff lint check"
	@echo "  make format       Ruff auto-format"
	@echo "  make typecheck    Mypy static analysis"
	@echo "  make test         Run all tests with coverage"
	@echo "  make evaluate     Run full evaluation (SHAP + ROC/PR + confusion matrix)"
	@echo "  make docker-build Build production Docker image"
	@echo "  make docker-up    Start full stack (API + MLflow UI)"
	@echo "  make mlflow-ui    Open MLflow at localhost:5000"
	@echo "  make tf-init      terraform init (AWS infra)"
	@echo "  make tf-plan      terraform plan"
	@echo "  make tf-apply     terraform apply (provision ECR + EC2)"
	@echo "  make clean        Remove build artifacts"
	@echo ""

# ── Setup ──────────────────────────────────────────────────────────────────────
setup:
	$(UV) sync --dev
	@echo "✅ Dependencies installed"

# ── Data ───────────────────────────────────────────────────────────────────────
ingest:
	$(PYTHON) -m churn.data.ingest

ingest-local:
	$(PYTHON) -m churn.data.ingest \
		--train data/raw/cell2celltrain.csv \
		--test  data/raw/cell2celltest.csv

# ── Training ───────────────────────────────────────────────────────────────────
train:
	$(PYTHON) -m churn.models.train --trials 50

train-fast:
	$(PYTHON) -m churn.models.train --skip-tuning

# ── API ────────────────────────────────────────────────────────────────────────
serve:
	$(UV) run uvicorn churn.api.main:app \
		--host 0.0.0.0 --port 8000 --reload

# ── Code quality ───────────────────────────────────────────────────────────────
lint:
	$(UV) run ruff check src/ tests/

format:
	$(UV) run ruff format src/ tests/
	$(UV) run ruff check --fix src/ tests/

typecheck:
	$(UV) run mypy src/churn --ignore-missing-imports

# ── Tests ──────────────────────────────────────────────────────────────────────
test:
	$(UV) run pytest tests/ -v \
		--cov=src/churn \
		--cov-report=term-missing \
		--cov-report=html:reports/coverage

test-unit:
	$(UV) run pytest tests/unit/ -v

test-integration:
	$(UV) run pytest tests/integration/ -v

# ── Docker ─────────────────────────────────────────────────────────────────────
docker-build:
	docker build --target runtime -t customer-churn-prediction:latest .

docker-up:
	docker compose up --build -d
	@echo "API   → http://localhost:8000/docs"
	@echo "MLflow → http://localhost:5000"

docker-down:
	docker compose down

mlflow-ui:
	@$(PYTHON) -c "import webbrowser; webbrowser.open('http://localhost:5000')"

# ── Evaluate ──────────────────────────────────────────────────────────────────
evaluate:
	$(PYTHON) -m churn.models.evaluate

# ── Terraform (AWS infra) ──────────────────────────────────────────────────────
tf-init:
	cd infrastructure && terraform init

tf-plan:
	cd infrastructure && terraform plan -out=tfplan

tf-apply:
	cd infrastructure && terraform apply tfplan

tf-destroy:
	cd infrastructure && terraform destroy

# ── AWS ECR push ───────────────────────────────────────────────────────────────
ecr-push:
	@echo "Pushing to ECR — set AWS_ACCOUNT_ID and AWS_REGION first"
	aws ecr get-login-password --region $(AWS_REGION) \
		| docker login --username AWS \
		  --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
	docker tag customer-churn-prediction:latest \
		$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/customer-churn-prediction:latest
	docker push \
		$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/customer-churn-prediction:latest

# ── Clean ──────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__"  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov"      -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ reports/coverage/
	@echo "🧹 Clean"
