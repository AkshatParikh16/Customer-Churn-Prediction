"""
configs/settings.py
───────────────────
Single source of truth for every configurable value.
Uses pydantic-settings: reads from environment variables and/or .env file.
No hardcoded secrets — everything is overridable at deploy time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent


class DataSettings(BaseSettings):
    raw_dir: Path       = ROOT_DIR / "data" / "raw"
    processed_dir: Path = ROOT_DIR / "data" / "processed"

    train_filename: str = "cell2celltrain.csv"
    test_filename: str  = "cell2celltest.csv"

    target_col: str    = "Churn"
    customer_id: str   = "CustomerID"

    # Cell2Cell: 71.2% non-churn / 28.8% churn → imbalanced
    test_size: float   = 0.20
    random_state: int  = 42

    @property
    def train_path(self) -> Path:
        return self.raw_dir / self.train_filename

    @property
    def test_path(self) -> Path:
        return self.raw_dir / self.test_filename

    model_config = SettingsConfigDict(env_prefix="DATA_", env_file=".env", extra="ignore")


class ModelSettings(BaseSettings):
    models_dir: Path   = ROOT_DIR / "models"
    cv_folds: int      = 5
    random_state: int  = 42

    # XGBoost — tuned for imbalanced Cell2Cell (28.8% churn)
    xgb_n_estimators: int        = 500
    xgb_max_depth: int           = 6
    xgb_learning_rate: float     = 0.05
    xgb_subsample: float         = 0.8
    xgb_colsample_bytree: float  = 0.8
    xgb_scale_pos_weight: float  = 2.47   # (1 - 0.288) / 0.288
    xgb_reg_alpha: float         = 0.1
    xgb_reg_lambda: float        = 1.0
    xgb_tree_method: str         = "hist"

    # Optuna tuning
    optuna_n_trials: int         = 50
    optuna_timeout_secs: int     = 600

    # Recall threshold — optimise for churn recall (business cost of FN > FP)
    decision_threshold: float    = 0.40

    model_config = SettingsConfigDict(env_prefix="MODEL_", env_file=".env", extra="ignore")


class MLflowSettings(BaseSettings):
    tracking_uri: str  = "sqlite:///" + ROOT_DIR.as_posix() + "/mlruns/mlflow.db"
    experiment:   str  = "customer-churn-cell2cell"
    registry_name: str = "churn-xgboost-prod"
    artifact_root: str = str(ROOT_DIR / "mlruns" / "artifacts")

    model_config = SettingsConfigDict(env_prefix="MLFLOW_", env_file=".env", extra="ignore")


class APISettings(BaseSettings):
    host: str          = "0.0.0.0"
    port: int          = 8000
    workers: int       = 1
    log_level: str     = "info"
    environment: Literal["development", "staging", "production"] = "development"
    model_uri: str     = str(ROOT_DIR / "models" / "churn_xgb_prod.joblib")

    @field_validator("port")
    @classmethod
    def valid_port(cls, v: int) -> int:
        if not (1024 <= v <= 65535):
            raise ValueError("Port must be between 1024 and 65535")
        return v

    model_config = SettingsConfigDict(env_prefix="API_", env_file=".env", extra="ignore")


class AWSSettings(BaseSettings):
    region: str             = "us-east-1"
    ecr_repository: str     = "customer-churn-prediction"
    ecr_registry: str       = ""           # filled at deploy: <account>.dkr.ecr.<region>.amazonaws.com
    s3_bucket: str          = "churn-mlflow-artifacts"
    ec2_instance_type: str  = "t3.medium"

    model_config = SettingsConfigDict(env_prefix="AWS_", env_file=".env", extra="ignore")


class Settings(BaseSettings):
    """Aggregate settings — import this everywhere."""
    data:   DataSettings   = Field(default_factory=DataSettings)
    model:  ModelSettings  = Field(default_factory=ModelSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    api:    APISettings    = Field(default_factory=APISettings)
    aws:    AWSSettings    = Field(default_factory=AWSSettings)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# ── Singleton ──────────────────────────────────────────────────────────────────
settings = Settings()
