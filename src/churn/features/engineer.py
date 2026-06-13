"""
src/churn/features/engineer.py
───────────────────────────────
Feature engineering on top of preprocessed Cell2Cell data:
  • SMOTE oversampling for the 28.8% minority class
  • Interaction & ratio features (usage patterns, pricing signals)
  • Feature selection via XGBoost importance + threshold

Cell2Cell domain knowledge encoded here:
  - High DroppedCalls + Low MonthsInService = churn risk
  - HandsetPrice Unknown / Missing = churn signal
  - CreditRating correlates with contract tenure
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from loguru import logger
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from configs.settings import settings


# ── SMOTE ─────────────────────────────────────────────────────────────────────

def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to balance the training set.
    Cell2Cell: 71.2% No-churn / 28.8% Churn → after SMOTE → ~50/50.
    Only applied to TRAINING data, never val/test.
    """
    logger.info(
        "Before SMOTE — class distribution: {}",
        dict(zip(*np.unique(y_train, return_counts=True))),
    )
    smote = SMOTE(
        sampling_strategy="minority",
        k_neighbors=5,
        random_state=settings.model.random_state,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.success(
        "After SMOTE  — class distribution: {}",
        dict(zip(*np.unique(y_res, return_counts=True))),
    )
    return X_res, y_res


# ── Interaction features (raw DataFrame level, before sklearn pipeline) ────────

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-driven interaction features for Cell2Cell.
    Call this on the RAW/cleaned DataFrame before the sklearn pipeline.
    Adds columns — preprocessor will pick them up as numeric.
    """
    df = df.copy()

    # Usage ratios ─────────────────────────────────────────────
    if {"MonthlyRevenue", "MonthlyMinutes"}.issubset(df.columns):
        df["RevenuePerMinute"] = df["MonthlyRevenue"] / (df["MonthlyMinutes"] + 1e-6)

    if {"DroppedCalls", "ReceivedCalls"}.issubset(df.columns):
        df["DropRate"] = df["DroppedCalls"] / (df["ReceivedCalls"] + 1e-6)

    if {"RoamingCalls", "MonthlyMinutes"}.issubset(df.columns):
        df["RoamingRatio"] = df["RoamingCalls"] / (df["MonthlyMinutes"] + 1e-6)

    # Tenure signal ────────────────────────────────────────────
    if "MonthsInService" in df.columns:
        df["IsNewCustomer"]  = (df["MonthsInService"] <= 3).astype(int)
        df["IsLongCustomer"] = (df["MonthsInService"] >= 24).astype(int)

    # Customer service interaction risk ───────────────────────
    if {"CustomerCareCalls", "DroppedCalls"}.issubset(df.columns):
        df["ServiceStressIndex"] = df["CustomerCareCalls"] * df["DroppedCalls"]

    # Overage risk ─────────────────────────────────────────────
    if {"OverageMinutes", "MonthlyMinutes"}.issubset(df.columns):
        df["OverageRatio"] = df["OverageMinutes"] / (df["MonthlyMinutes"] + 1e-6)

    logger.debug("Interaction features added. New shape: {}", df.shape)
    return df


# ── Feature selection via XGBoost importance ──────────────────────────────────

def select_top_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    threshold: str = "median",
) -> tuple[np.ndarray, list[str], SelectFromModel]:
    """
    Fit a fast XGBoost to get feature importances and keep top features.
    Returns reduced X_train, selected feature names, and the fitted selector.
    """
    logger.info("Running feature selection (threshold={})…", threshold)
    selector_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=settings.model.xgb_scale_pos_weight,
        tree_method="hist",
        use_label_encoder=False,
        eval_metric="auc",
        random_state=settings.model.random_state,
        n_jobs=-1,
    )
    selector = SelectFromModel(selector_model, threshold=threshold)
    X_selected = selector.fit_transform(X_train, y_train)

    selected_mask  = selector.get_support()
    selected_names = [n for n, m in zip(feature_names, selected_mask) if m]
    logger.success(
        "Feature selection: {} → {} features retained",
        len(feature_names), len(selected_names),
    )
    return X_selected, selected_names, selector
