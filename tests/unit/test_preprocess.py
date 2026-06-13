"""
tests/unit/test_preprocess.py
──────────────────────────────
Unit tests for the preprocessing pipeline.
Uses synthetic data — no dataset download required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from churn.data.preprocess import (
    basic_clean,
    build_preprocessor,
    engineer_features,
    get_feature_columns,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Minimal Cell2Cell-like DataFrame for unit tests."""
    return pd.DataFrame({
        "CustomerID":              ["C1", "C2", "C3", "C4", "C5"],
        "Churn":                   ["Yes", "No", "Yes", "No", "Yes"],
        "MonthlyRevenue":          [45.5, 78.2, 30.0, 99.9, 55.1],
        "MonthlyMinutes":          [300, 500, 200, 800, 350],
        "MonthsInService":         [12, 36, 3, 48, 6],
        "DroppedCalls":            [5, 1, 10, 0, 3],
        "ChildrenInHH":            ["Yes", "No", "No", "Yes", "No"],
        "HandsetRefurbished":      ["No", "No", "Yes", "No", "No"],
        "CreditRating":            ["Good", "Excellent", "Fair", "Good", "Poor"],
        "Occupation":              ["Professional", "Student", "Clerical", "Professional", "Student"],
        # columns required by fix 6 + engineer_features
        "PercChangeMinutes":       [5.0, -2.0, 0.0, 10.0, float("nan")],
        "PercChangeRevenues":      [3.0, -1.0, float("nan"), 8.0, 0.0],
        "AgeHH1":                  [35.0, 42.0, 0.0, 55.0, 28.0],
        "AgeHH2":                  [0.0, 38.0, 0.0, 50.0, float("nan")],
        "HandsetPrice":            [100.0, float("nan"), 200.0, 150.0, 75.0],
        "CurrentEquipmentDays":    [180, 400, 730, 90, -10],
        "RetentionOffersAccepted": [0, 1, 0, 0, 1],
    })


# ── Tests: basic_clean ────────────────────────────────────────────────────────

def test_basic_clean_drops_customer_id(sample_df: pd.DataFrame) -> None:
    cleaned, _ = basic_clean(sample_df)
    assert "CustomerID" not in cleaned.columns


def test_basic_clean_encodes_target(sample_df: pd.DataFrame) -> None:
    cleaned, _ = basic_clean(sample_df)
    assert set(cleaned["Churn"].unique()).issubset({0, 1})


def test_basic_clean_binary_encoding(sample_df: pd.DataFrame) -> None:
    cleaned, _ = basic_clean(sample_df)
    assert set(cleaned["ChildrenInHH"].unique()).issubset({0, 1})
    assert set(cleaned["HandsetRefurbished"].unique()).issubset({0, 1})


def test_basic_clean_preserves_row_count(sample_df: pd.DataFrame) -> None:
    cleaned, _ = basic_clean(sample_df)
    assert len(cleaned) == len(sample_df)


def test_basic_clean_returns_caps_dict(sample_df: pd.DataFrame) -> None:
    _, caps = basic_clean(sample_df)
    assert isinstance(caps, dict)


def test_basic_clean_adds_flag_columns(sample_df: pd.DataFrame) -> None:
    cleaned, _ = basic_clean(sample_df)
    for col in ["flag_no_usage_data", "flag_no_perc_change", "flag_no_age_data", "flag_no_handset_price"]:
        assert col in cleaned.columns


def test_basic_clean_clips_negative_equipment_days(sample_df: pd.DataFrame) -> None:
    cleaned, _ = basic_clean(sample_df)
    # CurrentEquipmentDays had one -10 value → should become NaN, not negative
    assert (cleaned["CurrentEquipmentDays"].dropna() >= 0).all()


# ── Tests: get_feature_columns ────────────────────────────────────────────────

def test_get_feature_columns_splits_correctly(sample_df: pd.DataFrame) -> None:
    cleaned, _ = basic_clean(sample_df)
    num_cols, cat_cols = get_feature_columns(cleaned)
    assert "Churn" not in num_cols
    assert "Churn" not in cat_cols
    # CreditRating and Occupation are object → categorical
    assert "CreditRating" in cat_cols
    assert "Occupation" in cat_cols


# ── Tests: build_preprocessor ────────────────────────────────────────────────

def test_preprocessor_is_pipeline(sample_df: pd.DataFrame) -> None:
    cleaned, _ = basic_clean(sample_df)
    num_cols, cat_cols = get_feature_columns(cleaned)
    prep = build_preprocessor(num_cols, cat_cols)
    assert hasattr(prep, "fit_transform")


def test_preprocessor_fit_transform_shape(sample_df: pd.DataFrame) -> None:
    cleaned, _ = basic_clean(sample_df)
    num_cols, cat_cols = get_feature_columns(cleaned)
    X = cleaned.drop(columns=["Churn"])
    prep = build_preprocessor(num_cols, cat_cols)
    X_t = prep.fit_transform(X)
    assert X_t.shape[0] == len(sample_df)
    assert X_t.shape[1] == len(num_cols) + len(cat_cols)


def test_preprocessor_handles_missing(sample_df: pd.DataFrame) -> None:
    """Preprocessor must not error on NaN values (Cell2Cell has 15 cols with NaN)."""
    df_with_nan = sample_df.copy()
    df_with_nan.loc[0, "MonthlyRevenue"] = float("nan")
    cleaned, _ = basic_clean(df_with_nan)
    num_cols, cat_cols = get_feature_columns(cleaned)
    X = cleaned.drop(columns=["Churn"])
    prep = build_preprocessor(num_cols, cat_cols)
    X_t = prep.fit_transform(X)
    assert not np.isnan(X_t).any()


# ── Tests: engineer_features ──────────────────────────────────────────────────

@pytest.fixture()
def eng_df() -> pd.DataFrame:
    """DataFrame with all columns engineer_features reads directly."""
    return pd.DataFrame({
        "MonthlyMinutes":          [300.0, 0.0, 500.0],
        "MonthlyRevenue":          [45.5, 0.0, 80.0],
        "DroppedCalls":            [5.0, 0.0, 2.0],
        "CurrentEquipmentDays":    [180.0, 400.0, 730.0],
        "PercChangeMinutes":       [5.0, float("nan"), -3.0],
        "PercChangeRevenues":      [3.0, 1.0, float("nan")],
        "AgeHH2":                  [0.0, 38.0, float("nan")],
        "MonthlyMinutes":          [300.0, 0.0, 500.0],
    })


def test_engineer_features_adds_all_columns(eng_df: pd.DataFrame) -> None:
    result = engineer_features(eng_df)
    for col in ["call_quality_rate", "revenue_per_minute", "equipment_age_tier",
                "service_depth_score", "engagement_trend", "retention_urgency",
                "is_multi_person_hh", "is_low_usage"]:
        assert col in result.columns, f"Missing: {col}"


def test_engineer_features_call_quality_zero_minutes(eng_df: pd.DataFrame) -> None:
    result = engineer_features(eng_df)
    # row with 0 minutes → safe_min is NaN → call_quality_rate filled to 0
    assert result.loc[1, "call_quality_rate"] == 0.0


def test_engineer_features_is_low_usage(eng_df: pd.DataFrame) -> None:
    result = engineer_features(eng_df)
    # 300 min → not low usage; 0 min → low usage; 500 min → not low
    assert result.loc[0, "is_low_usage"] == 0
    assert result.loc[1, "is_low_usage"] == 1
    assert result.loc[2, "is_low_usage"] == 0


def test_engineer_features_is_multi_person_hh(eng_df: pd.DataFrame) -> None:
    result = engineer_features(eng_df)
    assert result.loc[0, "is_multi_person_hh"] == 0   # AgeHH2 == 0
    assert result.loc[1, "is_multi_person_hh"] == 1   # AgeHH2 == 38


def test_engineer_features_engagement_trend_nan_fill(eng_df: pd.DataFrame) -> None:
    result = engineer_features(eng_df)
    # NaN in PercChange cols should be filled with 0 before averaging
    assert not pd.isna(result["engagement_trend"]).any()


def test_engineer_features_equipment_age_tier_dtype(eng_df: pd.DataFrame) -> None:
    result = engineer_features(eng_df)
    assert result["equipment_age_tier"].dtype == object


def test_engineer_features_does_not_mutate_input(eng_df: pd.DataFrame) -> None:
    original_cols = list(eng_df.columns)
    engineer_features(eng_df)
    assert list(eng_df.columns) == original_cols
