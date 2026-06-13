"""
src/churn/data/preprocess.py
─────────────────────────────
Preprocessing pipeline — original structure preserved, 9 data-quality
fixes and 8 engineered features added on top.

Data-quality fixes applied in basic_clean():
  Fix 1 — HandsetPrice "Unknown" string  → NaN
  Fix 2 — NewCellphoneUser / NotNewCellphoneUser mutual-exclusivity collapse
  Fix 3 — DroppedBlockedCalls aggregate dropped (components are reliable)
  Fix 4 — AgeHH1 / AgeHH2  zero          → NaN
  Fix 5 — Negative MonthlyRevenue / TotalRecurringCharge clipped to 0
           Negative CurrentEquipmentDays  → NaN
  Fix 6 — Block-missing patterns → 4 indicator features
  Fix 7 — PercChangeMinutes / PercChangeRevenues outlier cap (1st–99th pct)
  Fix 8 — MaritalStatus "Unknown"         → NaN
  Fix 9 — IncomeGroup 0                   → NaN  (treat as categorical)

Engineered features added in engineer_features():
  1. call_quality_rate    — dropped calls per 100 minutes
  2. revenue_per_minute   — spend-efficiency signal
  3. equipment_age_tier   — categorical bucket (12/24-month threshold effect)
  4. service_depth_score  — count of value-add service subscriptions
  5. engagement_trend     — average of PercChange dimensions
  6. retention_urgency    — called retention team AND rejected offer
  7. is_multi_person_hh   — AgeHH2 present → multi-person household
  8. is_low_usage         — < 200 min / month (disengagement flag)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from configs.settings import settings

BINARY_COLS: list[str] = [
    "ChildrenInHH", "HandsetRefurbished", "HandsetWebCapable",
    "TruckOwner", "RVOwner", "BuysViaMailOrder",
    "RespondsToMailOffers", "OptOutMailings", "NonUSTravel",
    "OwnsComputer", "HasCreditCard",
    "OwnsMotorcycle", "MadeCallToRetentionTeam",
]

DROP_COLS: list[str] = [settings.data.customer_id]

_ZERO_AS_NAN    = ["AgeHH1", "AgeHH2"]
_CLIP_TO_ZERO   = ["MonthlyRevenue", "TotalRecurringCharge"]
_NEG_TO_NAN     = ["CurrentEquipmentDays"]
_OUTLIER_CAP_COLS = [
    "PercChangeMinutes", "PercChangeRevenues",
    "OverageMinutes", "ReceivedCalls",
    "ReferralsMadeBySubscriber", "CustomerCareCalls", "RetentionCalls",
]
_ADDON_COLS = [
    "HandsetWebCapable", "BuysViaMailOrder", "RespondsToMailOffers",
    "OwnsComputer", "HasCreditCard",
]


def _fix_handset_price(df: pd.DataFrame) -> pd.DataFrame:
    if "HandsetPrice" in df.columns:
        before = df["HandsetPrice"].isna().sum()
        df["HandsetPrice"] = pd.to_numeric(df["HandsetPrice"], errors="coerce")
        after = df["HandsetPrice"].isna().sum()
        logger.debug("Fix 1 — HandsetPrice: {} 'Unknown' → NaN", after - before)
    return df


def _fix_mutual_exclusivity(df: pd.DataFrame) -> pd.DataFrame:
    if "NewCellphoneUser" in df.columns:
        df["is_new_cellphone_user"] = (df["NewCellphoneUser"] == "Yes").astype(int)
        df.drop(columns=["NewCellphoneUser", "NotNewCellphoneUser"], errors="ignore", inplace=True)
        logger.debug("Fix 2 — Collapsed mutual-exclusivity pair → is_new_cellphone_user")
    return df


def _fix_drop_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if "DroppedBlockedCalls" in df.columns:
        df.drop(columns=["DroppedBlockedCalls"], inplace=True)
        logger.debug("Fix 3 — Dropped unreliable DroppedBlockedCalls aggregate")
    return df


def _fix_zero_as_nan(df: pd.DataFrame) -> pd.DataFrame:
    for col in _ZERO_AS_NAN:
        if col in df.columns:
            n = (df[col] == 0).sum()
            df[col] = df[col].replace(0, np.nan)
            if n:
                logger.debug("Fix 4 — {}: {} zeros → NaN", col, n)
    return df


def _fix_negatives(df: pd.DataFrame) -> pd.DataFrame:
    for col in _CLIP_TO_ZERO:
        if col in df.columns:
            n = (df[col] < 0).sum()
            df[col] = df[col].clip(lower=0)
            if n:
                logger.debug("Fix 5a — {}: {} negatives clipped to 0", col, n)
    for col in _NEG_TO_NAN:
        if col in df.columns:
            n = (df[col] < 0).sum()
            df.loc[df[col] < 0, col] = np.nan
            if n:
                logger.debug("Fix 5b — {}: {} negatives → NaN", col, n)
    return df


def _fix_add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["flag_no_usage_data"]    = (df.get("MonthlyMinutes",    pd.Series(dtype=float)).isna() | df.get("MonthlyRevenue",      pd.Series(dtype=float)).isna()).astype(int)
    df["flag_no_perc_change"]   = (df.get("PercChangeMinutes", pd.Series(dtype=float)).isna() | df.get("PercChangeRevenues",   pd.Series(dtype=float)).isna()).astype(int)
    df["flag_no_age_data"]      = (df.get("AgeHH1",            pd.Series(dtype=float)).isna() & df.get("AgeHH2",              pd.Series(dtype=float)).isna()).astype(int)
    df["flag_no_handset_price"] = df.get("HandsetPrice",        pd.Series(dtype=float)).isna().astype(int)
    return df


def _fix_cap_outliers(df: pd.DataFrame, caps: dict | None = None) -> tuple[pd.DataFrame, dict]:
    if caps is None:
        caps = {}
        for col in _OUTLIER_CAP_COLS:
            if col in df.columns:
                caps[col] = (df[col].quantile(0.01), df[col].quantile(0.99))
    for col, (lo, hi) in caps.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    logger.debug("Fix 7 — Outliers capped in {} columns", len(caps))
    return df, caps


def _fix_marital_status(df: pd.DataFrame) -> pd.DataFrame:
    if "MaritalStatus" in df.columns:
        n = (df["MaritalStatus"] == "Unknown").sum()
        df["MaritalStatus"] = df["MaritalStatus"].replace("Unknown", np.nan)
        if n:
            logger.debug("Fix 8 — MaritalStatus: {} 'Unknown' → NaN", n)
    return df


def _fix_income_group(df: pd.DataFrame) -> pd.DataFrame:
    if "IncomeGroup" in df.columns:
        n = (df["IncomeGroup"] == 0).sum()
        df["IncomeGroup"] = df["IncomeGroup"].replace(0, np.nan)
        df["IncomeGroup"] = df["IncomeGroup"].astype("object")
        if n:
            logger.debug("Fix 9 — IncomeGroup: {} zeros → NaN (now categorical)", n)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    safe_min = df["MonthlyMinutes"].replace(0, np.nan)

    df["call_quality_rate"]   = (df["DroppedCalls"] / (safe_min / 100)).fillna(0).clip(0, 50)
    df["revenue_per_minute"]  = (df["MonthlyRevenue"] / safe_min).fillna(0).clip(0, 5)
    df["equipment_age_tier"]  = pd.cut(
        df["CurrentEquipmentDays"].clip(0, 1500),
        bins=[0, 180, 365, 540, 730, 1500],
        labels=["0-6mo", "6-12mo", "12-18mo", "18-24mo", "24mo+"],
    ).astype(str).fillna("unknown")

    addon_present = [c for c in _ADDON_COLS if c in df.columns]
    df["service_depth_score"] = df[addon_present].apply(lambda row: (row == 1).sum(), axis=1)

    pct_min = df["PercChangeMinutes"].fillna(0)
    pct_rev = df["PercChangeRevenues"].fillna(0)
    df["engagement_trend"] = (pct_min + pct_rev) / 2

    if "MadeCallToRetentionTeam" in df.columns and "RetentionOffersAccepted" in df.columns:
        df["retention_urgency"] = (
            (df["MadeCallToRetentionTeam"] == 1) & (df["RetentionOffersAccepted"] == 0)
        ).astype(int)
    else:
        df["retention_urgency"] = 0

    df["is_multi_person_hh"] = (df["AgeHH2"].fillna(0) > 0).astype(int)
    df["is_low_usage"]       = (df["MonthlyMinutes"].fillna(0) < 200).astype(int)

    logger.debug("Feature engineering — 8 features added")
    return df


def _binary_encode(df: pd.DataFrame) -> pd.DataFrame:
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})
    return df


def _encode_target(df: pd.DataFrame) -> pd.DataFrame:
    target = settings.data.target_col
    if target in df.columns:
        mapped = df[target].map({"Yes": 1, "No": 0})
        if mapped.isna().any():
            df = df.drop(columns=[target])
        else:
            df[target] = mapped.astype(int)
    return df


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Loading raw Cell2Cell CSVs…")
    train = pd.read_csv(settings.data.train_path)
    test  = pd.read_csv(settings.data.test_path)
    logger.info("Train shape: {}  |  Test shape: {}", train.shape, test.shape)
    return train, test


def basic_clean(df: pd.DataFrame, caps: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    df = _fix_handset_price(df)
    df = _fix_mutual_exclusivity(df)
    df = _fix_drop_aggregate(df)
    df = _fix_zero_as_nan(df)
    df = _fix_negatives(df)
    df = _fix_add_missing_indicators(df)
    df, caps = _fix_cap_outliers(df, caps)
    df = _fix_marital_status(df)
    df = _fix_income_group(df)
    df = _binary_encode(df)
    df = engineer_features(df)
    df = _encode_target(df)

    return df, caps


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    target = settings.data.target_col
    all_features = [c for c in df.columns if c != target]
    cat_cols = [c for c in all_features if df[c].dtype == object]
    num_cols = [c for c in all_features if c not in cat_cols]
    return num_cols, cat_cols


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute",  SimpleImputer(strategy="most_frequent")),
        ("encode",  OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_datasets() -> dict[str, object]:
    import joblib
    train_raw, test_raw = load_raw()

    train_clean, caps = basic_clean(train_raw, caps=None)
    test_clean,  _    = basic_clean(test_raw,  caps=caps)

    target = settings.data.target_col
    X = train_clean.drop(columns=[target])
    y = train_clean[target]
    X_test_raw = test_clean.drop(columns=[target], errors="ignore")
    y_test     = test_clean[target] if target in test_clean.columns else None

    num_cols, cat_cols = get_feature_columns(train_clean)
    logger.info("Numeric: {}  |  Categorical: {}  |  Total: {}", len(num_cols), len(cat_cols), len(num_cols)+len(cat_cols))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=settings.data.test_size,
        random_state=settings.data.random_state, stratify=y,
    )

    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t   = preprocessor.transform(X_val)
    X_test_t  = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names_out().tolist()

    logger.success("Preprocessing complete — train: {} | val: {} | test: {}",
                   X_train_t.shape, X_val_t.shape, X_test_t.shape)

    artifacts = Path(settings.model.models_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(caps,          artifacts / "outlier_caps.joblib")
    joblib.dump(preprocessor,  artifacts / "preprocessor.joblib")
    joblib.dump(feature_names, artifacts / "feature_names.joblib")
    logger.info("Artefacts saved → {}", artifacts)

    return {
        "X_train": X_train_t, "X_val": X_val_t, "X_test": X_test_t,
        "y_train": y_train.to_numpy(), "y_val": y_val.to_numpy(),
        "y_test":  y_test.to_numpy() if y_test is not None else np.array([]),
        "preprocessor": preprocessor, "feature_names": feature_names,
        "num_cols": num_cols, "cat_cols": cat_cols, "caps": caps,
    }
