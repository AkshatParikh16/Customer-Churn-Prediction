"""
src/churn/api/main.py
"""
from __future__ import annotations
import sys, time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import joblib, numpy as np, pandas as pd, uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from configs.settings import settings
from churn.api.schemas import (
    BatchPredictRequest, BatchPredictResponse,
    BatchRawRequest,
    HealthResponse,
    PredictRequest, PredictResponse,
    RawCustomerRequest,
    ShapReason,
)
from churn.data.preprocess import basic_clean

_MODEL            = None
_METADATA: dict   = {}
_SHAP_EXPLAINER   = None
_FEATURE_NAMES: list[str] = []
_PREPROCESSOR     = None
_OUTLIER_CAPS: dict = {}
_MODEL_LOADED_AT: float = 0.0


def _load_model() -> None:
    global _MODEL, _METADATA, _SHAP_EXPLAINER, _FEATURE_NAMES
    global _PREPROCESSOR, _OUTLIER_CAPS, _MODEL_LOADED_AT

    models_dir = Path(settings.model.models_dir)

    meta_path = models_dir / "model_metadata.joblib"
    if meta_path.exists():
        _METADATA = joblib.load(meta_path)
    else:
        _METADATA = {"best_threshold": settings.model.decision_threshold}

    model_filename = _METADATA.get("model_filename", "churn_xgb_prod.joblib")
    model_path = models_dir / model_filename
    if not model_path.exists():
        model_path = Path(settings.api.model_uri)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    _MODEL = joblib.load(model_path)
    _MODEL_LOADED_AT = time.time()

    feat_path = models_dir / "feature_names.joblib"
    if feat_path.exists():
        _FEATURE_NAMES = joblib.load(feat_path)

    prep_path = models_dir / "preprocessor.joblib"
    if prep_path.exists():
        _PREPROCESSOR = joblib.load(prep_path)
        logger.info("Preprocessor loaded")

    caps_path = models_dir / "outlier_caps.joblib"
    if caps_path.exists():
        _OUTLIER_CAPS = joblib.load(caps_path)

    model_name = _METADATA.get("model_name", "").lower().replace(" ", "_")
    for candidate in [model_name, "xgboost", "lightgbm", "catboost",
                      "randomforest", "extratrees", "stacking"]:
        p = models_dir / f"shap_explainer_{candidate}.joblib"
        if p.exists():
            _SHAP_EXPLAINER = joblib.load(p)
            logger.info("SHAP explainer loaded ({})", candidate)
            break

    logger.success(
        "Model loaded: {}  threshold: {}  features: {}  preprocessor: {}",
        _METADATA.get("model_name"),
        _METADATA.get("best_threshold"),
        len(_FEATURE_NAMES),
        _PREPROCESSOR is not None,
    )


def _get_shap_reasons(features: np.ndarray, n: int = 3) -> list[ShapReason]:
    if _SHAP_EXPLAINER is None:
        return []
    try:
        sv = _SHAP_EXPLAINER.shap_values(features)
        if isinstance(sv, list):
            sv = sv[1]
        sv = sv.flatten()
        top_idx = np.argsort(np.abs(sv))[::-1][:n]
        names = _FEATURE_NAMES or [f"feature_{i}" for i in range(len(sv))]
        return [
            ShapReason(
                feature=names[i] if i < len(names) else f"feature_{i}",
                shap_value=round(float(sv[i]), 4),
                direction="increases churn risk" if sv[i] > 0 else "reduces churn risk",
            )
            for i in top_idx
        ]
    except Exception as e:
        logger.warning("SHAP failed: {}", e)
        return []


def _risk_tier(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    if prob < 0.60:
        return "Medium"
    return "High"


def _raw_to_array(raw: RawCustomerRequest) -> np.ndarray:
    """Convert a RawCustomerRequest through the full preprocessing pipeline."""
    if _PREPROCESSOR is None:
        raise HTTPException(
            status_code=503,
            detail="Preprocessor not loaded — restart the API after running make train",
        )
    row = raw.model_dump(exclude={"customer_id"})
    df = pd.DataFrame([row])
    cleaned, _ = basic_clean(df.copy(), caps=_OUTLIER_CAPS)
    drop_cols = [c for c in [settings.data.target_col] if c in cleaned.columns]
    X = cleaned.drop(columns=drop_cols)
    return _PREPROCESSOR.transform(X)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting Churn API (env={})", settings.api.environment)
    _load_model()
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Cell2Cell churn prediction — 9 data fixes, 8 engineered features, "
        "SHAP explanations, business-optimised threshold.\n\n"
        "**Two prediction modes:**\n"
        "- `/predict` — pre-processed feature vector (faster)\n"
        "- `/predict/raw` — raw Cell2Cell columns (no preprocessing needed on client side)"
    ),
    version="1.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
Instrumentator(
    should_group_status_codes=False,
    excluded_handlers=["/health", "/ready", "/metrics"],
).instrument(app).expose(app)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(round((time.perf_counter() - start) * 1000, 2))
    return response


# ── Ops ───────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health():
    return HealthResponse(status="ok", environment=settings.api.environment)


@app.get("/ready", response_model=HealthResponse, tags=["Ops"])
async def ready():
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ready",
        environment=settings.api.environment,
        model_uptime_s=round(time.time() - _MODEL_LOADED_AT, 1),
    )


@app.get("/model/info", tags=["Ops"])
async def model_info():
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {
        **_METADATA,
        "n_features":        len(_FEATURE_NAMES),
        "feature_names":     _FEATURE_NAMES,
        "shap_available":    _SHAP_EXPLAINER is not None,
        "preprocessor_loaded": _PREPROCESSOR is not None,
    }


# ── Pre-processed vector endpoints ───────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["Inference (pre-processed)"])
async def predict(request: PredictRequest):
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    features  = np.array(request.features).reshape(1, -1)
    prob      = float(_MODEL.predict_proba(features)[0, 1])
    threshold = float(_METADATA.get("best_threshold", settings.model.decision_threshold))
    return PredictResponse(
        customer_id=request.customer_id,
        churn_probability=round(prob, 4),
        churn_predicted=prob >= threshold,
        risk_tier=_risk_tier(prob),
        threshold_used=threshold,
        top_reasons=_get_shap_reasons(features),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference (pre-processed)"])
async def predict_batch(request: BatchPredictRequest):
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    if len(request.rows) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 rows per request")
    X         = np.array([r.features for r in request.rows])
    probs     = _MODEL.predict_proba(X)[:, 1]
    threshold = float(_METADATA.get("best_threshold", settings.model.decision_threshold))
    predictions = [
        PredictResponse(
            customer_id=request.rows[i].customer_id,
            churn_probability=round(float(p), 4),
            churn_predicted=float(p) >= threshold,
            risk_tier=_risk_tier(float(p)),
            threshold_used=threshold,
            top_reasons=_get_shap_reasons(X[i: i + 1]),
        )
        for i, p in enumerate(probs)
    ]
    churners = sum(p.churn_predicted for p in predictions)
    return BatchPredictResponse(
        predictions=predictions,
        total=len(predictions),
        predicted_churners=churners,
        churn_rate=round(churners / len(predictions), 4),
    )


# ── Raw (un-preprocessed) endpoints ──────────────────────────────────────────

@app.post("/predict/raw", response_model=PredictResponse, tags=["Inference (raw columns)"])
async def predict_raw(request: RawCustomerRequest):
    """
    Accept a customer record in raw Cell2Cell column format.
    The API runs the full preprocessing pipeline (imputation, scaling, encoding,
    feature engineering) before scoring — no client-side preprocessing needed.
    """
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    features  = _raw_to_array(request)
    prob      = float(_MODEL.predict_proba(features)[0, 1])
    threshold = float(_METADATA.get("best_threshold", settings.model.decision_threshold))
    return PredictResponse(
        customer_id=request.customer_id,
        churn_probability=round(prob, 4),
        churn_predicted=prob >= threshold,
        risk_tier=_risk_tier(prob),
        threshold_used=threshold,
        top_reasons=_get_shap_reasons(features),
    )


@app.post("/predict/batch/raw", response_model=BatchPredictResponse, tags=["Inference (raw columns)"])
async def predict_batch_raw(request: BatchRawRequest):
    """
    Batch score up to 1,000 customers using raw Cell2Cell columns.
    """
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    if len(request.rows) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 rows per request")

    threshold = float(_METADATA.get("best_threshold", settings.model.decision_threshold))
    predictions: list[PredictResponse] = []
    for row in request.rows:
        features = _raw_to_array(row)
        prob     = float(_MODEL.predict_proba(features)[0, 1])
        predictions.append(
            PredictResponse(
                customer_id=row.customer_id,
                churn_probability=round(prob, 4),
                churn_predicted=prob >= threshold,
                risk_tier=_risk_tier(prob),
                threshold_used=threshold,
                top_reasons=_get_shap_reasons(features),
            )
        )
    churners = sum(p.churn_predicted for p in predictions)
    return BatchPredictResponse(
        predictions=predictions,
        total=len(predictions),
        predicted_churners=churners,
        churn_rate=round(churners / len(predictions), 4),
    )


# ── Error handler ─────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    logger.error("Unhandled: {}", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def serve() -> None:
    uvicorn.run(
        "churn.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        log_level=settings.api.log_level,
        reload=settings.api.environment == "development",
    )


if __name__ == "__main__":
    serve()
