"""
tests/integration/test_api.py
───────────────────────────────
Integration tests for the FastAPI app.
Uses httpx.AsyncClient — no actual model or preprocessor needed (mocked).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_model() -> MagicMock:
    """Stub model that returns fixed probabilities (single row)."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    return model


@pytest.fixture()
def mock_batch_model() -> MagicMock:
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2], [0.5, 0.5]])
    return model


@pytest.fixture()
def mock_preprocessor() -> MagicMock:
    """Stub preprocessor that returns a fixed array regardless of input."""
    prep = MagicMock()
    prep.transform.return_value = np.zeros((1, 57))
    return prep


@pytest.fixture()
async def client(mock_model: MagicMock, mock_preprocessor: MagicMock) -> AsyncClient:
    """AsyncClient with the FastAPI app, model + preprocessor pre-injected."""
    with (
        patch("churn.api.main._load_model"),
        patch("churn.api.main._MODEL", mock_model),
        patch("churn.api.main._PREPROCESSOR", mock_preprocessor),
        patch("churn.api.main._OUTLIER_CAPS", {}),
        patch("churn.api.main._METADATA", {"best_threshold": 0.40}),
    ):
        from churn.api.main import app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac


# ── Health ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_ok(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_ready_ok(client: AsyncClient) -> None:
    resp = await client.get("/ready")
    assert resp.status_code == 200


# ── Pre-processed vector predict ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_predict_returns_probability(client: AsyncClient) -> None:
    payload = {"customer_id": "C123", "features": [0.1] * 57}
    resp = await client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "churn_probability" in body
    assert 0.0 <= body["churn_probability"] <= 1.0


@pytest.mark.asyncio
async def test_predict_high_prob_is_churner(client: AsyncClient) -> None:
    """Model stub returns prob=0.7 → above threshold 0.40 → churn_predicted=True."""
    payload = {"customer_id": "C999", "features": [0.5] * 57}
    resp = await client.post("/predict", json=payload)
    assert resp.json()["churn_predicted"] is True
    assert resp.json()["risk_tier"] == "High"


@pytest.mark.asyncio
async def test_predict_nan_feature_rejected(client: AsyncClient) -> None:
    # None serialises as JSON null; Pydantic no_nan validator raises 422.
    payload = {"customer_id": "C_bad", "features": [None] + [0.1] * 56}
    resp = await client.post("/predict", json=payload)
    assert resp.status_code == 422


# ── Batch predict (pre-processed) ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_batch_predict_response_shape(
    client: AsyncClient, mock_batch_model: MagicMock
) -> None:
    with patch("churn.api.main._MODEL", mock_batch_model):
        payload = {
            "rows": [
                {"customer_id": f"C{i}", "features": [0.1] * 57}
                for i in range(3)
            ]
        }
        resp = await client.post("/predict/batch", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 3
        assert len(body["predictions"]) == 3


@pytest.mark.asyncio
async def test_batch_predict_over_limit(client: AsyncClient) -> None:
    payload = {
        "rows": [{"customer_id": f"C{i}", "features": [0.1] * 57} for i in range(1001)]
    }
    resp = await client.post("/predict/batch", json=payload)
    assert resp.status_code == 422  # Pydantic max_length rejects before handler


# ── Raw endpoint (/predict/raw) ───────────────────────────────────────────────

@pytest.fixture()
def raw_customer_payload() -> dict:
    """Minimal valid RawCustomerRequest payload."""
    return {
        "customer_id":     "C_raw_001",
        "MonthlyRevenue":  55.0,
        "MonthlyMinutes":  400.0,
        "DroppedCalls":    3.0,
        "MonthsInService": 18.0,
        "HandsetPrice":    150.0,
        "CreditRating":    "Good",
        "MadeCallToRetentionTeam": "No",
        "Occupation":      "Professional",
        "MaritalStatus":   "Married",
        "ChildrenInHH":    "No",
    }


@pytest.mark.asyncio
async def test_predict_raw_returns_probability(
    client: AsyncClient, raw_customer_payload: dict, mock_preprocessor: MagicMock
) -> None:
    """POST /predict/raw should call preprocessor and return a valid PredictResponse."""
    # Ensure preprocessor returns correct shape for a single row
    mock_preprocessor.transform.return_value = np.zeros((1, 57))

    with patch("churn.api.main._PREPROCESSOR", mock_preprocessor):
        resp = await client.post("/predict/raw", json=raw_customer_payload)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "churn_probability" in body
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["customer_id"] == "C_raw_001"
    assert body["risk_tier"] in {"Low", "Medium", "High"}


@pytest.mark.asyncio
async def test_predict_raw_high_prob_flags_churn(
    client: AsyncClient, raw_customer_payload: dict, mock_preprocessor: MagicMock
) -> None:
    """Verify churn_predicted=True when model probability (0.7) exceeds threshold (0.40)."""
    mock_preprocessor.transform.return_value = np.zeros((1, 57))

    with patch("churn.api.main._PREPROCESSOR", mock_preprocessor):
        resp = await client.post("/predict/raw", json=raw_customer_payload)

    body = resp.json()
    assert body["churn_predicted"] is True
    assert body["threshold_used"] == pytest.approx(0.40)


@pytest.mark.asyncio
async def test_predict_raw_empty_payload_still_works(
    client: AsyncClient, mock_preprocessor: MagicMock
) -> None:
    """An all-null payload should not raise a 422 (nulls are handled by imputation)."""
    mock_preprocessor.transform.return_value = np.zeros((1, 57))

    with patch("churn.api.main._PREPROCESSOR", mock_preprocessor):
        resp = await client.post("/predict/raw", json={})

    # 200 or 500 are both acceptable here (500 means basic_clean raised on truly empty df)
    # The key invariant is that it does NOT return 422 (schema rejection)
    assert resp.status_code != 422, "Empty raw payload should not be schema-rejected"


@pytest.mark.asyncio
async def test_predict_raw_no_preprocessor_returns_503(client: AsyncClient) -> None:
    """If preprocessor was not loaded, /predict/raw must return 503."""
    with patch("churn.api.main._PREPROCESSOR", None):
        resp = await client.post("/predict/raw", json={"customer_id": "C1"})
    assert resp.status_code == 503
