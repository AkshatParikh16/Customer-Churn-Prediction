"""
src/churn/api/schemas.py
─────────────────────────
Pydantic v2 request / response models.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class CustomerFeatures(BaseModel):
    """Single customer — pre-processed feature vector + optional ID."""
    customer_id: str | None = Field(None, description="Optional customer identifier")
    features: list[float] = Field(
        ...,
        description=(
            "Ordered feature vector matching the trained preprocessor. "
            "Use GET /model/info to retrieve the exact feature ordering."
        ),
        min_length=1,
    )

    @field_validator("features")
    @classmethod
    def no_nan(cls, v: list[float]) -> list[float]:
        import math
        for i, val in enumerate(v):
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Feature[{i}] is NaN or Inf — impute before calling the API")
        return v


# Alias kept for backward compat
PredictRequest = CustomerFeatures


class BatchPredictRequest(BaseModel):
    rows: list[CustomerFeatures] = Field(..., max_length=1000)


# ── Raw (un-preprocessed) request ─────────────────────────────────────────────

class RawCustomerRequest(BaseModel):
    """
    Single customer in raw Cell2Cell column format.
    All fields are optional at the API boundary — missing values are handled
    by the same preprocessing pipeline used during training (median/mode impute).
    Pass null for truly unknown fields.
    """
    customer_id: Optional[str] = None

    # Numeric billing / usage
    MonthlyRevenue: Optional[float] = None
    MonthlyMinutes: Optional[float] = None
    TotalRecurringCharge: Optional[float] = None
    DirectorAssistedCalls: Optional[float] = None
    OverageMinutes: Optional[float] = None
    RoamingCalls: Optional[float] = None
    PercChangeMinutes: Optional[float] = None
    PercChangeRevenues: Optional[float] = None

    # Call quality
    DroppedCalls: Optional[float] = None
    BlockedCalls: Optional[float] = None
    UnansweredCalls: Optional[float] = None
    CustomerCareCalls: Optional[float] = None
    ThreewayCalls: Optional[float] = None
    ReceivedCalls: Optional[float] = None
    OutboundCalls: Optional[float] = None
    InboundCalls: Optional[float] = None
    PeakCallsInOut: Optional[float] = None
    OffPeakCallsInOut: Optional[float] = None

    # Retention
    RetentionCalls: Optional[float] = None
    RetentionOffersAccepted: Optional[float] = None
    MadeCallToRetentionTeam: Optional[str] = None  # "Yes" / "No"

    # Referrals / offers
    ReferralsMadeBySubscriber: Optional[float] = None
    IncomeGroup: Optional[float] = None
    OwnsMotorcycle: Optional[str] = None
    AdjustmentsToCreditRating: Optional[float] = None

    # Demographic / household
    AgeHH1: Optional[float] = None
    AgeHH2: Optional[float] = None
    ChildrenInHH: Optional[str] = None            # "Yes" / "No"
    HandsetRefurbished: Optional[str] = None
    HandsetWebCapable: Optional[str] = None
    TruckOwner: Optional[str] = None
    RVOwner: Optional[str] = None
    Homeownership: Optional[str] = None
    BuysViaMailOrder: Optional[str] = None
    RespondsToMailOffers: Optional[str] = None
    OptOutMailings: Optional[str] = None
    NonUSTravel: Optional[str] = None
    OwnsComputer: Optional[str] = None
    HasCreditCard: Optional[str] = None
    NewCellphoneUser: Optional[str] = None
    NotNewCellphoneUser: Optional[str] = None
    MaritalStatus: Optional[str] = None
    Occupation: Optional[str] = None
    CreditRating: Optional[str] = None
    PrizmCode: Optional[str] = None
    Region: Optional[str] = None
    Ethnicity: Optional[str] = None

    # Handset / equipment
    HandsetPrice: Optional[float] = None
    CurrentEquipmentDays: Optional[float] = None
    HandsetModels: Optional[float] = None
    AccessReturns: Optional[float] = None
    DroppedBlockedCalls: Optional[float] = None
    UnansweredCalls2: Optional[float] = None
    MonthsInService: Optional[float] = None
    UniqueSubs: Optional[float] = None
    ActiveSubs: Optional[float] = None
    ServiceArea: Optional[str] = None
    Phones: Optional[float] = None
    Models: Optional[float] = None
    PosDaysChangedDuring30Mins: Optional[float] = None
    CompletedMVM: Optional[float] = None
    PeakCallsInOut2: Optional[float] = None


class BatchRawRequest(BaseModel):
    rows: list[RawCustomerRequest] = Field(..., max_length=1000)


# ── Shared response models ─────────────────────────────────────────────────────

class ShapReason(BaseModel):
    feature: str
    shap_value: float
    direction: str


class PredictResponse(BaseModel):
    customer_id: str | None
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_predicted: bool
    risk_tier: str = Field(..., pattern="^(Low|Medium|High)$")
    threshold_used: float
    top_reasons: list[ShapReason] = []


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    total: int
    predicted_churners: int
    churn_rate: float


class HealthResponse(BaseModel):
    status: str
    environment: str
    model_uptime_s: float | None = None

