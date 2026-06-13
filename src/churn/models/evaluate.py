"""
src/churn/models/evaluate.py
─────────────────────────────
Post-training evaluation:
  • Classification report + confusion matrix
  • ROC-AUC / PR-AUC curves (saved to reports/)
  • SHAP global feature importance (beeswarm + bar)
  • SHAP local explanation for a single customer
  • Threshold optimisation (maximise F1 or Recall)
  • All artifacts logged to the active MLflow run
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import shap
from loguru import logger
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from configs.settings import settings

REPORTS = settings.model.models_dir.parent / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


# ── Threshold optimisation ────────────────────────────────────────────────────

def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> float:
    """
    Sweep thresholds [0.1 … 0.9] and return the one that maximises
    the chosen metric: 'f1' | 'recall' | 'precision'.
    Cell2Cell business context: missing a churner (FN) is more costly
    than a false alarm (FP), so 'recall' or a low threshold is preferred.
    """
    thresholds = np.linspace(0.10, 0.90, 80)
    best_thresh, best_score = 0.5, 0.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            from sklearn.metrics import recall_score
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            from sklearn.metrics import precision_score
            score = precision_score(y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_thresh = float(t)

    logger.info("Best threshold ({}) = {:.3f}  (score={:.4f})", metric, best_thresh, best_score)
    return best_thresh


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split: str = "val",
) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred),
        display_labels=["No Churn", "Churn"],
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {split}")
    plt.tight_layout()
    path = REPORTS / f"confusion_matrix_{split}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix → {}", path)
    return path


# ── ROC + PR curves ───────────────────────────────────────────────────────────

def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    split: str = "val",
) -> tuple[Path, Path]:
    # ROC
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name="XGBoost")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title(f"ROC Curve — {split} (AUC={roc_auc_score(y_true, y_prob):.4f})")
    plt.tight_layout()
    roc_path = REPORTS / f"roc_curve_{split}.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax, name="XGBoost")
    ap = average_precision_score(y_true, y_prob)
    ax.set_title(f"Precision-Recall — {split} (AP={ap:.4f})")
    plt.tight_layout()
    pr_path = REPORTS / f"pr_curve_{split}.png"
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)

    logger.info("Saved ROC → {}  |  PR → {}", roc_path, pr_path)
    return roc_path, pr_path


# ── SHAP global importance ────────────────────────────────────────────────────

def plot_shap_importance(
    model: object,
    X: np.ndarray,
    feature_names: list[str],
    max_display: int = 20,
) -> tuple[Path, Path]:
    """
    Generate SHAP beeswarm + bar chart.
    Uses TreeExplainer — fast for XGBoost / LightGBM.
    Samples up to 2,000 rows for speed.
    """
    logger.info("Computing SHAP values (sample up to 2 000 rows)…")
    sample_size = min(2000, X.shape[0])
    rng = np.random.default_rng(settings.model.random_state)
    idx = rng.choice(X.shape[0], sample_size, replace=False)
    X_sample = X[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # If binary classification returns list, take positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Beeswarm
    fig, _ = plt.subplots(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title("SHAP Beeswarm — Top Feature Impacts")
    plt.tight_layout()
    bee_path = REPORTS / "shap_beeswarm.png"
    fig.savefig(bee_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Bar (mean |SHAP|)
    fig, _ = plt.subplots(figsize=(8, 5))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        max_display=max_display,
        plot_type="bar",
        show=False,
    )
    plt.title("SHAP Feature Importance (mean |SHAP value|)")
    plt.tight_layout()
    bar_path = REPORTS / "shap_bar.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.success("SHAP plots saved → {} | {}", bee_path, bar_path)
    return bee_path, bar_path


# ── SHAP local (single customer) ─────────────────────────────────────────────

def explain_single_prediction(
    model: object,
    x_row: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """
    Return top-10 SHAP features driving one customer's churn prediction.
    Use this in the API for explainable predictions.
    """
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(x_row.reshape(1, -1))
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_vals = shap_vals.flatten()
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:10]
    return {feature_names[i]: round(float(shap_vals[i]), 5) for i in top_idx}


# ── Full evaluation pipeline ──────────────────────────────────────────────────

def run_full_evaluation(
    model: object,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    log_to_mlflow: bool = True,
) -> dict[str, object]:
    """
    Run the complete evaluation suite and optionally log everything to MLflow.
    Returns a dict with all computed metrics.
    """
    threshold = settings.model.decision_threshold
    results: dict[str, object] = {}

    for split, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        if len(y) == 0:
            continue
        y_prob = model.predict_proba(X)[:, 1]  # type: ignore[union-attr]
        y_pred = (y_prob >= threshold).astype(int)

        auc   = roc_auc_score(y, y_prob)
        ap    = average_precision_score(y, y_prob)
        report = classification_report(y, y_pred, target_names=["No Churn", "Churn"])

        logger.info("\n── {} split ──\n{}", split.upper(), report)
        results[f"{split}_roc_auc"] = auc
        results[f"{split}_avg_precision"] = ap

        cm_path = plot_confusion_matrix(y, y_pred, split)
        roc_path, pr_path = plot_roc_pr_curves(y, y_prob, split)

        if log_to_mlflow:
            mlflow.log_metric(f"{split}_roc_auc", auc)
            mlflow.log_metric(f"{split}_avg_precision", ap)
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(roc_path))
            mlflow.log_artifact(str(pr_path))

    # SHAP (on val set)
    shap_bee, shap_bar = plot_shap_importance(model, X_val, feature_names)
    if log_to_mlflow:
        mlflow.log_artifact(str(shap_bee))
        mlflow.log_artifact(str(shap_bar))

    logger.success("✅ Full evaluation complete — reports saved to {}", REPORTS)
    return results


if __name__ == "__main__":
    from churn.data.preprocess import prepare_datasets
    data = prepare_datasets()
    model = joblib.load(settings.model.models_dir / "churn_xgb_prod.joblib")
    run_full_evaluation(
        model,
        data["X_val"],  data["y_val"],
        data["X_test"], data["y_test"],
        data["feature_names"],
        log_to_mlflow=False,
    )
