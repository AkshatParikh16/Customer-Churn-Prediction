"""
scripts/predict.py
───────────────────
CLI for offline batch predictions. Runs the full preprocessing pipeline
(basic_clean → preprocessor.transform) before calling model.predict_proba,
so it accepts raw Cell2Cell CSV files without any prior preparation.

Usage:
    # Score a raw CSV and write predictions alongside input
    uv run python scripts/predict.py batch data/raw/cell2celltest.csv

    # Single customer (JSON of raw feature values keyed by column name)
    uv run python scripts/predict.py single '{"MonthlyRevenue": 45.5, ...}'
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from configs.settings import settings
from churn.data.preprocess import basic_clean

app = typer.Typer(rich_markup_mode="rich")
console = Console()


def _load_artifacts() -> tuple[object, object, object, dict]:
    """Load model, preprocessor, feature names, and outlier caps."""
    models_dir = Path(settings.model.models_dir)

    meta_path = models_dir / "model_metadata.joblib"
    if not meta_path.exists():
        logger.error("model_metadata.joblib not found in {}. Run `make train` first.", models_dir)
        raise SystemExit(1)
    metadata = joblib.load(meta_path)

    model_filename = metadata.get("model_filename", "churn_xgb_prod.joblib")
    model_path = models_dir / model_filename
    if not model_path.exists():
        logger.error("Model file {} not found.", model_path)
        raise SystemExit(1)
    model = joblib.load(model_path)

    prep_path = models_dir / "preprocessor.joblib"
    if not prep_path.exists():
        logger.error("preprocessor.joblib not found. Re-run `make train` to regenerate.")
        raise SystemExit(1)
    preprocessor = joblib.load(prep_path)

    caps_path = models_dir / "outlier_caps.joblib"
    caps: dict = joblib.load(caps_path) if caps_path.exists() else {}

    logger.info(
        "Loaded: {} | threshold={} | ROC-AUC={}",
        metadata.get("model_name"), metadata.get("best_threshold"), metadata.get("roc_auc"),
    )
    return model, preprocessor, metadata, caps


def _preprocess(df: pd.DataFrame, preprocessor: object, caps: dict) -> np.ndarray:
    """Run basic_clean + preprocessor on a raw DataFrame."""
    cleaned, _ = basic_clean(df.copy(), caps=caps)
    drop = [c for c in [settings.data.customer_id, settings.data.target_col] if c in cleaned.columns]
    X = cleaned.drop(columns=drop)
    return preprocessor.transform(X)  # type: ignore[union-attr]


def _risk_tier(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    if prob < 0.60:
        return "Medium"
    return "High"


@app.command("batch")
def batch_predict(
    input_path: Path = typer.Argument(..., help="Raw CSV file to score"),
    output_path: Path = typer.Option("reports/predictions.csv", help="Where to write results"),
    threshold: float = typer.Option(settings.model.decision_threshold, help="Decision threshold"),
) -> None:
    """[bold green]Batch score[/bold green] a raw Cell2Cell CSV file."""
    model, preprocessor, metadata, caps = _load_artifacts()
    threshold = float(metadata.get("best_threshold", threshold))

    logger.info("Loading {} …", input_path)
    df = pd.read_csv(input_path)
    customer_ids = df.get(settings.data.customer_id, pd.Series(range(len(df)), name="idx"))

    X = _preprocess(df, preprocessor, caps)
    probs = model.predict_proba(X)[:, 1]  # type: ignore[union-attr]
    preds = (probs >= threshold).astype(int)

    out_df = pd.DataFrame({
        settings.data.customer_id: customer_ids,
        "churn_probability":       probs.round(4),
        "churn_predicted":         preds,
        "risk_tier":               [_risk_tier(p) for p in probs],
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    churners = int(preds.sum())
    console.print(f"\n[bold]Scored {len(df):,} customers[/bold]  (model={metadata.get('model_name')})")
    console.print(f"  Predicted churners : [red]{churners:,}[/red] ({churners/len(df)*100:.1f}%)")
    console.print(f"  Output saved       : [green]{output_path}[/green]\n")

    # Top 10 highest-risk customers
    top10 = out_df.nlargest(10, "churn_probability")
    table = Table(title="Top 10 Highest-Risk Customers", show_lines=True)
    for col in out_df.columns:
        table.add_column(col, justify="right" if col == "churn_probability" else "left")
    for _, row in top10.iterrows():
        table.add_row(*[str(v) for v in row])
    console.print(table)


@app.command("single")
def single_predict(
    features_json: str = typer.Argument(
        ..., help="JSON object of raw feature values, e.g. '{\"MonthlyRevenue\": 45.5, ...}'"
    ),
    customer_id: str = typer.Option("unknown", help="Customer identifier"),
    threshold: float = typer.Option(settings.model.decision_threshold),
) -> None:
    """[bold green]Predict churn[/bold green] for a single customer (raw feature JSON)."""
    model, preprocessor, metadata, caps = _load_artifacts()
    threshold = float(metadata.get("best_threshold", threshold))

    row = json.loads(features_json)
    df = pd.DataFrame([row])
    X = _preprocess(df, preprocessor, caps)

    prob = float(model.predict_proba(X)[0, 1])  # type: ignore[union-attr]
    churn = prob >= threshold
    tier = _risk_tier(prob)

    console.print(f"\n[bold]Customer:[/bold] {customer_id}")
    console.print(f"  Churn probability : [{'red' if churn else 'green'}]{prob:.4f}[/]")
    console.print(f"  Predicted churn   : [{'red bold' if churn else 'green'}]{churn}[/]")
    console.print(f"  Risk tier         : {tier}\n")


if __name__ == "__main__":
    app()
