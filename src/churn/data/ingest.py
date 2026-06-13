"""
src/churn/data/ingest.py
────────────────────────
Downloads Cell2Cell dataset (Duke University / Teradata) from Kaggle.
Falls back to local path if already present.

Dataset: https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom
  • cell2celltrain.csv — 51,047 rows × 58 cols
  • cell2celltest.csv  — 20,000 rows × 58 cols
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import typer
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from configs.settings import settings

app = typer.Typer(rich_markup_mode="rich")


def _already_downloaded() -> bool:
    return (
        settings.data.train_path.exists()
        and settings.data.test_path.exists()
    )


def download_via_kaggle() -> None:
    """Pull dataset using the Kaggle CLI (requires ~/.kaggle/kaggle.json)."""
    import subprocess

    logger.info("Downloading Cell2Cell dataset from Kaggle…")
    settings.data.raw_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "kaggle", "datasets", "download",
        "-d", "jpacse/datasets-for-churn-telecom",
        "-p", str(settings.data.raw_dir),
        "--unzip",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603

    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError("Kaggle download failed. Check your API key in ~/.kaggle/kaggle.json")

    logger.success("✅ Dataset downloaded to {}", settings.data.raw_dir)


def load_from_local(train_path: Path, test_path: Path) -> None:
    """Copy user-provided CSVs into the data/raw directory."""
    settings.data.raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(train_path, settings.data.train_path)
    shutil.copy(test_path, settings.data.test_path)
    logger.success("✅ Files copied to {}", settings.data.raw_dir)


@app.command()
def main(
    train: Path = typer.Option(None, help="Local path to cell2celltrain.csv"),
    test:  Path = typer.Option(None, help="Local path to cell2celltest.csv"),
) -> None:
    """
    [bold green]Ingest Cell2Cell dataset[/bold green].

    Priority:
      1. Already exists in data/raw/ → skip
      2. --train / --test provided   → copy local files
      3. Kaggle API available        → download automatically
    """
    if _already_downloaded():
        logger.info("Dataset already present — skipping download.")
        return

    if train and test:
        load_from_local(train, test)
    else:
        download_via_kaggle()


if __name__ == "__main__":
    app()
