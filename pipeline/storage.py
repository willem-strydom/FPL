"""Thin parquet read/write helpers so every module stores data the same way."""
import logging
from pathlib import Path

import pandas as pd

from pipeline.config import PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)


def _resolve(path: Path | str) -> Path:
    return Path(path)


def save_raw(df: pd.DataFrame, name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logger.info("saved raw %-30s %6d rows -> %s", name, len(df), path)
    return path


def save_processed(df: pd.DataFrame, name: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logger.info("saved processed %-30s %6d rows -> %s", name, len(df), path)
    return path


def load_raw(name: str) -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / f"{name}.parquet")


def load_processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
