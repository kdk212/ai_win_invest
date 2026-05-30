from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import pandas as pd


def ymd(value: date | datetime | str) -> str:
    if isinstance(value, str):
        return value.replace("-", "")
    return value.strftime("%Y%m%d")


def iso_date(value: date | datetime | str) -> str:
    if isinstance(value, str):
        if len(value) == 8 and value.isdigit():
            return f"{value[:4]}-{value[4:6]}-{value[6:]}"
        return value
    return value.strftime("%Y-%m-%d")


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value * 100:,.2f}%"


def krw(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    abs_value = abs(float(value))
    if abs_value >= 100_000_000:
        return f"{value / 100_000_000:,.1f}eok KRW"
    if abs_value >= 10_000:
        return f"{value / 10_000:,.1f}man KRW"
    return f"{value:,.0f} KRW"


def zscore(series: pd.Series) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan)
    std = clean.std(skipna=True)
    if std is None or math.isclose(float(std), 0.0) or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (clean - clean.mean(skipna=True)) / std


def safe_to_csv(df: pd.DataFrame, path, **kwargs) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, **kwargs)
        return True
    except OSError:
        return False
