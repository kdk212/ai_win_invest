from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_DIR
from .data import get_ohlcv
from .utils import iso_date, safe_to_csv


def latest_recommendation_file() -> Path | None:
    files = sorted((DATA_DIR / "recommendations").glob("recommendations_*.csv"))
    return files[-1] if files else None


def update_recommendation_returns() -> pd.DataFrame:
    records = []
    for path in sorted((DATA_DIR / "recommendations").glob("recommendations_*.csv")):
        recs = pd.read_csv(path, dtype={"ticker": str})
        for _, row in recs.iterrows():
            start = row["as_of"]
            ticker = row["ticker"]
            try:
                prices = get_ohlcv(ticker, start, iso_date(pd.Timestamp.today().date()), cache=False)
            except Exception:
                continue
            if prices.empty:
                continue
            base = float(row["close"])
            latest = float(prices["close"].iloc[-1])
            records.append(
                {
                    "as_of": start,
                    "rank": int(row["rank"]),
                    "ticker": ticker,
                    "name": row["name"],
                    "entry_close": base,
                    "latest_close": latest,
                    "return_since_recommendation": latest / base - 1,
                    "days_tracked": len(prices),
                }
            )
    df = pd.DataFrame(records)
    out = DATA_DIR / "performance" / "recommendation_returns.csv"
    safe_to_csv(df, out, index=False, encoding="utf-8-sig")
    return df
