from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .config import DATA_DIR
from .data import get_name, get_ohlcv
from .utils import iso_date


DB_PATH = DATA_DIR / "portfolio.db"


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_portfolio_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                name TEXT NOT NULL,
                buy_date TEXT NOT NULL,
                buy_price REAL NOT NULL,
                quantity REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'holding',
                sell_date TEXT,
                sell_price REAL,
                memo TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def add_position(ticker: str, buy_date: str, buy_price: float, quantity: float, memo: str = "") -> None:
    init_portfolio_db()
    ticker = ticker.strip().zfill(6)
    name = get_name(ticker)
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO positions (ticker, name, buy_date, buy_price, quantity, memo)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ticker, name, buy_date, float(buy_price), float(quantity), memo.strip()),
        )


def close_position(position_id: int, sell_date: str, sell_price: float) -> None:
    init_portfolio_db()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE positions
               SET status = 'closed', sell_date = ?, sell_price = ?, updated_at = CURRENT_TIMESTAMP
             WHERE id = ?
            """,
            (sell_date, float(sell_price), int(position_id)),
        )


def reopen_position(position_id: int) -> None:
    init_portfolio_db()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE positions
               SET status = 'holding', sell_date = NULL, sell_price = NULL, updated_at = CURRENT_TIMESTAMP
             WHERE id = ?
            """,
            (int(position_id),),
        )


def list_positions() -> pd.DataFrame:
    try:
        init_portfolio_db()
        with _connect() as conn:
            rows = conn.execute("SELECT * FROM positions ORDER BY status DESC, buy_date DESC, id DESC").fetchall()
    except sqlite3.Error:
        return pd.DataFrame()
    return pd.DataFrame([dict(row) for row in rows])


def latest_close(ticker: str) -> tuple[float | None, str | None]:
    end = iso_date(date.today())
    start = iso_date(date.today() - timedelta(days=20))
    try:
        prices = get_ohlcv(ticker, start, end, cache=True)
    except Exception:
        return None, None
    if prices.empty or "close" not in prices:
        return None, None
    return float(prices["close"].iloc[-1]), prices.index[-1].date().isoformat()


def portfolio_snapshot() -> pd.DataFrame:
    positions = list_positions()
    if positions.empty:
        return positions

    rows = []
    for row in positions.to_dict("records"):
        current_price, price_date = latest_close(str(row["ticker"]))
        exit_price = row.get("sell_price") if row.get("status") == "closed" and pd.notna(row.get("sell_price")) else current_price
        buy_value = float(row["buy_price"]) * float(row["quantity"])
        current_value = float(exit_price) * float(row["quantity"]) if exit_price else None
        pnl = current_value - buy_value if current_value is not None else None
        pnl_pct = pnl / buy_value if pnl is not None and buy_value else None
        row.update(
            {
                "current_price": current_price,
                "current_price_date": price_date,
                "market_value": current_value,
                "buy_value": buy_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def db_path() -> Path:
    return DB_PATH
