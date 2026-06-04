from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .config import DATA_DIR
from .data import get_ohlcv
from .utils import iso_date


DEFAULT_START_DATE = date(2026, 6, 1)


@dataclass
class Lot:
    ticker: str
    name: str
    rec_date: date
    buy_date: date
    buy_price: float
    allocation: float
    stop_price: float | None
    take_profit_trigger_price: float | None
    take_profit_trailing_pct: float
    shares: float = field(init=False)
    peak_price: float = field(init=False)
    sell_date: date | None = None
    sell_price: float | None = None
    sell_reason: str | None = None

    def __post_init__(self) -> None:
        self.shares = self.allocation / self.buy_price if self.buy_price else 0.0
        self.peak_price = self.buy_price

    @property
    def active(self) -> bool:
        return self.sell_date is None


def recommendation_files() -> list[Path]:
    return sorted((DATA_DIR / "recommendations").glob("recommendations_*.csv"))


def _to_date(value: object) -> date | None:
    try:
        return pd.Timestamp(value).date()
    except Exception:
        return None


def _load_recommendations() -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in recommendation_files():
        try:
            frame = pd.read_csv(path, dtype={"ticker": str})
        except Exception:
            continue
        if not frame.empty and {"ticker", "as_of"}.issubset(frame.columns):
            frames.append(frame)
    return frames


def _price_start(start_date: date) -> str:
    return iso_date(start_date - timedelta(days=7))


def _price_end() -> str:
    return iso_date(pd.Timestamp.today().date())


def _load_price_data(tickers: list[str], start_date: date) -> dict[str, pd.DataFrame]:
    prices: dict[str, pd.DataFrame] = {}
    for ticker in sorted(set(tickers)):
        try:
            frame = get_ohlcv(ticker, _price_start(start_date), _price_end())
        except Exception:
            frame = pd.DataFrame()
        if frame.empty:
            continue
        frame = frame.copy()
        frame.index = pd.to_datetime(frame.index)
        for column in ("open", "high", "low", "close"):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        prices[ticker] = frame.dropna(subset=["open", "high", "low", "close"], how="any")
    return prices


def _next_buy_open(price: pd.DataFrame, rec_date: date, start_date: date) -> tuple[date, float] | None:
    if price.empty:
        return None
    buy_from = max(rec_date, start_date)
    rows = price[price.index.date >= buy_from]
    if rows.empty:
        return None
    first = rows.iloc[0]
    return rows.index[0].date(), float(first["open"])


def _optional_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _build_buy_events(recommendations: list[pd.DataFrame], prices: dict[str, pd.DataFrame], start_date: date) -> list[Lot]:
    lots: list[Lot] = []
    for recs in recommendations:
        rec_date = _to_date(recs["as_of"].iloc[0])
        if rec_date is None:
            continue
        daily = recs.copy()
        daily["score"] = pd.to_numeric(daily.get("score"), errors="coerce")
        daily = daily.sort_values("score", ascending=False)
        daily = daily[daily["ticker"].astype(str).str.zfill(6).isin(prices)]
        if daily.empty:
            continue
        allocation = 1.0 / len(daily)
        for _, row in daily.iterrows():
            ticker = str(row["ticker"]).zfill(6)
            buy = _next_buy_open(prices[ticker], rec_date, start_date)
            if buy is None:
                continue
            buy_date, buy_price = buy
            lots.append(
                Lot(
                    ticker=ticker,
                    name=str(row.get("name", ticker)),
                    rec_date=rec_date,
                    buy_date=buy_date,
                    buy_price=buy_price,
                    allocation=allocation,
                    stop_price=_optional_float(row.get("stop_price")),
                    take_profit_trigger_price=_optional_float(row.get("take_profit_trigger_price")),
                    take_profit_trailing_pct=_optional_float(row.get("take_profit_trailing_pct")) or 0.10,
                )
            )
    return sorted(lots, key=lambda lot: (lot.buy_date, lot.ticker, lot.rec_date))


def _trade_dates(prices: dict[str, pd.DataFrame], start_date: date) -> list[date]:
    dates = set()
    for frame in prices.values():
        dates.update(index_date for index_date in frame.index.date if index_date >= start_date)
    return sorted(dates)


def _price_on(frame: pd.DataFrame | None, trading_date: date) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    rows = frame[frame.index.date == trading_date]
    return None if rows.empty else rows.iloc[0]


def _current_close(prices: dict[str, pd.DataFrame], ticker: str) -> tuple[date | None, float | None]:
    frame = prices.get(ticker)
    if frame is None or frame.empty:
        return None, None
    return frame.index[-1].date(), float(frame.iloc[-1]["close"])


def _close_ticker(lots: list[Lot], ticker: str, trading_date: date, sell_price: float, reason: str) -> None:
    for lot in lots:
        if lot.ticker == ticker and lot.active and lot.buy_date <= trading_date:
            lot.sell_date = trading_date
            lot.sell_price = sell_price
            lot.sell_reason = reason


def _daily_snapshot(active: list[Lot], prices: dict[str, pd.DataFrame], trading_date: date) -> dict[str, object]:
    contributed = sum(lot.allocation for lot in active if lot.buy_date <= trading_date)
    realized = sum((lot.sell_price or 0.0) * lot.shares - lot.allocation for lot in active if lot.sell_date and lot.sell_date <= trading_date)
    market_value = 0.0
    active_cost = 0.0
    for lot in active:
        if not lot.active or lot.buy_date > trading_date:
            continue
        day_price = _price_on(prices.get(lot.ticker), trading_date)
        if day_price is None:
            continue
        active_cost += lot.allocation
        market_value += lot.shares * float(day_price["close"])
    unrealized = market_value - active_cost
    return {
        "date": trading_date.isoformat(),
        "contributed": contributed,
        "active_cost": active_cost,
        "market_value": market_value,
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "total_pnl": realized + unrealized,
        "total_return": (realized + unrealized) / contributed if contributed else 0.0,
        "unrealized_return": unrealized / active_cost if active_cost else 0.0,
        "active_names": len({lot.ticker for lot in active if lot.active and lot.buy_date <= trading_date}),
    }


def _holding_rows(lots: list[Lot], prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    active_lots = [lot for lot in lots if lot.active]
    active_cost = sum(lot.allocation for lot in active_lots)
    rows: list[dict[str, object]] = []
    for ticker in sorted({lot.ticker for lot in active_lots}):
        ticker_lots = [lot for lot in active_lots if lot.ticker == ticker]
        cost = sum(lot.allocation for lot in ticker_lots)
        shares = sum(lot.shares for lot in ticker_lots)
        avg_buy = cost / shares if shares else 0.0
        price_date, current_price = _current_close(prices, ticker)
        market_value = shares * current_price if current_price is not None else None
        rows.append(
            {
                "ticker": ticker,
                "name": ticker_lots[0].name,
                "first_buy_date": min(lot.buy_date for lot in ticker_lots).isoformat(),
                "lots": len(ticker_lots),
                "weight": cost / active_cost if active_cost else 0.0,
                "avg_buy_price": avg_buy,
                "current_price": current_price,
                "current_price_date": price_date.isoformat() if price_date else "",
                "market_value": market_value,
                "return_pct": (market_value / cost - 1) if market_value is not None and cost else None,
                "stop_price": max((lot.stop_price or 0.0) for lot in ticker_lots) or None,
                "take_profit_trigger_price": min((lot.take_profit_trigger_price for lot in ticker_lots if lot.take_profit_trigger_price), default=None),
            }
        )
    return pd.DataFrame(rows).sort_values("weight", ascending=False) if rows else pd.DataFrame()


def _closed_rows(lots: list[Lot]) -> pd.DataFrame:
    rows = []
    for lot in lots:
        if lot.active:
            continue
        rows.append(
            {
                "ticker": lot.ticker,
                "name": lot.name,
                "rec_date": lot.rec_date.isoformat(),
                "buy_date": lot.buy_date.isoformat(),
                "buy_price": lot.buy_price,
                "sell_date": lot.sell_date.isoformat() if lot.sell_date else "",
                "sell_price": lot.sell_price,
                "sell_reason": lot.sell_reason,
                "return_pct": ((lot.sell_price or 0.0) / lot.buy_price - 1) if lot.buy_price else None,
            }
        )
    return pd.DataFrame(rows).sort_values("sell_date", ascending=False) if rows else pd.DataFrame()


def simulate_recommendation_portfolio(start_date: date = DEFAULT_START_DATE) -> dict[str, pd.DataFrame | str | float]:
    recommendations = _load_recommendations()
    if not recommendations:
        return _empty_result("No saved recommendation files yet.")
    tickers = sorted({str(row["ticker"]).zfill(6) for recs in recommendations for _, row in recs.iterrows()})
    prices = _load_price_data(tickers, start_date)
    if not prices:
        return _empty_result("Price data is unavailable.")
    lots = _build_buy_events(recommendations, prices, start_date)
    if not lots:
        return _empty_result("No recommendation could be converted into a buy event.")

    pending = list(lots)
    active: list[Lot] = []
    daily_rows: list[dict[str, object]] = []
    for trading_date in _trade_dates(prices, start_date):
        while pending and pending[0].buy_date == trading_date:
            active.append(pending.pop(0))
        for ticker in sorted({lot.ticker for lot in active if lot.active}):
            day_price = _price_on(prices.get(ticker), trading_date)
            if day_price is None:
                continue
            low = float(day_price["low"])
            high = float(day_price["high"])
            ticker_lots = [lot for lot in active if lot.ticker == ticker and lot.active]
            for lot in ticker_lots:
                lot.peak_price = max(lot.peak_price, high)
            stop_hits = [lot.stop_price for lot in ticker_lots if lot.stop_price and low <= lot.stop_price]
            if stop_hits:
                _close_ticker(active, ticker, trading_date, float(max(stop_hits)), "stop_loss")
                continue
            trailing_hits = []
            for lot in ticker_lots:
                trigger = lot.take_profit_trigger_price
                if trigger and lot.peak_price >= trigger:
                    trailing_price = lot.peak_price * (1 - lot.take_profit_trailing_pct)
                    if low <= trailing_price:
                        trailing_hits.append(trailing_price)
            if trailing_hits:
                _close_ticker(active, ticker, trading_date, float(max(trailing_hits)), "take_profit_trailing")
        daily_rows.append(_daily_snapshot(active, prices, trading_date))

    holdings = _holding_rows(active, prices)
    closed = _closed_rows(active)
    daily = pd.DataFrame(daily_rows)
    latest = daily.iloc[-1].to_dict() if not daily.empty else {}
    return {
        "status": "ok",
        "holdings": holdings,
        "closed": closed,
        "daily": daily,
        "latest_date": str(latest.get("date", "")),
        "total_return": float(latest.get("total_return", 0.0) or 0.0),
        "unrealized_return": float(latest.get("unrealized_return", 0.0) or 0.0),
        "active_count": float(len(holdings)),
    }


def _empty_result(message: str) -> dict[str, pd.DataFrame | str | float]:
    return {
        "status": message,
        "holdings": pd.DataFrame(),
        "closed": pd.DataFrame(),
        "daily": pd.DataFrame(),
        "latest_date": "",
        "total_return": 0.0,
        "unrealized_return": 0.0,
        "active_count": 0.0,
    }
