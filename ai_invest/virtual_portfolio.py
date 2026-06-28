from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .config import DATA_DIR
from .data import get_index_returns, get_ohlcv
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
    stop_loss_pct: float
    take_profit_trigger_price: float | None
    take_profit_trigger_pct: float
    take_profit_trailing_pct: float
    shares: float = field(init=False)
    peak_price: float = field(init=False)
    sell_date: date | None = None
    sell_price: float | None = None
    sell_reason: str | None = None

    def __post_init__(self) -> None:
        self.shares = self.allocation / self.buy_price if self.buy_price else 0.0
        self.peak_price = self.buy_price
        if not self.stop_price:
            self.stop_price = self.buy_price * (1 - self.stop_loss_pct)
        if not self.take_profit_trigger_price:
            self.take_profit_trigger_price = self.buy_price * (1 + self.take_profit_trigger_pct)

    @property
    def active(self) -> bool:
        return self.sell_date is None

    def stop_level(self) -> float:
        return self.stop_price or self.buy_price * (1 - self.stop_loss_pct)

    def take_profit_trigger_level(self) -> float:
        return self.take_profit_trigger_price or self.buy_price * (1 + self.take_profit_trigger_pct)


def recommendation_files() -> list[Path]:
    return sorted((DATA_DIR / "recommendations").glob("recommendations_*.csv"))


def _to_date(value: object) -> date | None:
    try:
        return pd.Timestamp(value).date()
    except Exception:
        return None


def _read_recommendations() -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in recommendation_files():
        try:
            frame = pd.read_csv(path, dtype={"ticker": str})
        except Exception:
            continue
        if not frame.empty and {"ticker", "as_of"}.issubset(frame.columns):
            frames.append(frame)
    return frames


def _load_prices(tickers: list[str], start_date: date) -> dict[str, pd.DataFrame]:
    prices: dict[str, pd.DataFrame] = {}
    start = iso_date(start_date - timedelta(days=7))
    end = iso_date(pd.Timestamp.today().date())
    for ticker in sorted(set(tickers)):
        try:
            frame = get_ohlcv(ticker, start, end)
        except Exception:
            frame = pd.DataFrame()
        if frame.empty:
            continue
        frame = frame.copy()
        frame.index = pd.to_datetime(frame.index)
        for col in ("open", "high", "low", "close"):
            if col in frame:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["open", "high", "low", "close"], how="any")
        if not frame.empty:
            prices[ticker] = frame
    return prices


def _next_buy_open(price: pd.DataFrame, rec_date: date, start_date: date) -> tuple[date, float] | None:
    rows = price[price.index.date >= max(rec_date, start_date)]
    if rows.empty:
        return None
    return rows.index[0].date(), float(rows.iloc[0]["open"])


def _float_or_none(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _buy_events(recommendations: list[pd.DataFrame], prices: dict[str, pd.DataFrame], start_date: date) -> list[Lot]:
    lots: list[Lot] = []
    for recs in recommendations:
        rec_date = _to_date(recs["as_of"].iloc[0])
        if rec_date is None:
            continue
        daily = recs.copy()
        daily["ticker"] = daily["ticker"].astype(str).str.zfill(6)
        daily["score"] = pd.to_numeric(daily.get("score"), errors="coerce")
        daily = daily[daily["ticker"].isin(prices)].sort_values("score", ascending=False)
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
                    stop_price=_float_or_none(row.get("stop_price")),
                    stop_loss_pct=_float_or_none(row.get("stop_loss_pct")) or 0.10,
                    take_profit_trigger_price=_float_or_none(row.get("take_profit_trigger_price")),
                    take_profit_trigger_pct=_float_or_none(row.get("take_profit_trigger_pct")) or 0.35,
                    take_profit_trailing_pct=_float_or_none(row.get("take_profit_trailing_pct")) or 0.10,
                )
            )
    return sorted(lots, key=lambda lot: (lot.buy_date, lot.ticker, lot.rec_date))


def _trade_dates(prices: dict[str, pd.DataFrame], start_date: date) -> list[date]:
    dates: set[date] = set()
    for frame in prices.values():
        dates.update(day for day in frame.index.date if day >= start_date)
    return sorted(dates)


def _price_on(frame: pd.DataFrame | None, trading_date: date) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    rows = frame[frame.index.date == trading_date]
    return None if rows.empty else rows.iloc[0]


def _close_ticker(lots: list[Lot], ticker: str, trading_date: date, sell_price: float, reason: str) -> None:
    for lot in lots:
        if lot.ticker == ticker and lot.active and lot.buy_date <= trading_date:
            lot.sell_date = trading_date
            lot.sell_price = sell_price
            lot.sell_reason = reason


def _apply_latest_risk_settings(lots: list[Lot], latest: Lot) -> None:
    for lot in lots:
        if lot.ticker != latest.ticker or not lot.active:
            continue
        lot.rec_date = latest.rec_date
        lot.stop_price = latest.stop_price
        lot.stop_loss_pct = latest.stop_loss_pct
        lot.take_profit_trigger_price = latest.take_profit_trigger_price
        lot.take_profit_trigger_pct = latest.take_profit_trigger_pct
        lot.take_profit_trailing_pct = latest.take_profit_trailing_pct


def _daily_snapshot(lots: list[Lot], prices: dict[str, pd.DataFrame], trading_date: date) -> dict[str, object]:
    contributed = sum(lot.allocation for lot in lots if lot.buy_date <= trading_date)
    realized = sum((lot.sell_price or 0.0) * lot.shares - lot.allocation for lot in lots if lot.sell_date and lot.sell_date <= trading_date)
    active_cost = 0.0
    market_value = 0.0
    for lot in lots:
        if not lot.active or lot.buy_date > trading_date:
            continue
        price = _price_on(prices.get(lot.ticker), trading_date)
        if price is None:
            continue
        active_cost += lot.allocation
        market_value += lot.shares * float(price["close"])
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
        "active_names": len({lot.ticker for lot in lots if lot.active and lot.buy_date <= trading_date}),
    }


def _holdings(lots: list[Lot], prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    active = [lot for lot in lots if lot.active]
    active_cost = sum(lot.allocation for lot in active)
    rows: list[dict[str, object]] = []
    for ticker in sorted({lot.ticker for lot in active}):
        ticker_lots = [lot for lot in active if lot.ticker == ticker]
        cost = sum(lot.allocation for lot in ticker_lots)
        shares = sum(lot.shares for lot in ticker_lots)
        frame = prices.get(ticker)
        current_date = frame.index[-1].date().isoformat() if frame is not None and not frame.empty else ""
        current_price = float(frame.iloc[-1]["close"]) if frame is not None and not frame.empty else None
        market_value = shares * current_price if current_price is not None else None
        rows.append(
            {
                "ticker": ticker,
                "name": ticker_lots[0].name,
                "first_buy_date": min(lot.buy_date for lot in ticker_lots).isoformat(),
                "lots": len(ticker_lots),
                "weight": cost / active_cost if active_cost else 0.0,
                "avg_buy_price": cost / shares if shares else 0.0,
                "current_price": current_price,
                "current_price_date": current_date,
                "market_value": market_value,
                "return_pct": (market_value / cost - 1) if market_value is not None and cost else None,
                "stop_price": max((lot.stop_level() for lot in ticker_lots), default=None),
                "take_profit_trigger_price": min((lot.take_profit_trigger_level() for lot in ticker_lots), default=None),
            }
        )
    return pd.DataFrame(rows).sort_values("weight", ascending=False) if rows else pd.DataFrame()


def _closed(lots: list[Lot]) -> pd.DataFrame:
    rows = []
    for lot in lots:
        if lot.active:
            continue
        rows.append(
            {
                "ticker": lot.ticker,
                "name": lot.name,
                "buy_date": lot.buy_date.isoformat(),
                "buy_price": lot.buy_price,
                "sell_date": lot.sell_date.isoformat() if lot.sell_date else "",
                "sell_price": lot.sell_price,
                "sell_reason": lot.sell_reason,
                "return_pct": ((lot.sell_price or 0.0) / lot.buy_price - 1) if lot.buy_price else None,
            }
        )
    return pd.DataFrame(rows).sort_values("sell_date", ascending=False) if rows else pd.DataFrame()


def _benchmark_returns(start_date: date) -> dict[str, float | str]:
    try:
        return get_index_returns(start_date)
    except Exception:
        return {
            "kospi_return": 0.0,
            "kospi_latest_date": "",
            "kosdaq_return": 0.0,
            "kosdaq_latest_date": "",
        }


def simulate_recommendation_portfolio(start_date: date = DEFAULT_START_DATE) -> dict[str, pd.DataFrame | str | float]:
    benchmarks = _benchmark_returns(start_date)
    recommendations = _read_recommendations()
    if not recommendations:
        result = _empty("저장된 추천 파일이 없습니다. daily 작업이 성공해야 가상 포트폴리오가 생성됩니다.")
        result.update(benchmarks)
        return result

    tickers = sorted({str(row["ticker"]).zfill(6) for recs in recommendations for _, row in recs.iterrows()})
    prices = _load_prices(tickers, start_date)
    if not prices:
        result = _empty("추천 파일은 있지만 가격 데이터를 불러오지 못했습니다. KRX/FinanceDataReader 접속 상태를 확인해야 합니다.")
        result.update({"recommendation_days": float(len(recommendations)), "recommendation_tickers": float(len(tickers)), **benchmarks})
        return result

    lots = _buy_events(recommendations, prices, start_date)
    if not lots:
        result = _empty("추천 파일과 가격 데이터는 있으나 2026-06-01 이후 시초가 매수로 전환된 종목이 없습니다.")
        result.update({"recommendation_days": float(len(recommendations)), "recommendation_tickers": float(len(tickers)), "priced_tickers": float(len(prices)), **benchmarks})
        return result

    pending = list(lots)
    active: list[Lot] = []
    daily_rows: list[dict[str, object]] = []
    for trading_date in _trade_dates(prices, start_date):
        while pending and pending[0].buy_date == trading_date:
            new_lot = pending.pop(0)
            active.append(new_lot)
            _apply_latest_risk_settings(active, new_lot)
        for ticker in sorted({lot.ticker for lot in active if lot.active}):
            price = _price_on(prices.get(ticker), trading_date)
            if price is None:
                continue
            low = float(price["low"])
            high = float(price["high"])
            ticker_lots = [lot for lot in active if lot.ticker == ticker and lot.active]
            for lot in ticker_lots:
                lot.peak_price = max(lot.peak_price, high)
            stop_hits = [lot.stop_level() for lot in ticker_lots if low <= lot.stop_level()]
            if stop_hits:
                _close_ticker(active, ticker, trading_date, float(max(stop_hits)), "stop_loss")
                continue
            trailing_hits = []
            for lot in ticker_lots:
                if lot.peak_price >= lot.take_profit_trigger_level():
                    trailing = lot.peak_price * (1 - lot.take_profit_trailing_pct)
                    if low <= trailing:
                        trailing_hits.append(trailing)
            if trailing_hits:
                _close_ticker(active, ticker, trading_date, float(max(trailing_hits)), "take_profit_trailing")
        daily_rows.append(_daily_snapshot(active, prices, trading_date))

    holdings = _holdings(active, prices)
    closed = _closed(active)
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
        "recommendation_days": float(len(recommendations)),
        "recommendation_tickers": float(len(tickers)),
        "priced_tickers": float(len(prices)),
        "buy_events": float(len(lots)),
        **benchmarks,
    }


def _empty(message: str) -> dict[str, pd.DataFrame | str | float]:
    return {
        "status": message,
        "holdings": pd.DataFrame(),
        "closed": pd.DataFrame(),
        "daily": pd.DataFrame(),
        "latest_date": "",
        "total_return": 0.0,
        "unrealized_return": 0.0,
        "active_count": 0.0,
        "recommendation_days": 0.0,
        "recommendation_tickers": 0.0,
        "priced_tickers": 0.0,
        "buy_events": 0.0,
        "kospi_return": 0.0,
        "kospi_latest_date": "",
        "kosdaq_return": 0.0,
        "kosdaq_latest_date": "",
    }
