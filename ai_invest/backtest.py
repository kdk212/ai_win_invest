from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import DATA_DIR, load_config
from .data import get_market_snapshot, get_price_panel
from .utils import iso_date, safe_to_csv, zscore


def _factor_frame(prices: pd.DataFrame, idx: int) -> pd.DataFrame:
    window = prices.iloc[: idx + 1]
    close = window.iloc[-1]
    daily = window.pct_change(fill_method=None)
    mom20 = close / window.iloc[-21] - 1
    mom60 = close / window.iloc[-61] - 1
    mom120 = close / window.iloc[-121] - 1
    ma20 = window.rolling(20).mean().iloc[-1]
    ma60 = window.rolling(60).mean().iloc[-1]
    trend = close / ma60 - 1 + (ma20 / ma60 - 1)
    vol60 = daily.tail(60).std() * np.sqrt(252)
    hit20 = (daily.tail(20) > 0).mean()
    peak20 = window.tail(20).max()
    trough20 = window.tail(20).min()
    peak60 = window.tail(60).max()
    pullback_resilience = close / peak20 - trough20 / peak20
    acceleration = mom20 - (mom60 - mom20) / 2
    overextension = close / ma20 - 1
    parabolic_penalty = (mom20 - 0.80).clip(lower=0) + (mom60 - 2.20).clip(lower=0) + (overextension - 0.35).clip(lower=0)
    drawdown = close / peak60 - 1
    risk_score = vol60 + (-drawdown).clip(lower=0) + ((close / ma20 - 1) - 0.18).clip(lower=0)

    frame = pd.DataFrame(
        {
            "mom20": mom20,
            "mom60": mom60,
            "mom120": mom120,
            "trend": trend,
            "vol60": vol60,
            "hit20": hit20,
            "pullback_resilience": pullback_resilience,
            "acceleration": acceleration,
            "overextension": overextension,
            "parabolic_penalty": parabolic_penalty,
            "drawdown": drawdown,
            "risk_score": risk_score,
        }
    )
    frame["score"] = (
        0.18 * zscore(frame["mom20"])
        + 0.28 * zscore(frame["mom60"])
        + 0.18 * zscore(frame["mom120"])
        + 0.18 * zscore(frame["trend"])
        + 0.18 * zscore(frame["hit20"])
        + 0.15 * zscore(frame["pullback_resilience"])
        + 0.12 * zscore(frame["acceleration"])
        - 0.22 * zscore(frame["vol60"])
        - 0.18 * zscore(frame["risk_score"])
        - 0.00 * zscore(frame["parabolic_penalty"])
    )
    return frame.replace([np.inf, -np.inf], np.nan)


def _dynamic_stop_pct(prices: pd.DataFrame, idx: int, holdings: list[str], multiplier: float) -> pd.Series:
    daily = prices.iloc[: idx + 1][holdings].pct_change(fill_method=None)
    daily_risk = daily.tail(20).std()
    return (multiplier * daily_risk * np.sqrt(5)).clip(lower=0.045, upper=0.18)


def _max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def run_backtest(
    start: str,
    end: str | None = None,
    top_n: int = 10,
    stop_multiplier: float | None = None,
    score_threshold: float | None = None,
    take_profit_trigger_pct: float | None = None,
    take_profit_trailing_pct: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()["backtest"]
    end = end or iso_date(date.today())
    warmup_start = iso_date(pd.to_datetime(start).date() - timedelta(days=260))

    snapshot = get_market_snapshot(end)
    if snapshot.empty:
        raise RuntimeError("Could not load the stock universe for backtesting.")
    value_col = "trading_value" if "trading_value" in snapshot.columns else "market_cap"
    snapshot[value_col] = pd.to_numeric(snapshot[value_col], errors="coerce")
    universe_size = int(load_config()["portfolio"]["universe_size"])
    tickers = snapshot.sort_values(value_col, ascending=False).head(universe_size).index.astype(str).tolist()

    prices = get_price_panel(tickers, warmup_start, end).dropna(axis=1, thresh=180)
    if prices.empty:
        raise RuntimeError("Could not load enough price data for backtesting.")

    returns = prices.pct_change(fill_method=None).fillna(0)
    rebalance_days = int(cfg["rebalance_days"])
    cost = float(cfg["transaction_cost_bps"]) / 10000
    stop_multiplier = float(stop_multiplier if stop_multiplier is not None else cfg.get("stop_multiplier", 2.2))
    score_threshold = score_threshold if score_threshold is not None else cfg.get("score_threshold")
    take_profit_trigger_pct = float(take_profit_trigger_pct if take_profit_trigger_pct is not None else cfg.get("take_profit_trigger_pct", 0.35))
    take_profit_trailing_pct = float(take_profit_trailing_pct if take_profit_trailing_pct is not None else cfg.get("take_profit_trailing_pct", 0.12))
    start_ts = pd.to_datetime(start)
    equity = 1.0
    active_holdings: set[str] = set()
    target_holdings: set[str] = set()
    stopped_until_rebalance: set[str] = set()
    profit_taken_until_rebalance: set[str] = set()
    entry_prices = pd.Series(dtype=float)
    peak_prices = pd.Series(dtype=float)
    daily_rows = []
    holding_rows = []

    for idx in range(130, len(prices) - 1):
        current_date = prices.index[idx]
        if current_date < start_ts:
            continue

        should_rebalance = len(daily_rows) == 0 or len(daily_rows) % rebalance_days == 0
        if should_rebalance:
            factors = _factor_frame(prices, idx)
            ranked = factors["score"].dropna().sort_values(ascending=False)
            selected = ranked[ranked >= float(score_threshold)].head(top_n) if score_threshold is not None else ranked.head(top_n)
            target_holdings = set(selected.index)
            stopped_until_rebalance = set()
            profit_taken_until_rebalance = set()
            new_active = target_holdings.copy()
            turnover = len(new_active.symmetric_difference(active_holdings)) / max(top_n, 1)
            equity *= 1 - cost * turnover
            active_holdings = new_active
            entry_prices = prices.loc[current_date, list(active_holdings)].astype(float)
            peak_prices = entry_prices.copy()
            for ticker in active_holdings:
                holding_rows.append({"date": current_date.date().isoformat(), "ticker": ticker, "score": float(ranked[ticker]), "entry_price": float(entry_prices[ticker])})

        if active_holdings:
            active_list = sorted(active_holdings)
            today_prices = prices.loc[current_date, active_list].astype(float)
            peak_prices.loc[active_list] = pd.concat([peak_prices.loc[active_list], today_prices], axis=1).max(axis=1)
            stop_pct = _dynamic_stop_pct(prices, idx, active_list, stop_multiplier)
            drawdown_from_entry = today_prices / entry_prices.loc[active_list] - 1
            drawdown_from_peak = today_prices / peak_prices.loc[active_list] - 1
            stop_mask = (drawdown_from_entry <= -stop_pct) | (drawdown_from_peak <= -(stop_pct * 0.85))
            take_profit_mask = (peak_prices.loc[active_list] / entry_prices.loc[active_list] - 1 >= take_profit_trigger_pct) & (drawdown_from_peak <= -take_profit_trailing_pct)
            stopped = set(stop_mask[stop_mask].index)
            profit_taken = set(take_profit_mask[take_profit_mask].index) - stopped
            exited = stopped | profit_taken
            stopped_until_rebalance |= stopped
            profit_taken_until_rebalance |= profit_taken
            active_holdings -= exited

        next_date = prices.index[idx + 1]
        next_return = returns.loc[next_date, sorted(active_holdings)].mean() if active_holdings else 0.0
        equity *= 1 + float(next_return)
        daily_rows.append(
            {
                "date": next_date.date().isoformat(),
                "equity": equity,
                "daily_return": float(next_return),
                "active_positions": len(active_holdings),
                "max_positions": top_n,
                "stopped_positions": len(stopped_until_rebalance),
                "profit_taken_positions": len(profit_taken_until_rebalance),
            }
        )

    result = pd.DataFrame(daily_rows)
    holdings_df = pd.DataFrame(holding_rows)
    if result.empty:
        raise RuntimeError("Backtest result is empty. Use a longer date range.")

    threshold_label = "top" if score_threshold is None else f"score{float(score_threshold):.1f}"
    suffix = f"{start}_{end}_{threshold_label}_max{top_n}_stop{stop_multiplier:.1f}_tp{take_profit_trigger_pct:.2f}_trail{take_profit_trailing_pct:.2f}"
    out = DATA_DIR / "backtests" / f"backtest_{suffix}.csv"
    safe_to_csv(result, out, index=False, encoding="utf-8-sig")
    safe_to_csv(holdings_df, out.with_name(out.stem + "_holdings.csv"), index=False, encoding="utf-8-sig")
    return result, holdings_df


def optimize_stop_loss(start: str, end: str | None = None, top_n: int = 10) -> pd.DataFrame:
    rows = []
    for multiplier in [1.4, 1.7, 2.0, 2.2, 2.5, 2.8, 3.2]:
        result, _ = run_backtest(start, end=end, top_n=top_n, stop_multiplier=multiplier)
        rows.append({"stop_multiplier": multiplier, **summarize_backtest(result)})
    return pd.DataFrame(rows).sort_values(["sharpe", "mdd", "cagr"], ascending=[False, False, False])


def optimize_score_threshold(start: str, end: str | None = None, top_n: int = 10) -> pd.DataFrame:
    rows = []
    for threshold in [1.5, 2.0, 2.5, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5]:
        result, holdings = run_backtest(start, end=end, top_n=top_n, score_threshold=threshold)
        rows.append({"score_threshold": threshold, "avg_positions": float(result["active_positions"].mean()), "max_positions": int(result["active_positions"].max()), "rebalance_count": int(holdings["date"].nunique()) if not holdings.empty else 0, **summarize_backtest(result)})
    return pd.DataFrame(rows).sort_values(["sharpe", "cagr", "mdd"], ascending=[False, False, False])


def optimize_take_profit(start: str, end: str | None = None, top_n: int = 7) -> pd.DataFrame:
    rows = []
    for trigger, trailing in [(0.25, 0.08), (0.25, 0.10), (0.30, 0.10), (0.30, 0.12), (0.35, 0.10), (0.35, 0.12), (0.40, 0.12), (0.40, 0.15), (0.50, 0.15)]:
        result, holdings = run_backtest(start, end=end, top_n=top_n, score_threshold=2.0, take_profit_trigger_pct=trigger, take_profit_trailing_pct=trailing)
        rows.append({"take_profit_trigger_pct": trigger, "take_profit_trailing_pct": trailing, "avg_positions": float(result["active_positions"].mean()), **summarize_backtest(result)})
    return pd.DataFrame(rows).sort_values(["sharpe", "cagr", "mdd"], ascending=[False, False, False])


def summarize_backtest(result: pd.DataFrame) -> dict[str, float]:
    equity = result["equity"]
    daily = result["daily_return"]
    total_return = float(equity.iloc[-1] - 1)
    years = max(len(result) / 252, 1 / 252)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1)
    mdd = _max_drawdown(equity)
    sharpe = float((daily.mean() / daily.std()) * np.sqrt(252)) if daily.std() else 0.0
    denominator = float(result["max_positions"].iloc[0]) if "max_positions" in result else 10.0
    exposure = float(result["active_positions"].mean() / denominator) if "active_positions" in result else 1.0
    return {"total_return": total_return, "cagr": cagr, "mdd": mdd, "sharpe": sharpe, "exposure": exposure}
