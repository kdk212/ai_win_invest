from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import load_config
from .data import get_fundamentals, get_macro_proxy_prices, get_market_snapshot, get_name, get_ohlcv
from .optimizer import load_optimized_strategy
from .utils import iso_date, zscore


def macro_regime_score() -> tuple[float, str]:
    cfg = load_config()["macro"]
    prices = get_macro_proxy_prices(cfg["proxies"], period="1y")
    if prices.empty or len(prices) < 80:
        return 0.0, "macro neutral: proxy data unavailable"
    aligned = prices.ffill()
    returns_20 = aligned.pct_change(20, fill_method=None).iloc[-1]
    returns_60 = aligned.pct_change(60, fill_method=None).iloc[-1]
    risk_on = returns_20.get("nasdaq", 0) + returns_20.get("sp500", 0) + returns_20.get("korea", 0)
    ai = 0.7 * returns_20.get("ai_semiconductor", 0) + 0.3 * returns_60.get("ai_semiconductor", 0)
    dollar_pressure = returns_20.get("dollar", 0) + returns_20.get("usdkrw", 0)
    score = float(0.45 * risk_on + 0.45 * ai - 0.25 * dollar_pressure)
    sector_names = {
        "ai_semiconductor": "AI/semiconductor",
        "technology": "technology",
        "industrials": "industrials",
        "financials": "financials",
        "healthcare": "healthcare",
        "consumer_discretionary": "consumer discretionary",
        "consumer_staples": "consumer staples",
        "energy": "energy",
        "materials": "materials",
        "utilities": "utilities",
        "defense_aerospace": "defense/aerospace",
        "korea": "Korea",
    }
    sector_scores = {}
    for key in sector_names:
        if key in aligned.columns:
            sector_scores[key] = float(0.65 * returns_20.get(key, 0) + 0.35 * returns_60.get(key, 0))
    leaders = sorted(sector_scores.items(), key=lambda item: item[1], reverse=True)[:3]
    leader_label = ", ".join(sector_names[key] for key, value in leaders if pd.notna(value))
    if score > 0.05:
        label = f"global risk-on; leading proxies: {leader_label}"
    elif score < -0.05:
        label = f"risk-off or dollar pressure; relative leaders: {leader_label}"
    else:
        label = f"macro neutral; relative leaders: {leader_label}"
    return score, label


def _fundamental_value(fundamentals: pd.DataFrame, ticker: str) -> tuple[float, float, float, float]:
    per = pbr = dividend = np.nan
    if ticker in fundamentals.index:
        row = fundamentals.loc[ticker]
        per = float(row.get("PER", np.nan))
        pbr = float(row.get("PBR", np.nan))
        dividend = float(row.get("DIV", np.nan))
    valuation = 0.0
    if pd.notna(per) and 0 < per < 25:
        valuation += 0.5
    if pd.notna(pbr) and 0 < pbr < 3:
        valuation += 0.4
    if pd.notna(dividend) and dividend > 0:
        valuation += 0.1
    return per, pbr, dividend, valuation


def _theme_label(ticker: str) -> str:
    by_theme = {
        "AI/semiconductor supply chain": {"000660", "005930", "005935", "009150", "009155", "353200", "036930", "042700", "058470", "000990"},
        "AI optical/network infrastructure": {"010170", "007660", "222800", "356680"},
        "EV/battery supply chain": {"373220", "006400", "086520", "247540", "066970", "003670"},
        "Electronics component supply chain": {"011070", "001820", "090460", "033240"},
        "Software/automation": {"307950", "012510", "035720", "035420", "036570", "259960"},
        "Holding/financial": {"034730", "105560", "055550", "086790", "138040", "001740"},
        "Consumer/industrial": {"066570", "005380", "000270", "000150", "012450", "267250"},
    }
    for label, tickers in by_theme.items():
        if ticker in tickers:
            return label
    return "Other momentum"


def _risk_plan(close: pd.Series) -> dict[str, float]:
    daily = close.pct_change()
    vol20 = float(daily.tail(20).std() * np.sqrt(252))
    vol60 = float(daily.tail(60).std() * np.sqrt(252))
    recent_peak = float(close.tail(60).max())
    latest = float(close.iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma60 = float(close.rolling(60).mean().iloc[-1])
    ma10 = float(close.rolling(10).mean().iloc[-1])
    daily_risk = float(daily.tail(20).std())
    cfg = load_config()["backtest"]
    optimized = load_optimized_strategy()
    stop_multiplier = float(optimized.get("stop_multiplier", cfg.get("stop_multiplier", 2.2)))
    volatility_stop = min(max(stop_multiplier * daily_risk * np.sqrt(5), 0.055), 0.18)
    support_stop_price = max(ma20 * 0.965, ma60 * 0.94)
    volatility_stop_price = latest * (1 - volatility_stop)
    stop_price = max(support_stop_price, volatility_stop_price)
    stop_loss_pct = max(0.035, min((latest - stop_price) / latest, 0.20))
    stop_price = latest * (1 - stop_loss_pct)
    trailing_stop_pct = min(max(1.7 * daily_risk * np.sqrt(5), 0.045), 0.15)
    warning_stop_pct = min(max(1.25 * daily_risk * np.sqrt(5), 0.035), 0.10)
    warning_price = max(ma10 * 0.985, latest * (1 - warning_stop_pct))
    warning_stop_pct = max(0.025, min((latest - warning_price) / latest, 0.12))
    warning_price = latest * (1 - warning_stop_pct)
    drawdown_from_peak = latest / recent_peak - 1 if recent_peak else 0.0
    risk_score = vol60 + max(0.0, -drawdown_from_peak) + max(0.0, (latest / ma20 - 1) - 0.18)
    take_profit_trigger_pct = float(optimized.get("take_profit_trigger_pct", cfg.get("take_profit_trigger_pct", 0.35)))
    take_profit_trailing_pct = float(optimized.get("take_profit_trailing_pct", cfg.get("take_profit_trailing_pct", 0.12)))
    take_profit_trigger_price = latest * (1 + take_profit_trigger_pct)
    take_profit_trailing_price = take_profit_trigger_price * (1 - take_profit_trailing_pct)
    return {
        "vol20": vol20,
        "vol60": vol60,
        "stop_loss_pct": float(stop_loss_pct),
        "stop_price": float(stop_price),
        "warning_price": float(warning_price),
        "warning_stop_pct": float(warning_stop_pct),
        "trailing_stop_pct": float(trailing_stop_pct),
        "take_profit_trigger_pct": take_profit_trigger_pct,
        "take_profit_trigger_price": float(take_profit_trigger_price),
        "take_profit_trailing_pct": take_profit_trailing_pct,
        "take_profit_trailing_price": float(take_profit_trailing_price),
        "drawdown_from_60d_peak": float(drawdown_from_peak),
        "risk_score": float(risk_score),
    }


def score_stock(ticker: str, start: str, end: str, fundamentals: pd.DataFrame) -> dict[str, float | str] | None:
    df = get_ohlcv(ticker, start, end)
    if df.empty or len(df) < 130:
        return None
    close = df["close"].astype(float)
    trading_value = df.get("trading_value", close * df["volume"]).astype(float)
    latest_close = float(close.iloc[-1])
    avg_trading_value_20 = float(trading_value.tail(20).mean())
    mom20 = close.iloc[-1] / close.iloc[-21] - 1
    mom60 = close.iloc[-1] / close.iloc[-61] - 1
    mom120 = close.iloc[-1] / close.iloc[-121] - 1
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    trend = latest_close / ma60 - 1 + (ma20 / ma60 - 1)
    volatility = close.pct_change().tail(60).std() * np.sqrt(252)
    hit_ratio_20 = float((close.pct_change().tail(20) > 0).mean())
    pullback_resilience = float(close.iloc[-1] / close.tail(20).max() - close.tail(20).min() / close.tail(20).max())
    acceleration = float(mom20 - (mom60 - mom20) / 2)
    overextension = float(latest_close / ma20 - 1)
    parabolic_penalty = max(0.0, mom20 - 0.80) + max(0.0, mom60 - 2.20) + max(0.0, overextension - 0.35)
    turnover_change = float(trading_value.tail(5).mean() / trading_value.tail(60).mean() - 1)
    per, pbr, dividend, valuation = _fundamental_value(fundamentals, ticker)
    risk = _risk_plan(close)
    return {
        "ticker": ticker,
        "name": get_name(ticker),
        "theme": _theme_label(ticker),
        "price_date": df.index[-1].date().isoformat(),
        "close": latest_close,
        "avg_trading_value_20": avg_trading_value_20,
        "mom20": float(mom20),
        "mom60": float(mom60),
        "mom120": float(mom120),
        "trend": float(trend),
        "volatility": float(volatility),
        "hit_ratio_20": hit_ratio_20,
        "pullback_resilience": pullback_resilience,
        "acceleration": acceleration,
        "overextension": overextension,
        "parabolic_penalty": float(parabolic_penalty),
        "turnover_change": turnover_change,
        "per": per,
        "pbr": pbr,
        "dividend": dividend,
        "valuation": valuation,
        **risk,
    }


def build_recommendations(as_of: date | None = None, top_n: int | None = None, min_raw_score: float | None = None) -> pd.DataFrame:
    cfg = load_config()
    as_of = as_of or date.today()
    end = iso_date(as_of)
    start = iso_date(as_of - timedelta(days=430))
    optimized = load_optimized_strategy()
    if top_n is None and optimized.get("top_n"):
        top_n = int(optimized["top_n"])
    top_n = top_n or int(cfg["portfolio"]["top_n"])
    if min_raw_score is None and optimized.get("score_threshold") is not None:
        min_raw_score = float(optimized["score_threshold"])
    min_raw_score = float(min_raw_score if min_raw_score is not None else cfg["portfolio"].get("min_raw_score", -999))
    min_value = float(cfg["portfolio"]["min_trading_value_krw"])
    universe_size = int(cfg["portfolio"]["universe_size"])
    fundamentals = get_fundamentals(end)
    snapshot = get_market_snapshot(end)
    if snapshot.empty:
        tickers = []
    else:
        value_col = "trading_value" if "trading_value" in snapshot.columns else "market_cap"
        snapshot[value_col] = pd.to_numeric(snapshot[value_col], errors="coerce")
        eligible = snapshot[snapshot[value_col] >= min_value].sort_values(value_col, ascending=False)
        tickers = eligible.head(universe_size).index.astype(str).tolist()
    rows = []
    for ticker in tickers:
        try:
            row = score_stock(ticker, start, end, fundamentals)
        except Exception:
            row = None
        if row and row["avg_trading_value_20"] >= min_value:
            rows.append(row)
    scores = pd.DataFrame(rows)
    if scores.empty:
        return scores
    weights = cfg["strategy"]
    raw_score = (
        weights["momentum_20d_weight"] * zscore(scores["mom20"])
        + weights["momentum_60d_weight"] * zscore(scores["mom60"])
        + weights["momentum_120d_weight"] * zscore(scores["mom120"])
        + weights["trend_weight"] * zscore(scores["trend"])
        + weights["liquidity_weight"] * zscore(np.log1p(scores["avg_trading_value_20"]))
        + weights["volatility_weight"] * zscore(scores["volatility"])
        + weights["valuation_weight"] * zscore(scores["valuation"])
    )
    forward_quality = (
        0.22 * zscore(scores["hit_ratio_20"])
        + 0.18 * zscore(scores["pullback_resilience"])
        + 0.18 * zscore(scores["acceleration"])
        + 0.12 * zscore(scores["turnover_change"].clip(-1, 5))
        - 0.22 * zscore(scores["risk_score"])
        - 0.18 * zscore(scores["drawdown_from_60d_peak"].abs())
        - 0.00 * zscore(scores["parabolic_penalty"])
    )
    scores["forward_quality_score"] = forward_quality
    scores["score"] = raw_score + forward_quality
    scores["score_100"] = (scores["score"].rank(pct=True) * 100).clip(0, 100)
    macro_score, macro_label = macro_regime_score()
    scores["macro_score"] = macro_score
    scores["macro_label"] = macro_label
    scores["as_of"] = scores["price_date"].mode().iloc[0] if not scores["price_date"].mode().empty else end
    scores = scores.sort_values("score", ascending=False)
    scores = scores[scores["score"] >= min_raw_score].head(top_n).reset_index(drop=True)
    scores["rank"] = scores.index + 1
    return scores[
        [
            "as_of",
            "rank",
            "ticker",
            "name",
            "theme",
            "price_date",
            "close",
            "score",
            "score_100",
            "forward_quality_score",
            "macro_score",
            "macro_label",
            "mom20",
            "mom60",
            "mom120",
            "trend",
            "volatility",
            "hit_ratio_20",
            "pullback_resilience",
            "acceleration",
            "overextension",
            "parabolic_penalty",
            "turnover_change",
            "avg_trading_value_20",
            "warning_price",
            "warning_stop_pct",
            "stop_price",
            "stop_loss_pct",
            "take_profit_trigger_price",
            "take_profit_trigger_pct",
            "take_profit_trailing_price",
            "take_profit_trailing_pct",
            "trailing_stop_pct",
            "risk_score",
            "per",
            "pbr",
            "dividend",
        ]
    ]
