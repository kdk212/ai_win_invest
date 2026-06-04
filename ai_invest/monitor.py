from __future__ import annotations

import json
from datetime import date
from typing import Any

import pandas as pd

from .config import DATA_DIR
from .optimizer import load_optimized_strategy, optimize_strategy_two_stage
from .utils import safe_to_csv
from .virtual_portfolio import DEFAULT_START_DATE, simulate_recommendation_portfolio


MONITOR_PATH = DATA_DIR / "performance" / "strategy_monitor.json"


def _expected_return(cagr: float, start_date: date, latest_date: str | None) -> tuple[float, int]:
    if not latest_date:
        return 0.0, 0
    days = max((pd.Timestamp(latest_date).date() - start_date).days, 0)
    if days == 0:
        return 0.0, 0
    return float((1 + cagr) ** (days / 365.25) - 1), days


def evaluate_strategy_performance(auto_optimize: bool = False) -> dict[str, Any]:
    optimized = load_optimized_strategy()
    virtual = simulate_recommendation_portfolio(DEFAULT_START_DATE)
    latest_date = str(virtual.get("latest_date") or "")
    actual_return = float(virtual.get("total_return", 0.0) or 0.0)
    optimized_cagr = float(optimized.get("cagr", 0.0) or 0.0)
    expected_return, elapsed_days = _expected_return(optimized_cagr, DEFAULT_START_DATE, latest_date)

    shortfall = actual_return - expected_return
    ratio = actual_return / expected_return if expected_return > 0 else None
    needs_review = bool(
        optimized
        and elapsed_days >= 5
        and expected_return > 0
        and (shortfall < -0.05 or (ratio is not None and ratio < 0.5))
    )

    review_result: dict[str, Any] = {}
    bootstrap_required = bool(auto_optimize and not optimized)
    if bootstrap_required:
        _, best = optimize_strategy_two_stage()
        review_result = best
        optimized = best or optimized
    elif auto_optimize and needs_review:
        _, best = optimize_strategy_two_stage()
        review_result = best

    result: dict[str, Any] = {
        "checked_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "latest_date": latest_date,
        "elapsed_days": elapsed_days,
        "actual_total_return": actual_return,
        "expected_return_from_backtest_cagr": expected_return,
        "shortfall": shortfall,
        "actual_to_expected_ratio": ratio,
        "bootstrap_required": bootstrap_required,
        "needs_review": needs_review,
        "optimized_strategy": optimized,
        "auto_optimized": bool(review_result),
        "new_strategy": review_result,
    }

    MONITOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    MONITOR_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    safe_to_csv(pd.DataFrame([result]), DATA_DIR / "performance" / "strategy_monitor_latest.csv", index=False, encoding="utf-8-sig")
    return result
