from __future__ import annotations

import json
from datetime import date
from typing import Any

import pandas as pd

from .backtest import run_backtest, summarize_backtest
from .config import DATA_DIR, ROOT
from .utils import iso_date, safe_to_csv


OPTIMIZED_CONFIG_PATH = ROOT / "config" / "optimized_strategy.json"


def load_optimized_strategy() -> dict[str, Any]:
    if not OPTIMIZED_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(OPTIMIZED_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _window_start(end: str, months: int) -> str:
    end_ts = pd.Timestamp(end)
    return (end_ts - pd.DateOffset(months=months)).date().isoformat()


def _objective(summary: dict[str, float]) -> float:
    cagr = float(summary.get("cagr", 0.0))
    mdd = float(summary.get("mdd", 0.0))
    sharpe = float(summary.get("sharpe", 0.0))
    exposure = float(summary.get("exposure", 0.0))
    drawdown_penalty = max(0.15, 1.0 + mdd)
    return cagr * drawdown_penalty + 0.15 * sharpe + 0.05 * exposure


def optimize_strategy_windows(
    end: str | None = None,
    windows: list[int] | None = None,
    top_ns: list[int] | None = None,
    score_thresholds: list[float] | None = None,
    stop_multipliers: list[float] | None = None,
    take_profit_pairs: list[tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    end = end or iso_date(date.today())
    windows = windows or [12, 18, 24]
    top_ns = top_ns or [5, 7]
    score_thresholds = score_thresholds or [2.0, 2.5, 3.0]
    stop_multipliers = stop_multipliers or [2.0, 2.5, 3.0]
    take_profit_pairs = take_profit_pairs or [(0.25, 0.08), (0.30, 0.10), (0.35, 0.12)]

    rows: list[dict[str, Any]] = []
    for months in windows:
        start = _window_start(end, months)
        for top_n in top_ns:
            for threshold in score_thresholds:
                for stop in stop_multipliers:
                    for tp_trigger, tp_trailing in take_profit_pairs:
                        try:
                            result, holdings = run_backtest(
                                start=start,
                                end=end,
                                top_n=top_n,
                                score_threshold=threshold,
                                stop_multiplier=stop,
                                take_profit_trigger_pct=tp_trigger,
                                take_profit_trailing_pct=tp_trailing,
                            )
                            summary = summarize_backtest(result)
                        except Exception as exc:
                            rows.append(
                                {
                                    "window_months": months,
                                    "start": start,
                                    "end": end,
                                    "top_n": top_n,
                                    "score_threshold": threshold,
                                    "stop_multiplier": stop,
                                    "take_profit_trigger_pct": tp_trigger,
                                    "take_profit_trailing_pct": tp_trailing,
                                    "error": str(exc),
                                }
                            )
                            continue
                        rows.append(
                            {
                                "window_months": months,
                                "start": start,
                                "end": end,
                                "top_n": top_n,
                                "score_threshold": threshold,
                                "stop_multiplier": stop,
                                "take_profit_trigger_pct": tp_trigger,
                                "take_profit_trailing_pct": tp_trailing,
                                "avg_positions": float(result["active_positions"].mean()),
                                "rebalance_count": int(holdings["date"].nunique()) if not holdings.empty else 0,
                                "objective": _objective(summary),
                                "error": "",
                                **summary,
                            }
                        )

    report = pd.DataFrame(rows)
    best: dict[str, Any] = {}
    if not report.empty:
        valid = report[report["error"].fillna("") == ""].copy()
        if not valid.empty:
            best_row = valid.sort_values(["objective", "sharpe", "cagr", "mdd"], ascending=[False, False, False, False]).iloc[0]
            best = {
                "selected_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
                "selection_rule": "max objective = CAGR drawdown-adjusted + Sharpe/exposure bonus",
                "window_months": int(best_row["window_months"]),
                "start": str(best_row["start"]),
                "end": str(best_row["end"]),
                "top_n": int(best_row["top_n"]),
                "score_threshold": float(best_row["score_threshold"]),
                "stop_multiplier": float(best_row["stop_multiplier"]),
                "take_profit_trigger_pct": float(best_row["take_profit_trigger_pct"]),
                "take_profit_trailing_pct": float(best_row["take_profit_trailing_pct"]),
                "objective": float(best_row["objective"]),
                "cagr": float(best_row["cagr"]),
                "mdd": float(best_row["mdd"]),
                "sharpe": float(best_row["sharpe"]),
                "exposure": float(best_row["exposure"]),
            }

    out = DATA_DIR / "backtests" / f"strategy_window_optimization_{end}.csv"
    safe_to_csv(report, out, index=False, encoding="utf-8-sig")
    if best:
        OPTIMIZED_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        OPTIMIZED_CONFIG_PATH.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, best
