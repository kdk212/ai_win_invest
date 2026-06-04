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


def _pick_best(report: pd.DataFrame, end: str, phase: str) -> dict[str, Any]:
    if report.empty:
        return {}
    valid = report[report["error"].fillna("") == ""].copy()
    if valid.empty:
        return {}
    best_row = valid.sort_values(["objective", "sharpe", "cagr", "mdd"], ascending=[False, False, False, False]).iloc[0]
    return {
        "selected_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "selection_phase": phase,
        "selection_rule": "max objective = CAGR drawdown-adjusted + Sharpe/exposure bonus",
        "window_months": int(best_row["window_months"]),
        "start": str(best_row["start"]),
        "end": end,
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


def optimize_strategy_windows(
    end: str | None = None,
    windows: list[int] | None = None,
    top_ns: list[int] | None = None,
    score_thresholds: list[float] | None = None,
    stop_multipliers: list[float] | None = None,
    take_profit_pairs: list[tuple[float, float]] | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    end = end or iso_date(date.today())
    windows = windows or [12, 18, 24]
    top_ns = top_ns or [5, 7]
    score_thresholds = score_thresholds or [2.5, 3.0]
    stop_multipliers = stop_multipliers or [2.5, 3.0]
    take_profit_pairs = take_profit_pairs or [(0.30, 0.10), (0.35, 0.12)]

    rows: list[dict[str, Any]] = []
    total = len(windows) * len(top_ns) * len(score_thresholds) * len(stop_multipliers) * len(take_profit_pairs)
    current = 0
    for months in windows:
        start = _window_start(end, months)
        for top_n in top_ns:
            for threshold in score_thresholds:
                for stop in stop_multipliers:
                    for tp_trigger, tp_trailing in take_profit_pairs:
                        current += 1
                        if verbose:
                            print(
                                f"[{current}/{total}] window={months}m top={top_n} "
                                f"raw>={threshold:.2f} stop={stop:.2f} "
                                f"take={tp_trigger:.0%}/{tp_trailing:.0%}",
                                flush=True,
                            )
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
                            if verbose:
                                print(f"  -> failed: {exc}", flush=True)
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
                        if verbose:
                            print(
                                f"  -> CAGR {summary['cagr']:.2%}, MDD {summary['mdd']:.2%}, "
                                f"Sharpe {summary['sharpe']:.2f}",
                                flush=True,
                            )
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
    best = _pick_best(report, end, "coarse")

    out = DATA_DIR / "backtests" / f"strategy_window_optimization_{end}.csv"
    safe_to_csv(report, out, index=False, encoding="utf-8-sig")
    if best:
        OPTIMIZED_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        OPTIMIZED_CONFIG_PATH.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, best


def _bounded_values(center: float, offsets: list[float], low: float, high: float) -> list[float]:
    values = {round(min(max(center + offset, low), high), 4) for offset in offsets}
    return sorted(values)


def refine_optimized_strategy(
    coarse_best: dict[str, Any],
    end: str | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not coarse_best:
        return pd.DataFrame(), {}

    end = end or str(coarse_best.get("end") or iso_date(date.today()))
    window = int(coarse_best["window_months"])
    top_n = int(coarse_best["top_n"])
    threshold = float(coarse_best["score_threshold"])
    stop = float(coarse_best["stop_multiplier"])
    tp_trigger = float(coarse_best["take_profit_trigger_pct"])
    tp_trailing = float(coarse_best["take_profit_trailing_pct"])
    refined_take_profit_pairs = [
        (
            round(min(max(tp_trigger + trigger_offset, 0.20), 0.45), 4),
            round(min(max(tp_trailing + trailing_offset, 0.06), 0.16), 4),
        )
        for trigger_offset, trailing_offset in [(-0.05, -0.02), (0.0, 0.0), (0.05, 0.02)]
    ]

    report, refined_best = optimize_strategy_windows(
        end=end,
        windows=[window],
        top_ns=[top_n],
        score_thresholds=_bounded_values(threshold, [-0.25, 0.0, 0.25], 1.5, 4.0),
        stop_multipliers=_bounded_values(stop, [-0.25, 0.0, 0.25], 1.5, 3.5),
        take_profit_pairs=sorted(set(refined_take_profit_pairs)),
        verbose=verbose,
    )
    if refined_best:
        refined_best["selection_phase"] = "refined"
        refined_best["coarse_strategy"] = coarse_best
        OPTIMIZED_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        OPTIMIZED_CONFIG_PATH.write_text(json.dumps(refined_best, ensure_ascii=False, indent=2), encoding="utf-8")
    out = DATA_DIR / "backtests" / f"strategy_window_refinement_{end}.csv"
    safe_to_csv(report, out, index=False, encoding="utf-8-sig")
    return report, refined_best


def optimize_strategy_two_stage(
    end: str | None = None,
    windows: list[int] | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    coarse_report, coarse_best = optimize_strategy_windows(end=end, windows=windows, verbose=verbose)
    if not coarse_best:
        return coarse_report, {}
    safe_to_csv(
        coarse_report,
        DATA_DIR / "backtests" / f"strategy_window_coarse_optimization_{coarse_best['end']}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    refine_report, refined_best = refine_optimized_strategy(coarse_best, end=end or coarse_best["end"], verbose=verbose)
    if refined_best:
        combined = pd.concat([coarse_report.assign(phase="coarse"), refine_report.assign(phase="refined")], ignore_index=True)
        out = DATA_DIR / "backtests" / f"strategy_two_stage_optimization_{refined_best['end']}.csv"
        safe_to_csv(combined, out, index=False, encoding="utf-8-sig")
        return combined, refined_best
    return coarse_report, coarse_best
