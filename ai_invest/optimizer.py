from __future__ import annotations

import json
from datetime import date
from typing import Any

import pandas as pd

from .backtest import run_backtest, summarize_backtest
from .config import DATA_DIR, ROOT
from .utils import iso_date, safe_to_csv


OPTIMIZED_CONFIG_PATH = ROOT / "config" / "optimized_strategy.json"
STRATEGY_HISTORY_PATH = DATA_DIR / "performance" / "strategy_history.csv"
TARGET_CAGR = 1.0
HIGH_RETURN_CAGR = 6.0


def load_optimized_strategy() -> dict[str, Any]:
    if not OPTIMIZED_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(OPTIMIZED_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_optimized_strategy(strategy: dict[str, Any]) -> None:
    OPTIMIZED_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    OPTIMIZED_CONFIG_PATH.write_text(json.dumps(strategy, ensure_ascii=False, indent=2), encoding="utf-8")

    STRATEGY_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history_row = {key: value for key, value in strategy.items() if key != "coarse_strategy"}
    history = pd.DataFrame([history_row])
    if STRATEGY_HISTORY_PATH.exists():
        try:
            previous = pd.read_csv(STRATEGY_HISTORY_PATH)
            history = pd.concat([previous, history], ignore_index=True)
        except Exception:
            pass
    safe_to_csv(history, STRATEGY_HISTORY_PATH, index=False, encoding="utf-8-sig")


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


def _validation_summary(result: pd.DataFrame, days: int = 42) -> dict[str, float]:
    empty = {
        "validation_total_return": 0.0,
        "validation_cagr": 0.0,
        "validation_mdd": 0.0,
        "validation_sharpe": 0.0,
        "validation_avg_return": 0.0,
        "validation_worst_return": 0.0,
        "validation_weak_count": 0.0,
    }
    if result.empty:
        return empty
    window = result.tail(min(days, len(result))).copy()
    if window.empty:
        return empty
    base_equity = float(window["equity"].iloc[0])
    if base_equity:
        window["equity"] = window["equity"].astype(float) / base_equity
    summary = summarize_backtest(window)
    fold_returns = []
    fold_mdds = []
    for fold in range(3):
        end_idx = len(result) - fold * days
        start_idx = max(0, end_idx - days)
        fold_window = result.iloc[start_idx:end_idx].copy()
        if len(fold_window) < 5:
            continue
        fold_base = float(fold_window["equity"].iloc[0])
        if fold_base:
            fold_window["equity"] = fold_window["equity"].astype(float) / fold_base
        fold_summary = summarize_backtest(fold_window)
        fold_returns.append(float(fold_summary["total_return"]))
        fold_mdds.append(float(fold_summary["mdd"]))
    validation_avg_return = float(pd.Series(fold_returns).mean()) if fold_returns else 0.0
    validation_worst_return = float(min(fold_returns)) if fold_returns else 0.0
    validation_weak_count = float(sum(1 for ret, mdd in zip(fold_returns, fold_mdds) if ret < 0 or mdd <= -0.12))
    return {
        "validation_total_return": float(summary["total_return"]),
        "validation_cagr": float(summary["cagr"]),
        "validation_mdd": float(summary["mdd"]),
        "validation_sharpe": float(summary["sharpe"]),
        "validation_avg_return": validation_avg_return,
        "validation_worst_return": validation_worst_return,
        "validation_weak_count": validation_weak_count,
    }


def _selection_objective(summary: dict[str, float], validation: dict[str, float]) -> float:
    train_score = _objective(summary)
    validation_total_return = float(validation.get("validation_total_return", 0.0))
    validation_avg_return = float(validation.get("validation_avg_return", validation_total_return))
    validation_worst_return = float(validation.get("validation_worst_return", validation_total_return))
    validation_weak_count = float(validation.get("validation_weak_count", 0.0))
    validation_cagr = float(validation.get("validation_cagr", 0.0))
    validation_mdd = float(validation.get("validation_mdd", 0.0))
    validation_sharpe = float(validation.get("validation_sharpe", 0.0))
    validation_score = validation_cagr * max(0.10, 1.0 + validation_mdd) + 0.10 * validation_sharpe
    stability_penalty = max(0.0, float(summary.get("cagr", 0.0)) - max(validation_cagr, 0.0)) * 0.15
    validation_loss_penalty = max(0.0, -validation_total_return) * 4.0
    validation_fold_loss_penalty = max(0.0, -validation_avg_return) * 3.0 + max(0.0, -validation_worst_return) * 2.0
    validation_weak_penalty = validation_weak_count * 0.40
    validation_drawdown_penalty = max(0.0, abs(validation_mdd) - 0.08) * 3.5
    exposure = float(summary.get("exposure", 0.0))
    low_selectivity_penalty = max(0.0, exposure - 0.82) * 0.10
    return (
        0.70 * train_score
        + 0.30 * validation_score
        - stability_penalty
        - validation_loss_penalty
        - validation_fold_loss_penalty
        - validation_weak_penalty
        - validation_drawdown_penalty
        - low_selectivity_penalty
    )


def _validation_grade(validation: dict[str, float]) -> str:
    validation_total_return = float(validation.get("validation_total_return", 0.0))
    validation_avg_return = float(validation.get("validation_avg_return", validation_total_return))
    validation_worst_return = float(validation.get("validation_worst_return", validation_total_return))
    validation_weak_count = float(validation.get("validation_weak_count", 0.0))
    validation_mdd = float(validation.get("validation_mdd", 0.0))
    validation_sharpe = float(validation.get("validation_sharpe", 0.0))
    if validation_total_return >= 0 and validation_avg_return >= 0 and validation_worst_return >= -0.03 and validation_mdd > -0.08 and validation_sharpe >= 0 and validation_weak_count == 0:
        return "pass"
    if validation_total_return < 0 or validation_avg_return < 0 or validation_worst_return < -0.08 or validation_mdd <= -0.12 or validation_weak_count >= 2:
        return "weak"
    return "watch"


def _row_to_strategy(best_row: pd.Series, end: str, phase: str, selection_mode: str) -> dict[str, Any]:
    return {
        "selected_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "selection_phase": phase,
        "selection_rule": "seek CAGR >= 100%; if CAGR >= 600% candidates exist, minimize risk first",
        "selection_mode": selection_mode,
        "target_cagr": TARGET_CAGR,
        "high_return_risk_minimize_cagr": HIGH_RETURN_CAGR,
        "window_months": int(best_row["window_months"]),
        "start": str(best_row["start"]),
        "end": end,
        "top_n": int(best_row["top_n"]),
        "score_threshold": float(best_row["score_threshold"]),
        "stop_multiplier": float(best_row["stop_multiplier"]),
        "take_profit_trigger_pct": float(best_row["take_profit_trigger_pct"]),
        "take_profit_trailing_pct": float(best_row["take_profit_trailing_pct"]),
        "objective": float(best_row["objective"]),
        "selection_objective": float(best_row["selection_objective"]),
        "cagr": float(best_row["cagr"]),
        "mdd": float(best_row["mdd"]),
        "sharpe": float(best_row["sharpe"]),
        "exposure": float(best_row["exposure"]),
        "validation_total_return": float(best_row.get("validation_total_return", 0.0)),
        "validation_cagr": float(best_row.get("validation_cagr", 0.0)),
        "validation_mdd": float(best_row.get("validation_mdd", 0.0)),
        "validation_sharpe": float(best_row.get("validation_sharpe", 0.0)),
        "validation_avg_return": float(best_row.get("validation_avg_return", 0.0)),
        "validation_worst_return": float(best_row.get("validation_worst_return", 0.0)),
        "validation_weak_count": float(best_row.get("validation_weak_count", 0.0)),
        "validation_grade": str(best_row.get("validation_grade", "-")),
    }


def _pick_best(report: pd.DataFrame, end: str, phase: str) -> dict[str, Any]:
    if report.empty:
        return {}
    valid = report[report["error"].fillna("") == ""].copy()
    if valid.empty:
        return {}

    high_return = valid[valid["cagr"] >= HIGH_RETURN_CAGR].copy()
    if not high_return.empty:
        stable_high_return = high_return[(high_return["validation_grade"] != "weak") & (high_return["validation_weak_count"].fillna(0) < 2)].copy()
        candidates = stable_high_return if not stable_high_return.empty else high_return
        selection_mode = "six_hundred_cagr_stable_risk_minimized" if not stable_high_return.empty else "six_hundred_cagr_risk_minimized"
        best_row = candidates.sort_values(
            ["validation_weak_count", "mdd", "validation_mdd", "validation_worst_return", "exposure", "score_threshold", "sharpe", "cagr"],
            ascending=[True, False, False, False, True, False, False, False],
        ).iloc[0]
        return _row_to_strategy(best_row, end, phase, selection_mode)

    target_met = valid[valid["cagr"] >= TARGET_CAGR].copy()
    if not target_met.empty:
        stable_target = target_met[(target_met["validation_grade"] != "weak") & (target_met["validation_weak_count"].fillna(0) < 2)].copy()
        candidates = stable_target if not stable_target.empty else target_met
        selection_mode = "target_cagr_stable_return_optimized" if not stable_target.empty else "target_cagr_return_optimized"
        best_row = candidates.sort_values(
            ["selection_objective", "validation_weak_count", "validation_mdd", "validation_worst_return", "sharpe", "cagr", "mdd"],
            ascending=[False, True, False, False, False, False, False],
        ).iloc[0]
        return _row_to_strategy(best_row, end, phase, selection_mode)

    best_row = valid.sort_values(
        ["cagr", "selection_objective", "sharpe", "mdd", "validation_worst_return"],
        ascending=[False, False, False, False, False],
    ).iloc[0]
    return _row_to_strategy(best_row, end, phase, "return_seeking_until_target_cagr")


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
    top_ns = top_ns or [1, 2, 3, 5, 7]
    score_thresholds = score_thresholds or [2.75, 3.0, 3.25, 3.5, 4.0]
    stop_multipliers = stop_multipliers or [1.5, 2.0, 2.5, 3.0]
    take_profit_pairs = take_profit_pairs or [(0.20, 0.08), (0.25, 0.08), (0.30, 0.10), (0.35, 0.12)]

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
                            validation = _validation_summary(result)
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
                        objective = _objective(summary)
                        selection_objective = _selection_objective(summary, validation)
                        selection_objective -= max(0, top_n - 5) * 0.08
                        validation_grade = _validation_grade(validation)
                        if verbose:
                            print(
                                f"  -> CAGR {summary['cagr']:.2%}, MDD {summary['mdd']:.2%}, "
                                f"Sharpe {summary['sharpe']:.2f}, "
                                f"validation {validation['validation_total_return']:.2%} ({validation_grade})",
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
                                "objective": objective,
                                "selection_objective": selection_objective,
                                "validation_grade": validation_grade,
                                "error": "",
                                **summary,
                                **validation,
                            }
                        )

    report = pd.DataFrame(rows)
    best = _pick_best(report, end, "coarse")

    out = DATA_DIR / "backtests" / f"strategy_window_optimization_{end}.csv"
    safe_to_csv(report, out, index=False, encoding="utf-8-sig")
    if best:
        save_optimized_strategy(best)
    return report, best


def _bounded_values(center: float, offsets: list[float], low: float, high: float) -> list[float]:
    values = {round(min(max(center + offset, low), high), 4) for offset in offsets}
    return sorted(values)


def _bounded_int_values(center: int, offsets: list[int], low: int, high: int) -> list[int]:
    values = {min(max(center + offset, low), high) for offset in offsets}
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
            round(min(max(tp_trigger + trigger_offset, 0.18), 0.50), 4),
            round(min(max(tp_trailing + trailing_offset, 0.06), 0.18), 4),
        )
        for trigger_offset, trailing_offset in [(-0.05, -0.02), (0.0, 0.0), (0.05, 0.02)]
    ]

    report, refined_best = optimize_strategy_windows(
        end=end,
        windows=[window],
        top_ns=_bounded_int_values(top_n, [-1, 0, 1], 1, 7),
        score_thresholds=_bounded_values(threshold, [-0.25, -0.10, 0.0, 0.10, 0.25], 2.5, 4.75),
        stop_multipliers=_bounded_values(stop, [-0.5, -0.25, 0.0, 0.25, 0.5], 1.0, 3.5),
        take_profit_pairs=sorted(set(refined_take_profit_pairs)),
        verbose=verbose,
    )
    if refined_best:
        refined_best["selection_phase"] = "refined"
        refined_best["coarse_strategy"] = coarse_best
        save_optimized_strategy(refined_best)
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
