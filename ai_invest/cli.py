from __future__ import annotations

import argparse
from datetime import date, timedelta

from .backtest import optimize_score_threshold, optimize_stop_loss, optimize_take_profit, run_backtest, summarize_backtest
from .config import DATA_DIR, ensure_dirs
from .monitor import evaluate_strategy_performance
from .news import enrich_recommendations_with_news
from .optimizer import load_optimized_strategy, optimize_strategy_two_stage
from .strategy import build_recommendations
from .telegram import format_recommendations, format_strategy_monitor, send_telegram
from .tracker import update_recommendation_returns
from .utils import pct, safe_to_csv
from .web import serve


def _cmd_daily(args: argparse.Namespace) -> None:
    ensure_dirs()
    as_of = date.fromisoformat(args.as_of_date) if args.as_of_date else None
    recs = build_recommendations(as_of=as_of, top_n=args.top_n)
    if recs.empty:
        print("No recommendation candidates were generated. Check data access or the as-of date.")
        return
    if args.with_news:
        recs = enrich_recommendations_with_news(recs)
    out = DATA_DIR / "recommendations" / f"recommendations_{recs['as_of'].iloc[0]}.csv"
    saved = safe_to_csv(recs, out, index=False, encoding="utf-8-sig")
    message = format_recommendations(recs)
    print(message)
    print(f"\nSaved: {out}" if saved else "\nSave skipped: this execution environment blocked new file writes.")
    if args.send_telegram:
        sent = send_telegram(message)
        print("Telegram sent" if sent else "Telegram skipped: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing.")


def _save_recommendations_for_date(as_of: date, top_n: int | None = None, with_news: bool = False) -> tuple[str, int]:
    recs = build_recommendations(as_of=as_of, top_n=top_n)
    if recs.empty:
        return as_of.isoformat(), 0
    recs = recs.copy()
    recs["as_of"] = as_of.isoformat()
    if with_news:
        recs = enrich_recommendations_with_news(recs)
    out = DATA_DIR / "recommendations" / f"recommendations_{as_of.isoformat()}.csv"
    safe_to_csv(recs, out, index=False, encoding="utf-8-sig")
    return as_of.isoformat(), len(recs)


def _cmd_backfill_recommendations(args: argparse.Namespace) -> None:
    ensure_dirs()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()
    current = start
    saved: dict[str, int] = {}
    while current <= end:
        as_of, count = _save_recommendations_for_date(current, top_n=args.top_n, with_news=args.with_news)
        if count:
            saved[as_of] = count
            print(f"Saved {as_of}: {count} recommendations")
        else:
            print(f"Skipped {current.isoformat()}: no recommendation candidates")
        current += timedelta(days=1)
    print(f"Backfill complete: {len(saved)} recommendation date files")


def _cmd_backtest(args: argparse.Namespace) -> None:
    ensure_dirs()
    result, _ = run_backtest(
        args.start,
        args.end,
        args.top_n,
        stop_multiplier=args.stop_multiplier,
        score_threshold=args.score_threshold,
    )
    summary = summarize_backtest(result)
    print("Backtest summary")
    print(f"Total return: {pct(summary['total_return'])}")
    print(f"CAGR: {pct(summary['cagr'])}")
    print(f"MDD: {pct(summary['mdd'])}")
    print(f"Sharpe: {summary['sharpe']:.2f}")
    print(f"Average exposure: {pct(summary['exposure'])}")


def _cmd_optimize(args: argparse.Namespace) -> None:
    ensure_dirs()
    result = optimize_stop_loss(args.start, args.end, args.top_n)
    print("Stop-loss optimization")
    for _, row in result.iterrows():
        print(
            f"multiplier {row['stop_multiplier']:.1f}: "
            f"CAGR {pct(row['cagr'])}, MDD {pct(row['mdd'])}, Sharpe {row['sharpe']:.2f}"
        )


def _cmd_optimize_score(args: argparse.Namespace) -> None:
    ensure_dirs()
    result = optimize_score_threshold(args.start, args.end, args.top_n)
    print("Raw score threshold optimization")
    for _, row in result.iterrows():
        print(
            f"raw >= {row['score_threshold']:.2f}: "
            f"avg positions {row['avg_positions']:.2f}, "
            f"CAGR {pct(row['cagr'])}, MDD {pct(row['mdd'])}, Sharpe {row['sharpe']:.2f}"
        )


def _cmd_optimize_take_profit(args: argparse.Namespace) -> None:
    ensure_dirs()
    result = optimize_take_profit(args.start, args.end, args.top_n)
    print("Take-profit optimization")
    for _, row in result.iterrows():
        print(
            f"trigger {pct(row['take_profit_trigger_pct'])}, trail {pct(row['take_profit_trailing_pct'])}: "
            f"avg positions {row['avg_positions']:.2f}, "
            f"CAGR {pct(row['cagr'])}, MDD {pct(row['mdd'])}, Sharpe {row['sharpe']:.2f}"
        )


def _cmd_optimize_strategy(args: argparse.Namespace) -> None:
    ensure_dirs()
    windows = [int(value.strip()) for value in args.windows.split(",") if value.strip()]
    report, best = optimize_strategy_two_stage(end=args.end, windows=windows, verbose=True)
    if report.empty or not best:
        print("Strategy optimization did not produce a valid candidate.")
        return
    print("Two-stage strategy optimization complete")
    print(f"Selection phase: {best.get('selection_phase', '-')}")
    print(f"Selected window: {best['window_months']} months ({best['start']} ~ {best['end']})")
    print(f"Top N: {best['top_n']}")
    print(f"Raw score threshold: {best['score_threshold']:.2f}")
    print(f"Stop multiplier: {best['stop_multiplier']:.2f}")
    print(
        f"Take-profit: trigger {pct(best['take_profit_trigger_pct'])}, "
        f"trailing {pct(best['take_profit_trailing_pct'])}"
    )
    print(f"CAGR: {pct(best['cagr'])}, MDD: {pct(best['mdd'])}, Sharpe: {best['sharpe']:.2f}")
    print("Saved: config/optimized_strategy.json")


def _cmd_monitor_strategy(args: argparse.Namespace) -> None:
    ensure_dirs()
    result = evaluate_strategy_performance(auto_optimize=args.auto_optimize)
    print("Strategy performance monitor")
    print(f"Latest portfolio date: {result['latest_date'] or '-'}")
    print(f"Elapsed days: {result['elapsed_days']}")
    print(f"Actual return: {pct(result['actual_total_return'])}")
    print(f"Expected return: {pct(result['expected_return_from_backtest_cagr'])}")
    print(f"Shortfall: {pct(result['shortfall'])}")
    print(f"Needs review: {result['needs_review']}")
    print(f"Auto optimized: {result['auto_optimized']}")
    print("Saved: data/performance/strategy_monitor.json")
    if args.send_telegram and (result["needs_review"] or result["auto_optimized"]):
        sent = send_telegram(format_strategy_monitor(result))
        print("Telegram sent" if sent else "Telegram skipped: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing.")


def _cmd_strategy_status(_: argparse.Namespace) -> None:
    strategy = load_optimized_strategy()
    if not strategy:
        print("No optimized strategy is saved yet. Run: python main.py optimize-strategy --windows 12,18,24")
        return
    print("Optimized strategy status")
    print(f"Selected at: {strategy.get('selected_at', '-')}")
    print(f"Selection phase: {strategy.get('selection_phase', '-')}")
    print(f"Window: {strategy.get('window_months', '-')} months ({strategy.get('start', '-')} ~ {strategy.get('end', '-')})")
    print(f"Top N: {strategy.get('top_n', '-')}")
    print(f"Raw score threshold: {float(strategy.get('score_threshold', 0.0)):.2f}")
    print(f"Stop multiplier: {float(strategy.get('stop_multiplier', 0.0)):.2f}")
    print(
        f"Take-profit: trigger {pct(strategy.get('take_profit_trigger_pct', 0.0))}, "
        f"trailing {pct(strategy.get('take_profit_trailing_pct', 0.0))}"
    )
    print(f"Backtest CAGR: {pct(strategy.get('cagr', 0.0))}")
    print(f"Backtest MDD: {pct(strategy.get('mdd', 0.0))}")
    print(f"Sharpe: {float(strategy.get('sharpe', 0.0)):.2f}")
    print(f"Recent validation return: {pct(strategy.get('validation_total_return', 0.0))}")
    print(f"Recent validation MDD: {pct(strategy.get('validation_mdd', 0.0))}")
    print(f"Recent validation Sharpe: {float(strategy.get('validation_sharpe', 0.0)):.2f}")


def _format_strategy_review(start: str, end: str | None, top_n: int) -> str:
    score = optimize_score_threshold(start, end, top_n).head(5)
    stop = optimize_stop_loss(start, end, top_n).head(5)
    take_profit = optimize_take_profit(start, end, top_n).head(5)

    lines = [
        "[AI Invest] Weekly strategy review",
        "",
        f"Period: {start} ~ {end or 'latest'}",
        f"Max positions: {top_n}",
        "",
        "====================",
        "",
        "Raw score threshold candidates",
    ]
    for _, row in score.iterrows():
        lines.append(
            f"- raw >= {row['score_threshold']:.2f} | avg {row['avg_positions']:.2f} positions | "
            f"CAGR {pct(row['cagr'])} | MDD {pct(row['mdd'])} | Sharpe {row['sharpe']:.2f}"
        )

    lines.extend(["", "Stop multiplier candidates"])
    for _, row in stop.iterrows():
        lines.append(
            f"- multiplier {row['stop_multiplier']:.1f} | CAGR {pct(row['cagr'])} | "
            f"MDD {pct(row['mdd'])} | Sharpe {row['sharpe']:.2f}"
        )

    lines.extend(["", "Take-profit / trailing candidates"])
    for _, row in take_profit.iterrows():
        lines.append(
            f"- trigger {pct(row['take_profit_trigger_pct'])}, trail {pct(row['take_profit_trailing_pct'])} | "
            f"CAGR {pct(row['cagr'])} | MDD {pct(row['mdd'])} | Sharpe {row['sharpe']:.2f}"
        )

    lines.extend(
        [
            "",
            "====================",
            "",
            "Review only. Strategy settings are not changed automatically.",
            "Apply changes only after confirming stability across multiple periods.",
        ]
    )
    return "\n".join(lines)


def _cmd_review_strategy(args: argparse.Namespace) -> None:
    ensure_dirs()
    message = _format_strategy_review(args.start, args.end, args.top_n)
    print(message)
    if args.send_telegram:
        sent = send_telegram(message)
        print("Telegram sent" if sent else "Telegram skipped: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing.")


def _cmd_track(_: argparse.Namespace) -> None:
    ensure_dirs()
    df = update_recommendation_returns()
    print(f"Updated {len(df)} tracked recommendation rows.")


def _cmd_web(args: argparse.Namespace) -> None:
    ensure_dirs()
    serve(port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Invest Korea")
    sub = parser.add_subparsers(dest="command", required=True)

    daily = sub.add_parser("daily", help="Generate today's recommendation candidates")
    daily.add_argument("--top-n", type=int, default=None)
    daily.add_argument("--as-of-date", default=None, help="Recommendation date in YYYY-MM-DD format")
    daily.add_argument("--send-telegram", action="store_true")
    daily.add_argument("--with-news", action="store_true")
    daily.set_defaults(func=_cmd_daily)

    backfill = sub.add_parser("backfill-recommendations", help="Generate recommendation files for a date range")
    backfill.add_argument("--start", default="2026-06-01")
    backfill.add_argument("--end", default=None)
    backfill.add_argument("--top-n", type=int, default=None)
    backfill.add_argument("--with-news", action="store_true")
    backfill.set_defaults(func=_cmd_backfill_recommendations)

    backtest = sub.add_parser("backtest", help="Run a strategy backtest")
    backtest.add_argument("--start", default="2023-01-01")
    backtest.add_argument("--end", default=None)
    backtest.add_argument("--top-n", type=int, default=7)
    backtest.add_argument("--stop-multiplier", type=float, default=None)
    backtest.add_argument("--score-threshold", type=float, default=None)
    backtest.set_defaults(func=_cmd_backtest)

    optimize = sub.add_parser("optimize-stop", help="Find a better volatility stop-loss setting")
    optimize.add_argument("--start", default="2024-11-01")
    optimize.add_argument("--end", default=None)
    optimize.add_argument("--top-n", type=int, default=10)
    optimize.set_defaults(func=_cmd_optimize)

    optimize_score = sub.add_parser("optimize-score", help="Find a better raw score cutoff")
    optimize_score.add_argument("--start", default="2024-11-01")
    optimize_score.add_argument("--end", default=None)
    optimize_score.add_argument("--top-n", type=int, default=7)
    optimize_score.set_defaults(func=_cmd_optimize_score)

    optimize_tp = sub.add_parser("optimize-take-profit", help="Find a better take-profit trigger and trailing setting")
    optimize_tp.add_argument("--start", default="2024-11-01")
    optimize_tp.add_argument("--end", default=None)
    optimize_tp.add_argument("--top-n", type=int, default=7)
    optimize_tp.set_defaults(func=_cmd_optimize_take_profit)

    optimize_strategy = sub.add_parser("optimize-strategy", help="Compare 12/18/24 month windows and save best strategy settings")
    optimize_strategy.add_argument("--end", default=None)
    optimize_strategy.add_argument("--windows", default="12,18,24")
    optimize_strategy.set_defaults(func=_cmd_optimize_strategy)

    monitor_strategy = sub.add_parser("monitor-strategy", help="Compare live virtual portfolio return with optimized backtest expectation")
    monitor_strategy.add_argument("--auto-optimize", action="store_true")
    monitor_strategy.add_argument("--send-telegram", action="store_true")
    monitor_strategy.set_defaults(func=_cmd_monitor_strategy)

    strategy_status = sub.add_parser("strategy-status", help="Show the saved optimized strategy settings")
    strategy_status.set_defaults(func=_cmd_strategy_status)

    review = sub.add_parser("review-strategy", help="Send a weekly strategy optimization review")
    review.add_argument("--start", default="2024-11-01")
    review.add_argument("--end", default=None)
    review.add_argument("--top-n", type=int, default=7)
    review.add_argument("--send-telegram", action="store_true")
    review.set_defaults(func=_cmd_review_strategy)

    track = sub.add_parser("track", help="Update returns after recommendation")
    track.set_defaults(func=_cmd_track)

    web = sub.add_parser("web", help="Run local dashboard")
    web.add_argument("--port", type=int, default=8787)
    web.set_defaults(func=_cmd_web)

    args = parser.parse_args()
    args.func(args)
