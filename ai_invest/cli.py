from __future__ import annotations

import argparse

from .backtest import optimize_score_threshold, optimize_stop_loss, optimize_take_profit, run_backtest, summarize_backtest
from .config import DATA_DIR, ensure_dirs
from .news import enrich_recommendations_with_news
from .strategy import build_recommendations
from .telegram import format_recommendations, send_telegram
from .tracker import update_recommendation_returns
from .utils import pct, safe_to_csv
from .web import serve


def _cmd_daily(args: argparse.Namespace) -> None:
    ensure_dirs()
    recs = build_recommendations(top_n=args.top_n)
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
    daily.add_argument("--send-telegram", action="store_true")
    daily.add_argument("--with-news", action="store_true")
    daily.set_defaults(func=_cmd_daily)

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

    track = sub.add_parser("track", help="Update returns after recommendation")
    track.set_defaults(func=_cmd_track)

    web = sub.add_parser("web", help="Run local dashboard")
    web.add_argument("--port", type=int, default=8787)
    web.set_defaults(func=_cmd_web)

    args = parser.parse_args()
    args.func(args)
