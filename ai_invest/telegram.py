from __future__ import annotations

import pandas as pd
import requests

from .secrets import load_telegram_credentials
from .utils import krw, pct


def format_recommendations(df: pd.DataFrame) -> str:
    if df.empty:
        return "No recommendation candidates were generated."
    as_of = df["as_of"].iloc[0]
    macro = df["macro_label"].iloc[0]
    lines = [f"[AI Invest Korea] {as_of}", f"Market regime: {macro}", ""]
    for _, row in df.iterrows():
        score_100 = row["score_100"] if "score_100" in row and pd.notna(row["score_100"]) else min(max(row["score"] * 10 + 50, 0), 100)
        lines.append(
            f"{int(row['rank'])}. {row['name']}({row['ticker']}) "
            f"date {row['as_of']} | close {row['close']:,.0f} | score {score_100:.1f}/100 "
            f"(raw {row['score']:.2f}) | 1M {pct(row['mom20'])} | 3M {pct(row['mom60'])} "
            f"| value {krw(row['avg_trading_value_20'])} "
            f"| warning {row['warning_price']:,.0f} ({pct(row['warning_stop_pct'])}) "
            f"| final stop {row['stop_price']:,.0f} ({pct(row['stop_loss_pct'])})"
            f" | TP trigger {row['take_profit_trigger_price']:,.0f} "
            f"| trailing TP {row['take_profit_trailing_price']:,.0f}"
        )
        news_summary = row.get("news_risk_summary", row.get("previous_day_news_summary", ""))
        if news_summary and pd.notna(news_summary):
            lines.append(f"   7d risk: {news_summary}")
    lines.append("")
    lines.append("Screening output only. Check disclosures, earnings, liquidity, and your risk limit before trading.")
    return "\n".join(lines)


def send_telegram(message: str) -> bool:
    token, chat_id = load_telegram_credentials()
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    response = requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=15)
    response.raise_for_status()
    return True
