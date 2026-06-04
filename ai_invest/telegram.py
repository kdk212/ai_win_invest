from __future__ import annotations

import pandas as pd
import requests

from .secrets import load_telegram_credentials
from .utils import krw, pct


SEPARATOR = "━━━━━━━━━━━━━━━━━━━━"
MAX_TELEGRAM_MESSAGE_LENGTH = 3800


def _money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.0f}원"


def _value(row: pd.Series, key: str, default: object = "-") -> object:
    value = row.get(key, default)
    if value is None or pd.isna(value):
        return default
    return value


def _split_message(message: str, limit: int = MAX_TELEGRAM_MESSAGE_LENGTH) -> list[str]:
    if len(message) <= limit:
        return [message]

    chunks: list[str] = []
    current = ""
    for block in message.split("\n\n"):
        candidate = f"{current}\n\n{block}".strip() if current else block
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = block
    if current:
        chunks.append(current)
    return chunks


def format_recommendations(df: pd.DataFrame) -> str:
    if df.empty:
        return "오늘 생성된 추천 후보가 없습니다."

    as_of = df["as_of"].iloc[0]
    macro = df["macro_label"].iloc[0] if "macro_label" in df else "-"
    lines = [
        "📈 AI Invest Korea",
        "",
        f"추천 기준일: {as_of}",
        f"시장 환경: {macro}",
        f"추천 종목수: {len(df)}개",
        "",
        SEPARATOR,
    ]

    for _, row in df.iterrows():
        score_100 = _value(row, "score_100", None)
        score_text = f"{float(score_100):.1f}/100" if score_100 is not None else "-"
        raw_score = float(_value(row, "score", 0.0))
        risk_summary = _value(row, "news_risk_summary", _value(row, "previous_day_news_summary", ""))

        lines.extend(
            [
                "",
                f"{int(row['rank'])}. {row['name']} ({str(row['ticker']).zfill(6)})",
                f"테마: {_value(row, 'theme')}",
                "",
                f"추천일 종가: {_money(row['close'])}",
                f"점수: {score_text} | raw {raw_score:.2f}",
                f"1개월: {pct(row['mom20'])} | 3개월: {pct(row['mom60'])}",
                f"거래대금: {krw(row['avg_trading_value_20'])}",
                "",
                "매매 기준",
                f"경고가: {_money(row.get('warning_price'))}",
                f"손절가: {_money(row.get('stop_price'))} ({pct(row.get('stop_loss_pct'))})",
                f"익절 트리거: {_money(row.get('take_profit_trigger_price'))}",
                f"추적 익절가: {_money(row.get('take_profit_trailing_price'))}",
            ]
        )
        if risk_summary:
            lines.extend(["", "최근 7일 리스크", str(risk_summary)])
        lines.extend(["", SEPARATOR])

    lines.extend(
        [
            "",
            "자동 선별 결과입니다.",
            "실제 매매 전 공시, 실적, 거래량, 본인 위험 한도를 확인하세요.",
        ]
    )
    return "\n".join(lines)


def send_telegram(message: str) -> bool:
    token, chat_id = load_telegram_credentials()
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for chunk in _split_message(message):
        response = requests.post(url, json={"chat_id": chat_id, "text": chunk}, timeout=15)
        response.raise_for_status()
    return True
