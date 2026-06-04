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


def format_strategy_monitor(result: dict) -> str:
    optimized = result.get("optimized_strategy") or {}
    new_strategy = result.get("new_strategy") or {}
    active_strategy = new_strategy or optimized
    status = "재최적화 실행" if result.get("auto_optimized") else "정상 추적" if not result.get("needs_review") else "재검토 필요"
    lines = [
        "AI Invest 전략 점검",
        "",
        f"판정: {status}",
        f"기준일: {result.get('latest_date') or '-'}",
        f"실전 수익률: {pct(result.get('actual_total_return', 0.0))}",
        f"백테스트 기대: {pct(result.get('expected_return_from_backtest_cagr', 0.0))}",
        f"차이: {pct(result.get('shortfall', 0.0))}",
        "",
        SEPARATOR,
    ]

    if active_strategy:
        ratio = result.get("actual_to_expected_ratio")
        ratio_text = "-" if ratio is None else f"{float(ratio):.2f}배"
        lines.extend(
            [
                "",
                "현재 적용 전략",
                f"검증 기간: {active_strategy.get('window_months', '-')}개월",
                f"최적화 단계: {active_strategy.get('selection_phase', '-')}",
                f"추천 개수: Top {active_strategy.get('top_n', '-')}",
                f"raw 기준: {float(active_strategy.get('score_threshold', 0.0)):.2f} 이상",
                f"손절 계수: {float(active_strategy.get('stop_multiplier', 0.0)):.2f}",
                (
                    "익절/추적: "
                    f"{pct(active_strategy.get('take_profit_trigger_pct', 0.0))} / "
                    f"{pct(active_strategy.get('take_profit_trailing_pct', 0.0))}"
                ),
                f"백테스트 CAGR: {pct(active_strategy.get('cagr', 0.0))}",
                f"백테스트 MDD: {pct(active_strategy.get('mdd', 0.0))}",
                f"Sharpe: {float(active_strategy.get('sharpe', 0.0)):.2f}",
                f"기대 대비: {ratio_text}",
            ]
        )

    if result.get("auto_optimized"):
        lines.extend(["", "성과 미달 조건이 감지되어 새 전략을 저장했습니다. 다음 추천부터 새 기준이 반영됩니다."])
    elif result.get("needs_review"):
        lines.extend(["", "성과가 기대치보다 낮습니다. 자동 재최적화 옵션이 꺼져 있으면 수동 점검이 필요합니다."])
    else:
        lines.extend(["", "실전 포트가 백테스트 기대 범위 안에서 추적 중입니다."])

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
