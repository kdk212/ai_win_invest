from __future__ import annotations

from datetime import date, timedelta
from urllib.parse import quote_plus
from xml.etree import ElementTree

import pandas as pd
import requests

from .secrets import load_openai_api_key


def _google_news_rss_url(query: str, end_date: date, days: int = 7) -> str:
    after = (end_date - timedelta(days=days)).isoformat()
    before = (end_date + timedelta(days=1)).isoformat()
    q = quote_plus(f"{query} after:{after} before:{before}")
    return f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"


def _fetch_news_from_url(url: str, limit: int) -> list[dict[str, str]]:
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    root = ElementTree.fromstring(response.content)
    items = []
    for item in root.findall(".//item")[:limit]:
        title = item.findtext("title", default="")
        link = item.findtext("link", default="")
        pub_date = item.findtext("pubDate", default="")
        if title:
            items.append({"title": title, "link": link, "published": pub_date})
    return items


def fetch_previous_day_news(name: str, ticker: str, recommendation_date: str, limit: int = 5) -> list[dict[str, str]]:
    target = pd.to_datetime(recommendation_date).date() - timedelta(days=1)
    url = _google_news_rss_url(f"{name} {ticker}", target, days=1)
    return _fetch_news_from_url(url, limit)


def fetch_recent_news(
    name: str,
    ticker: str,
    recommendation_date: str,
    days: int = 7,
    limit: int = 8,
) -> list[dict[str, str]]:
    target = pd.to_datetime(recommendation_date).date()
    url = _google_news_rss_url(f"{name} {ticker}", target, days=days)
    return _fetch_news_from_url(url, limit)


def summarize_news_with_openai(name: str, ticker: str, news_items: list[dict[str, str]]) -> str:
    if not news_items:
        return "리스크: 최근 7일 뉴스 없음\n판정: 진입 가능\n이유: 뉴스 리스크는 낮지만 가격 기준은 별도 확인"

    api_key = load_openai_api_key()
    if not api_key:
        titles = "; ".join(item["title"] for item in news_items[:3])
        return f"리스크: OpenAI 키 없음. 최근 주요 제목만 확인: {titles}\n판정: 관망\n이유: 정성 리스크 요약 미완료"

    prompt_items = "\n".join(f"- {item['title']}" for item in news_items)
    payload = {
        "model": "gpt-4.1-mini",
        "input": [
            {
                "role": "system",
                "content": (
                    "You analyze Korean stock news only for risk control in a trading dashboard. "
                    "The stock has already passed a quantitative screen, so do not list positive catalysts. "
                    "Focus on downside risks, overheating, regulatory/disclosure issues, earnings uncertainty, "
                    "supply-demand crowding, valuation pressure, and news-quality concerns. "
                    "Be concise, practical, and do not guarantee returns."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"종목: {name}({ticker})\n"
                    f"최근 7일 뉴스 제목:\n{prompt_items}\n\n"
                    "아래 형식으로 한국어로 짧게 정리해줘. 줄바꿈을 반드시 지켜줘.\n"
                    "리스크: 하락 또는 변동성 확대를 만들 수 있는 핵심 리스크 1~2개만\n"
                    "판정: 진입 가능 / 분할 진입 / 관망 / 제외 중 하나\n"
                    "이유: 판정의 투자 행동 의미를 1문장으로 명확히 설명\n\n"
                    "판정 기준:\n"
                    "- 진입 가능: 뉴스 리스크가 낮아 기존 정량 추천과 손절/익절 기준 유지\n"
                    "- 분할 진입: 호재 선반영, 급등, 변동성 확대가 있어 비중을 줄여 단계적으로 진입\n"
                    "- 관망: 과열, 투자경고, 실적/공시 불확실성 등으로 신규 진입은 조정 확인 후 검토\n"
                    "- 제외: 중대한 악재, 회계/소송/규제/유동성 리스크가 있어 당일 추천에서 제거 권고"
                ),
            },
        ],
        "max_output_tokens": 220,
    }
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    texts = []
    for output in data.get("output", []):
        for content in output.get("content", []):
            if content.get("type") == "output_text":
                texts.append(content.get("text", ""))
    return "\n".join(texts).strip() or "리스크: 요약 생성 실패\n판정: 관망\n이유: 리스크 확인 전 신규 진입 보류"


def enrich_recommendations_with_news(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    enriched = df.copy()
    summaries = []
    links = []
    for _, row in enriched.iterrows():
        try:
            items = fetch_recent_news(str(row["name"]), str(row["ticker"]), str(row["as_of"]))
            summaries.append(summarize_news_with_openai(str(row["name"]), str(row["ticker"]), items))
            links.append(items[0]["link"] if items else "")
        except Exception as exc:
            summaries.append(f"리스크: 뉴스 조회 실패: {type(exc).__name__}\n판정: 관망\n이유: 최신 리스크 확인 전 신규 진입 보류")
            links.append("")
    enriched["news_risk_summary"] = summaries
    enriched["news_risk_link"] = links
    enriched["previous_day_news_summary"] = summaries
    enriched["previous_day_news_link"] = links
    return enriched
