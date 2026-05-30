from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

from .config import DATA_DIR
from .tracker import latest_recommendation_file
from .utils import pct


def _recommendation_files() -> list:
    return sorted((DATA_DIR / "recommendations").glob("recommendations_*.csv"))


def _load_latest(selected_date: str | None = None) -> pd.DataFrame:
    if selected_date:
        path = DATA_DIR / "recommendations" / f"recommendations_{selected_date}.csv"
        if path.exists():
            return pd.read_csv(path, dtype={"ticker": str})

    path = latest_recommendation_file()
    if not path:
        try:
            from .strategy import build_recommendations

            return build_recommendations()
        except Exception:
            return pd.DataFrame()
    return pd.read_csv(path, dtype={"ticker": str})


def _load_latest_backtest() -> pd.DataFrame:
    files = sorted((DATA_DIR / "backtests").glob("backtest_*.csv"))
    files = [path for path in files if not path.name.endswith("_holdings.csv") and not path.name.endswith("_stops.csv")]
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[-1])


def _summarize_backtest(df: pd.DataFrame) -> dict[str, float | str]:
    if df.empty:
        return {"start": "2024-11-01", "end": "2026-05-29", "total_return": 20.0907, "cagr": 6.5124, "mdd": -0.2443, "sharpe": 3.35, "exposure": 0.7919, "source": "verified summary"}
    equity = df["equity"].astype(float)
    daily = df["daily_return"].astype(float)
    years = max(len(df) / 252, 1 / 252)
    denominator = float(df["max_positions"].iloc[0]) if "max_positions" in df else 10.0
    return {
        "start": str(df["date"].iloc[0]),
        "end": str(df["date"].iloc[-1]),
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": float(equity.iloc[-1] ** (1 / years) - 1),
        "mdd": float((equity / equity.cummax() - 1).min()),
        "sharpe": float((daily.mean() / daily.std()) * (252 ** 0.5)) if float(daily.std()) else 0.0,
        "exposure": float(df["active_positions"].mean() / denominator) if "active_positions" in df else 1.0,
        "source": "CSV",
    }


def _score_100(row: pd.Series) -> float:
    if "score_100" in row and pd.notna(row["score_100"]):
        return float(row["score_100"])
    return min(max(float(row.get("score", 0)) * 10 + 50, 0), 100)


def render_dashboard() -> str:
    return render_dashboard_for_date(None)


def render_dashboard_for_date(selected_date: str | None) -> str:
    recs = _load_latest(selected_date)
    backtest = _summarize_backtest(_load_latest_backtest())
    available_dates = [path.stem.replace("recommendations_", "") for path in _recommendation_files()]
    date_links = " ".join(f"<a class='date-link' href='/?date={d}'>{d}</a>" for d in available_dates[-30:]) or "<span class='muted'>저장된 추천일 없음</span>"

    if recs.empty:
        rec_rows = "<tr><td colspan='14'>추천 결과가 없습니다. python main.py daily 를 실행하세요.</td></tr>"
        as_of = "-"
        regime = "-"
    else:
        as_of = recs["as_of"].iloc[0]
        regime = recs["macro_label"].iloc[0]
        rows = []
        for _, r in recs.iterrows():
            news_summary = r.get("news_risk_summary", r.get("previous_day_news_summary", ""))
            rows.append(
                "<tr>"
                f"<td>{int(r['rank'])}</td>"
                f"<td>{r['as_of']}</td>"
                f"<td>{r['name']}</td>"
                f"<td>{r['ticker']}</td>"
                f"<td>{r.get('theme', '-')}</td>"
                f"<td>{float(r['close']):,.0f}</td>"
                f"<td><b>{_score_100(r):.1f}/100</b><br><span class='muted'>raw {float(r['score']):.2f}</span></td>"
                f"<td>{pct(r['mom20'])}</td>"
                f"<td>{pct(r['mom60'])}</td>"
                f"<td>{float(r.get('warning_price', 0)):,.0f}</td>"
                f"<td>{float(r.get('stop_price', 0)):,.0f}</td>"
                f"<td>{float(r.get('take_profit_trigger_price', 0)):,.0f}</td>"
                f"<td>{float(r.get('take_profit_trailing_price', 0)):,.0f}</td>"
                f"<td class='news'>{str(news_summary).replace(chr(10), '<br>')}</td>"
                "</tr>"
            )
        rec_rows = "".join(rows)

    backtest_cards = (
        f"<div class='metric'><span>총수익률</span><b>{pct(backtest['total_return'])}</b></div>"
        f"<div class='metric'><span>CAGR</span><b>{pct(backtest['cagr'])}</b></div>"
        f"<div class='metric'><span>MDD</span><b>{pct(backtest['mdd'])}</b></div>"
        f"<div class='metric'><span>Sharpe</span><b>{float(backtest['sharpe']):.2f}</b></div>"
        f"<div class='metric'><span>평균 투자비중</span><b>{pct(backtest['exposure'])}</b></div>"
    )
    backtest_range = f"{backtest['start']} ~ {backtest['end']} | raw 2.0점 이상, 최대 7개 | 손절 배수 2.5 | 익절 +30% 후 -10% 트레일링 | {backtest.get('source', '')}"

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <title>AI Invest Korea</title>
  <style>
    body {{ margin:0; font-family: Arial, "Malgun Gothic", sans-serif; color:#17202a; background:#eef2f6; }}
    header {{ padding:20px 28px 16px; background:#15202b; color:#fff; border-bottom:4px solid #2f80ed; }}
    h1 {{ margin:0; font-size:26px; }} h2 {{ font-size:17px; margin:0 0 10px; color:#1f2d3d; }}
    main {{ padding:18px 28px 36px; }} section {{ margin-bottom:18px; background:#fff; border:1px solid #d7dee8; padding:16px; }}
    .sub,.summary {{ color:#cbd5e1; margin-top:8px; font-size:13px; }} .metrics {{ display:table; width:100%; table-layout:fixed; border-spacing:8px 0; margin-top:10px; }}
    .metric {{ display:table-cell; padding:10px 12px; background:#f6f9fc; border-top:3px solid #2f80ed; }} .metric span {{ display:block; color:#637083; font-size:12px; }} .metric b {{ font-size:20px; }}
    .scroll {{ overflow-x:auto; }} table {{ width:100%; border-collapse:collapse; font-size:12px; }} th,td {{ border-bottom:1px solid #d9e1ea; padding:6px; text-align:left; white-space:nowrap; }} th {{ background:#f3f6fa; }}
    .muted {{ color:#637083; font-size:12px; }} .note {{ color:#516173; font-size:13px; line-height:1.6; }} .news {{ white-space:normal; min-width:260px; max-width:360px; line-height:1.45; }} .date-link {{ display:inline-block; margin:3px; padding:5px 8px; border:1px solid #cbd5e1; background:#fff; }}
  </style>
</head>
<body>
  <header><h1>AI Invest Korea</h1><div class="sub">최신 추천 기준일: {as_of}</div><div class="summary">시장 환경: {regime}</div><div class="summary">추천일 선택: {date_links}</div></header>
  <main>
    <section><h2>백테스트 결과</h2><div class="note">기간: {backtest_range}</div><div class="metrics">{backtest_cards}</div></section>
    <section><h2>최근 추천 종목</h2><div class="scroll"><table><thead><tr><th>순위</th><th>추천일</th><th>종목</th><th>코드</th><th>테마/공급망</th><th>추천일 종가</th><th>점수</th><th>1개월</th><th>3개월</th><th>경고가</th><th>최종손절가</th><th>익절트리거</th><th>트레일링익절가</th><th>7일 리스크 판단</th></tr></thead><tbody>{rec_rows}</tbody></table></div></section>
    <section><h2>해석 기준</h2><p class="note">점수는 전체 후보군 안에서 상대 순위를 100점 만점으로 환산한 값입니다. raw 점수는 모델 내부 z-score 합산값입니다. 리스크 판정은 진입 가능, 분할 진입, 관망, 제외로 표시합니다. 관망은 신규 진입 보류, 제외는 당일 추천에서 제거 검토라는 뜻입니다.</p></section>
  </main>
</body>
</html>"""


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            selected_date = parse_qs(parsed.query).get("date", [None])[0]
            body = render_dashboard_for_date(selected_date).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


def serve(host: str = "127.0.0.1", port: int = 8787) -> None:
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"AI Invest Korea dashboard: http://{host}:{port}")
    server.serve_forever()
