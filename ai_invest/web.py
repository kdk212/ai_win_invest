from __future__ import annotations

from html import escape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

from .config import DATA_DIR
from .portfolio import add_position, close_position, portfolio_snapshot, reopen_position
from .tracker import latest_recommendation_file
from .utils import pct


def _load_latest(selected_date: str | None = None) -> pd.DataFrame:
    if selected_date:
        path = DATA_DIR / "recommendations" / f"recommendations_{selected_date}.csv"
        if path.exists():
            return pd.read_csv(path, dtype={"ticker": str})
    path = latest_recommendation_file()
    if path:
        return pd.read_csv(path, dtype={"ticker": str})
    try:
        from .strategy import build_recommendations
        return build_recommendations()
    except Exception:
        return pd.DataFrame()


def _recommendation_files() -> list:
    return sorted((DATA_DIR / "recommendations").glob("recommendations_*.csv"))


def _money(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.0f}"


def _signed(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    sign = "+" if float(value) > 0 else ""
    return f"{sign}{float(value):,.0f}"


def _pnl_class(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return "profit" if float(value) > 0 else "loss" if float(value) < 0 else ""


def _score_100(row: pd.Series) -> float:
    if "score_100" in row and pd.notna(row["score_100"]):
        return float(row["score_100"])
    return min(max(float(row.get("score", 0)) * 10 + 50, 0), 100)


def _backtest_summary() -> dict[str, float | str]:
    files = sorted((DATA_DIR / "backtests").glob("backtest_*.csv"))
    files = [p for p in files if not p.name.endswith("_holdings.csv") and not p.name.endswith("_stops.csv")]
    if not files:
        return {"start": "2024-11-01", "end": "2026-05-29", "cagr": 6.5124, "mdd": -0.2443, "sharpe": 3.35}
    df = pd.read_csv(files[-1])
    equity = df["equity"].astype(float)
    daily = df["daily_return"].astype(float)
    years = max(len(df) / 252, 1 / 252)
    return {
        "start": str(df["date"].iloc[0]),
        "end": str(df["date"].iloc[-1]),
        "cagr": float(equity.iloc[-1] ** (1 / years) - 1),
        "mdd": float((equity / equity.cummax() - 1).min()),
        "sharpe": float((daily.mean() / daily.std()) * (252 ** 0.5)) if float(daily.std()) else 0.0,
    }


def _dates() -> str:
    links = [p.stem.replace("recommendations_", "") for p in _recommendation_files()]
    if not links:
        return "<span class='muted'>저장된 추천일 없음</span>"
    return "".join(f"<a class='date-link' href='/?date={escape(d)}'>{escape(d)}</a>" for d in links[-30:])


def _portfolio_rows(df: pd.DataFrame) -> str:
    if df.empty:
        return "<tr><td colspan='13'>등록된 포트폴리오가 없습니다.</td></tr>"
    rows = []
    for _, r in df.iterrows():
        if r["status"] == "holding":
            action = f"""
            <form method='post' action='/portfolio/close' class='inline'>
              <input type='hidden' name='id' value='{int(r['id'])}'>
              <input type='date' name='sell_date' required>
              <input type='number' step='0.01' name='sell_price' placeholder='매도가' required>
              <button>매도</button>
            </form>"""
        else:
            action = f"""
            <form method='post' action='/portfolio/reopen' class='inline'>
              <input type='hidden' name='id' value='{int(r['id'])}'>
              <button>보유전환</button>
            </form>"""
        rows.append(
            "<tr>"
            f"<td>{'보유중' if r['status'] == 'holding' else '매도완료'}</td>"
            f"<td>{escape(str(r['buy_date']))}</td>"
            f"<td><b>{escape(str(r['name']))}</b><br><span class='muted'>{escape(str(r['ticker']))}</span></td>"
            f"<td>{_money(r['buy_price'])}</td><td>{_money(r['quantity'])}</td><td>{_money(r.get('buy_value'))}</td>"
            f"<td>{_money(r.get('current_price'))}<br><span class='muted'>{escape(str(r.get('current_price_date') or ''))}</span></td>"
            f"<td>{_money(r.get('market_value'))}</td>"
            f"<td class='{_pnl_class(r.get('pnl'))}'>{_signed(r.get('pnl'))}</td>"
            f"<td class='{_pnl_class(r.get('pnl'))}'>{pct(r.get('pnl_pct'))}</td>"
            f"<td>{escape(str(r.get('sell_date') or ''))}</td><td>{_money(r.get('sell_price'))}</td><td>{action}</td>"
            "</tr>"
        )
    return "".join(rows)


def _recommendation_rows(df: pd.DataFrame) -> str:
    if df.empty:
        return "<tr><td colspan='13'>추천 결과가 없습니다. daily 작업을 먼저 실행하세요.</td></tr>"
    rows = []
    for _, r in df.iterrows():
        risk = str(r.get("news_risk_summary", r.get("previous_day_news_summary", ""))).replace("\n", "<br>") or "-"
        rows.append(
            "<tr>"
            f"<td>{int(r['rank'])}</td><td>{escape(str(r['as_of']))}</td>"
            f"<td><b>{escape(str(r['name']))}</b><br><span class='muted'>{escape(str(r['ticker']))}</span></td>"
            f"<td>{escape(str(r.get('theme', '-')))}</td><td>{_money(r['close'])}</td>"
            f"<td><b>{_score_100(r):.1f}/100</b><br><span class='muted'>raw {float(r['score']):.2f}</span></td>"
            f"<td>{pct(r['mom20'])}</td><td>{pct(r['mom60'])}</td>"
            f"<td>{_money(r.get('warning_price'))}</td><td>{_money(r.get('stop_price'))}</td>"
            f"<td>{_money(r.get('take_profit_trigger_price'))}</td><td>{_money(r.get('take_profit_trailing_price'))}</td>"
            f"<td class='news'>{risk}</td></tr>"
        )
    return "".join(rows)


def render_dashboard_for_date(selected_date: str | None = None) -> str:
    recs = _load_latest(selected_date)
    pf = portfolio_snapshot()
    bt = _backtest_summary()
    holding = pf[pf["status"] == "holding"] if not pf.empty else pd.DataFrame()
    buy_sum = float(holding["buy_value"].sum()) if not holding.empty else 0.0
    value_sum = float(holding["market_value"].dropna().sum()) if not holding.empty else 0.0
    pnl = value_sum - buy_sum if buy_sum else 0.0
    pnl_pct = pnl / buy_sum if buy_sum else 0.0
    as_of = recs["as_of"].iloc[0] if not recs.empty else "-"
    regime = recs["macro_label"].iloc[0] if not recs.empty else "-"
    return f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>AI Invest Korea</title>
<style>
body{{margin:0;font-family:Arial,'Malgun Gothic',sans-serif;color:#17202a;background:#eef2f6}}header{{padding:22px 28px 16px;background:#111827;color:#fff;border-bottom:4px solid #2563eb}}h1{{margin:0;font-size:26px}}h2{{margin:0 0 12px;font-size:18px;color:#1f2937}}main{{padding:18px 28px 36px}}section{{margin-bottom:18px;background:#fff;border:1px solid #d7dee8;padding:16px}}.sub,.summary{{color:#d1d5db;margin-top:8px;font-size:13px}}.metrics{{display:grid;grid-template-columns:repeat(5,minmax(130px,1fr));gap:8px;margin-top:12px}}.metric{{padding:11px 12px;background:#f8fafc;border-top:3px solid #2563eb}}.metric span{{display:block;color:#64748b;font-size:12px;margin-bottom:4px}}.metric b{{font-size:19px;color:#111827}}.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{border-bottom:1px solid #e2e8f0;padding:7px 8px;text-align:left;white-space:nowrap;vertical-align:top}}th{{background:#f1f5f9;color:#334155}}input,button{{font-family:inherit;font-size:12px;padding:6px}}button{{background:#1d4ed8;color:#fff;border:0;cursor:pointer}}.muted{{color:#64748b;font-size:12px}}.news{{white-space:normal;min-width:280px;max-width:420px;line-height:1.45}}.date-link{{display:inline-block;margin:3px 5px 3px 0;padding:5px 8px;color:#0b5394;text-decoration:none;border:1px solid #cbd5e1;background:#fff}}.form-grid{{display:grid;grid-template-columns:110px 140px 130px 100px 1fr 80px;gap:8px;align-items:end}}.form-grid label{{display:block;color:#64748b;font-size:12px;margin-bottom:4px}}.form-grid input{{width:100%;box-sizing:border-box}}.inline{{display:flex;gap:5px;align-items:center}}.inline input{{width:95px}}.inline input[type=number]{{width:80px}}.profit{{color:#dc2626!important}}.loss{{color:#2563eb!important}}.note{{color:#475569;font-size:13px;line-height:1.55}}@media(max-width:900px){{.metrics{{grid-template-columns:repeat(2,1fr)}}.form-grid{{grid-template-columns:1fr 1fr}}}}
</style></head><body><header><h1>AI Invest Korea</h1><div class='sub'>추천 기준일: {escape(str(as_of))}</div><div class='summary'>시장 환경: {escape(str(regime))}</div></header><main>
<section><h2>운영 요약</h2><div class='note'>백테스트: {escape(str(bt['start']))} ~ {escape(str(bt['end']))} | raw 2.0 이상 | 최대 7종목 | 손절 배수 2.5 | 익절 +30% 후 -10% 트레일링</div><div class='metrics'><div class='metric'><span>백테스트 CAGR</span><b>{pct(bt['cagr'])}</b></div><div class='metric'><span>백테스트 MDD</span><b>{pct(bt['mdd'])}</b></div><div class='metric'><span>Sharpe</span><b>{float(bt['sharpe']):.2f}</b></div><div class='metric'><span>보유 평가손익</span><b class='{_pnl_class(pnl)}'>{_signed(pnl)}</b></div><div class='metric'><span>보유 수익률</span><b class='{_pnl_class(pnl)}'>{pct(pnl_pct)}</b></div></div></section>
<section><h2>포트폴리오</h2><form method='post' action='/portfolio/add' class='form-grid'><div><label>종목코드</label><input name='ticker' placeholder='005930' required></div><div><label>매수일</label><input type='date' name='buy_date' required></div><div><label>매수가</label><input type='number' step='0.01' name='buy_price' required></div><div><label>수량</label><input type='number' step='0.0001' name='quantity' required></div><div><label>메모</label><input name='memo' placeholder='매수 근거 또는 계획'></div><div><button>추가</button></div></form><br><div class='scroll'><table><thead><tr><th>상태</th><th>매수일</th><th>종목</th><th>매수가</th><th>수량</th><th>매수금액</th><th>현재가</th><th>평가금액</th><th>손익</th><th>수익률</th><th>매도일</th><th>매도가</th><th>관리</th></tr></thead><tbody>{_portfolio_rows(pf)}</tbody></table></div></section>
<section><h2>추천일 선택</h2>{_dates()}</section>
<section><h2>최근 추천 종목</h2><div class='scroll'><table><thead><tr><th>순위</th><th>추천일</th><th>종목</th><th>테마/공급망</th><th>추천일 종가</th><th>점수</th><th>1개월</th><th>3개월</th><th>경고가</th><th>최종손절가</th><th>익절트리거</th><th>트레일링익절가</th><th>7일 리스크 판단</th></tr></thead><tbody>{_recommendation_rows(recs)}</tbody></table></div></section>
<p class='note'>자동 선별 및 포트폴리오 기록 도구입니다. 실제 매매 전 공시, 실적, 거래량, 손절 기준, 본인 위험 한도를 별도로 확인하세요.</p></main></body></html>"""


def render_dashboard() -> str:
    return render_dashboard_for_date(None)


class DashboardHandler(SimpleHTTPRequestHandler):
    def _redirect_home(self) -> None:
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def _form(self) -> dict[str, str]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        return {key: values[0] for key, values in parse_qs(body).items()}

    def do_POST(self) -> None:
        form = self._form()
        path = urlparse(self.path).path
        if path == "/portfolio/add":
            add_position(form["ticker"], form["buy_date"], float(form["buy_price"]), float(form["quantity"]), form.get("memo", ""))
            self._redirect_home(); return
        if path == "/portfolio/close":
            close_position(int(form["id"]), form["sell_date"], float(form["sell_price"])); self._redirect_home(); return
        if path == "/portfolio/reopen":
            reopen_position(int(form["id"])); self._redirect_home(); return
        self.send_error(404)

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
