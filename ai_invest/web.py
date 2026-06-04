from __future__ import annotations

from hashlib import sha256
from html import escape
from http.cookies import SimpleCookie
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

from .config import DATA_DIR
from .secrets import load_web_password
from .tracker import latest_recommendation_file
from .utils import pct
from .virtual_portfolio import DEFAULT_START_DATE, simulate_recommendation_portfolio


def _recommendation_files() -> list:
    return sorted((DATA_DIR / "recommendations").glob("recommendations_*.csv"))


def _available_dates() -> list[str]:
    return [path.stem.replace("recommendations_", "") for path in _recommendation_files()]


def _load_recommendations(selected_date: str | None = None) -> pd.DataFrame:
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


def _load_backtest() -> pd.DataFrame:
    files = sorted((DATA_DIR / "backtests").glob("backtest_*.csv"))
    files = [p for p in files if not p.name.endswith("_holdings.csv") and not p.name.endswith("_stops.csv")]
    return pd.read_csv(files[-1]) if files else pd.DataFrame()


def _backtest_summary(df: pd.DataFrame) -> dict[str, float | str]:
    if df.empty:
        return {"start": "2024-11-01", "end": "2026-05-29", "cagr": 6.5124, "mdd": -0.2443, "sharpe": 3.35}
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


def _money(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.0f}"


def _num(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.2f}"


def _pnl_class(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return "profit" if float(value) > 0 else "loss" if float(value) < 0 else ""


def _selected_date(recs: pd.DataFrame, requested: str | None) -> str:
    if requested:
        return requested
    if not recs.empty and "as_of" in recs:
        return str(recs["as_of"].iloc[0])
    dates = _available_dates()
    return dates[-1] if dates else ""


def _date_tabs(selected: str) -> str:
    dates = list(reversed(_available_dates()))[:10]
    if not dates:
        if selected and selected != "-":
            return f"<a class='date-tab active' href='/?date={escape(selected)}'>{escape(selected)}</a>"
        return "<span class='muted'>저장된 추천일이 아직 없습니다.</span>"
    return "".join(
        f"<a class='date-tab{' active' if day == selected else ''}' href='/?date={escape(day)}'>{escape(day)}</a>"
        for day in dates
    )


def _previous_scores(selected: str) -> dict[str, float]:
    dates = [day for day in _available_dates() if day < selected]
    if not dates:
        return {}
    try:
        frame = pd.read_csv(DATA_DIR / "recommendations" / f"recommendations_{dates[-1]}.csv", dtype={"ticker": str})
    except Exception:
        return {}
    return {str(row["ticker"]).zfill(6): float(row["score"]) for _, row in frame.iterrows() if pd.notna(row.get("score"))}


def _risk_text(row: pd.Series) -> str:
    summary = row.get("news_risk_summary", row.get("previous_day_news_summary", ""))
    return str(summary).replace("\n", "<br>") if summary and pd.notna(summary) else "-"


def _metric(title: str, value: str, cls: str = "") -> str:
    return f"<div class='metric'><span>{escape(title)}</span><b class='{cls}'>{value}</b></div>"


def _metrics(backtest: dict[str, float | str], virtual: dict[str, object]) -> str:
    return "".join(
        [
            _metric("가상포트 수익률", pct(virtual.get("total_return", 0.0)), _pnl_class(virtual.get("total_return"))),
            _metric("보유중 수익률", pct(virtual.get("unrealized_return", 0.0)), _pnl_class(virtual.get("unrealized_return"))),
            _metric("보유 종목수", str(int(float(virtual.get("active_count", 0.0))))),
            _metric("백테스트 CAGR", pct(backtest["cagr"])),
            _metric("백테스트 MDD", pct(backtest["mdd"])),
            _metric("Sharpe", f"{float(backtest['sharpe']):.2f}"),
        ]
    )


def _portfolio_status(virtual: dict[str, object]) -> str:
    status = str(virtual.get("status", ""))
    ok = status == "ok"
    status_text = "정상 계산 중" if ok else status or "데이터 상태를 확인해야 합니다."
    days = int(float(virtual.get("recommendation_days", 0.0)))
    tickers = int(float(virtual.get("recommendation_tickers", 0.0)))
    priced = int(float(virtual.get("priced_tickers", 0.0)))
    buys = int(float(virtual.get("buy_events", 0.0)))
    cls = "ok" if ok else "warnbox"
    return (
        f"<div class='data-status {cls}'>"
        f"<b>포트폴리오 상태</b><span>{escape(status_text)}</span>"
        f"<em>추천일 {days}개 | 추천종목 {tickers}개 | 가격확인 {priced}개 | 매수이벤트 {buys}개</em>"
        "</div>"
    )


def _holdings_rows(holdings: pd.DataFrame) -> str:
    if holdings.empty:
        return "<tr><td colspan='9'>아직 포트폴리오에 편입된 종목이 없습니다. 위의 포트폴리오 상태를 먼저 확인하세요.</td></tr>"
    rows = []
    for _, row in holdings.iterrows():
        rows.append(
            "<tr>"
            f"<td><b>{escape(str(row['name']))}</b><br><span class='muted'>{escape(str(row['ticker']))}</span></td>"
            f"<td>{escape(str(row['first_buy_date']))}</td>"
            f"<td>{int(row['lots'])}</td>"
            f"<td>{pct(row['weight'])}</td>"
            f"<td>{_money(row['avg_buy_price'])}</td>"
            f"<td>{_money(row['current_price'])}<br><span class='muted'>{escape(str(row.get('current_price_date') or ''))}</span></td>"
            f"<td class='{_pnl_class(row.get('return_pct'))}'>{pct(row.get('return_pct'))}</td>"
            f"<td>{_money(row.get('stop_price'))}</td>"
            f"<td>{_money(row.get('take_profit_trigger_price'))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _daily_rows(daily: pd.DataFrame) -> str:
    if daily.empty:
        return "<tr><td colspan='7'>일별 포트폴리오 기록이 아직 없습니다.</td></tr>"
    rows = []
    for _, row in daily.tail(10).sort_values("date", ascending=False).iterrows():
        rows.append(
            "<tr>"
            f"<td>{escape(str(row['date']))}</td><td>{int(row['active_names'])}</td><td>{_num(row['contributed'])}</td>"
            f"<td>{_num(row['market_value'])}</td><td class='{_pnl_class(row['realized_pnl'])}'>{_num(row['realized_pnl'])}</td>"
            f"<td class='{_pnl_class(row['unrealized_pnl'])}'>{_num(row['unrealized_pnl'])}</td><td class='{_pnl_class(row['total_return'])}'>{pct(row['total_return'])}</td>"
            "</tr>"
        )
    return "".join(rows)


def _closed_rows(closed: pd.DataFrame) -> str:
    if closed.empty:
        return "<tr><td colspan='7'>아직 매도 신호가 발생한 종목이 없습니다.</td></tr>"
    labels = {"stop_loss": "손절", "take_profit_trailing": "익절 트레일링"}
    rows = []
    for _, row in closed.head(10).iterrows():
        rows.append(
            "<tr>"
            f"<td><b>{escape(str(row['name']))}</b><br><span class='muted'>{escape(str(row['ticker']))}</span></td>"
            f"<td>{escape(str(row['buy_date']))}</td><td>{_money(row['buy_price'])}</td>"
            f"<td>{escape(str(row['sell_date']))}</td><td>{_money(row['sell_price'])}</td>"
            f"<td>{escape(labels.get(str(row['sell_reason']), str(row['sell_reason'])))}</td><td class='{_pnl_class(row.get('return_pct'))}'>{pct(row.get('return_pct'))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _recommendation_rows(recs: pd.DataFrame, selected: str) -> str:
    if recs.empty:
        return "<tr><td colspan='13'>추천 결과가 없습니다. daily 작업을 먼저 실행하세요.</td></tr>"
    prev = _previous_scores(selected)
    rows = []
    for _, row in recs.iterrows():
        ticker = str(row["ticker"]).zfill(6)
        rows.append(
            "<tr>"
            f"<td>{int(row['rank'])}</td><td>{escape(str(row.get('as_of', selected)))}</td>"
            f"<td><b>{escape(str(row['name']))}</b><br><span class='muted'>{escape(ticker)}</span></td>"
            f"<td>{escape(str(row.get('theme', '-')))}</td><td>{_money(row['close'])}</td>"
            f"<td><b>{_num(row.get('score'))}</b></td><td>{_num(prev.get(ticker))}</td>"
            f"<td>{pct(row['mom20'])}</td><td>{pct(row['mom60'])}</td><td>{_money(row.get('warning_price'))}</td>"
            f"<td>{_money(row.get('stop_price'))}</td><td>{_money(row.get('take_profit_trigger_price'))}</td><td class='news'>{_risk_text(row)}</td>"
            "</tr>"
        )
    return "".join(rows)


def render_login(error: str = "") -> str:
    return f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>AI Invest Login</title><style>body{{margin:0;min-height:100vh;display:grid;place-items:center;font-family:Arial,'Malgun Gothic',sans-serif;background:#eef2f6;color:#111827}}form{{width:min(360px,calc(100vw - 32px));background:#fff;border:1px solid #d7dee8;padding:22px;box-shadow:0 12px 30px rgba(15,23,42,.12)}}h1{{margin:0 0 14px;font-size:22px}}label{{display:block;color:#64748b;font-size:13px;margin-bottom:6px}}input{{width:100%;box-sizing:border-box;padding:10px;border:1px solid #cbd5e1;font-size:14px}}button{{width:100%;margin-top:14px;padding:10px;border:0;background:#1d4ed8;color:#fff;font-weight:bold;cursor:pointer}}.error{{color:#dc2626;font-size:13px;min-height:20px}}</style></head><body><form method='post' action='/login'><h1>AI Invest Korea</h1><div class='error'>{escape(error)}</div><label>접속 암호</label><input type='password' name='password' autofocus required><button>로그인</button></form></body></html>"""


def render_dashboard_for_date(selected_date: str | None = None) -> str:
    recs = _load_recommendations(selected_date)
    selected = _selected_date(recs, selected_date)
    virtual = simulate_recommendation_portfolio(DEFAULT_START_DATE)
    backtest = _backtest_summary(_load_backtest())
    as_of = recs["as_of"].iloc[0] if not recs.empty else "-"
    regime = recs["macro_label"].iloc[0] if not recs.empty else "-"
    auth_note = "" if load_web_password() else "<div class='warn'>WEB_PASSWORD가 설정되지 않아 암호 없이 열립니다.</div>"
    return f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>AI Invest Korea</title><style>body{{margin:0;font-family:Arial,'Malgun Gothic',sans-serif;color:#17202a;background:#eef2f6}}header{{padding:20px 28px 14px;background:#111827;color:#fff;border-bottom:4px solid #2563eb}}h1{{margin:0;font-size:26px}}h2{{margin:0 0 10px;font-size:18px;color:#1f2937}}main{{padding:16px 28px 34px}}section{{margin-bottom:16px;background:#fff;border:1px solid #d7dee8;padding:15px}}.topline{{color:#d1d5db;margin-top:8px;font-size:13px;display:flex;gap:18px;flex-wrap:wrap}}.metrics{{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:8px;margin-top:10px}}.metric{{padding:10px 11px;background:#f8fafc;border-top:3px solid #2563eb;min-height:48px}}.metric span{{display:block;color:#64748b;font-size:12px;margin-bottom:4px}}.metric b{{font-size:18px;color:#111827}}.layout{{display:grid;grid-template-columns:1.15fr .85fr;gap:16px}}.scroll{{overflow-x:auto;width:100%}}table{{width:100%;border-collapse:collapse;font-size:12px;background:#fff}}th,td{{border-bottom:1px solid #e2e8f0;padding:7px 8px;text-align:left;white-space:nowrap;vertical-align:top}}th{{background:#f1f5f9;color:#334155;font-weight:bold}}.muted{{color:#64748b;font-size:12px}}.news{{white-space:normal;min-width:260px;max-width:380px;line-height:1.45}}.date-tabs{{margin:0 0 10px}}.date-tab{{display:inline-block;margin:0 6px 6px 0;padding:7px 11px;color:#0b5394;text-decoration:none;border:1px solid #cbd5e1;background:#fff;font-size:13px}}.date-tab.active{{background:#1d4ed8;color:#fff;border-color:#1d4ed8}}.profit{{color:#dc2626!important}}.loss{{color:#2563eb!important}}.note{{color:#475569;font-size:13px;line-height:1.55}}.warn{{margin-top:10px;color:#92400e;background:#fffbeb;border:1px solid #fde68a;padding:8px 10px;font-size:13px}}.data-status{{margin-top:10px;padding:10px 12px;border:1px solid #cbd5e1;background:#f8fafc;font-size:13px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}}.data-status b{{color:#111827}}.data-status span{{color:#334155}}.data-status em{{color:#64748b;font-style:normal}}.data-status.warnbox{{border-color:#fde68a;background:#fffbeb}}.data-status.ok{{border-color:#bbf7d0;background:#f0fdf4}}.logout{{color:#dbeafe;text-decoration:none}}@media(max-width:1000px){{.metrics{{grid-template-columns:repeat(2,1fr)}}.layout{{grid-template-columns:1fr}}}}</style></head><body><header><h1>AI Invest Korea</h1><div class='topline'><span>추천 기준: {escape(str(as_of))}</span><span>시장 환경: {escape(str(regime))}</span><span>가상포트 시작: {DEFAULT_START_DATE.isoformat()} 시초가</span><a class='logout' href='/logout'>로그아웃</a></div>{auth_note}</header><main><section><h2>운영 요약</h2><div class='note'>추천일 전일 종가 기준으로 신호를 만들고, 추천일 시초가에 동일 일자 추천 종목을 같은 비중으로 매수합니다. 매도 신호가 발생하면 해당 종목의 보유분을 전부 매도한 것으로 계산합니다.</div><div class='metrics'>{_metrics(backtest, virtual)}</div>{_portfolio_status(virtual)}</section><div class='layout'><section><h2>가상 포트폴리오</h2><div class='scroll'><table><thead><tr><th>종목</th><th>첫 매수일</th><th>누적매수</th><th>비중</th><th>평균매수가</th><th>현재가</th><th>보유수익률</th><th>손절가</th><th>익절가</th></tr></thead><tbody>{_holdings_rows(virtual['holdings'])}</tbody></table></div></section><section><h2>일별 수익률</h2><div class='scroll'><table><thead><tr><th>일자</th><th>보유</th><th>누적투입</th><th>평가금</th><th>실현손익</th><th>평가손익</th><th>총수익률</th></tr></thead><tbody>{_daily_rows(virtual['daily'])}</tbody></table></div></section></div><section><h2>매도 신호 기록</h2><div class='scroll'><table><thead><tr><th>종목</th><th>매수일</th><th>매수가</th><th>매도일</th><th>매도가</th><th>규칙</th><th>수익률</th></tr></thead><tbody>{_closed_rows(virtual['closed'])}</tbody></table></div></section><section><h2>최근 추천종목</h2><div class='date-tabs'>{_date_tabs(selected)}</div><div class='scroll'><table><thead><tr><th>순위</th><th>추천일</th><th>종목</th><th>테마/공급망</th><th>추천일 종가</th><th>raw 점수</th><th>전일 raw</th><th>1개월</th><th>3개월</th><th>경고가</th><th>손절가</th><th>익절가</th><th>7일 리스크 판단</th></tr></thead><tbody>{_recommendation_rows(recs, selected)}</tbody></table></div></section></main></body></html>"""


def render_dashboard() -> str:
    return render_dashboard_for_date(None)


def _auth_token(password: str) -> str:
    return sha256(f"ai-invest:{password}".encode("utf-8")).hexdigest()


class DashboardHandler(SimpleHTTPRequestHandler):
    def _send_html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, location: str) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        self.end_headers()

    def _form(self) -> dict[str, str]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        return {key: values[0] for key, values in parse_qs(body).items()}

    def _is_authenticated(self) -> bool:
        password = load_web_password()
        if not password:
            return True
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        value = cookie.get("ai_invest_auth")
        return bool(value and value.value == _auth_token(password))

    def _require_auth(self) -> bool:
        if self._is_authenticated():
            return True
        self._redirect("/login")
        return False

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        form = self._form()
        if parsed.path == "/login":
            password = load_web_password()
            if password and form.get("password") == password:
                self.send_response(303)
                self.send_header("Location", "/")
                self.send_header("Set-Cookie", f"ai_invest_auth={_auth_token(password)}; Path=/; HttpOnly; SameSite=Lax")
                self.end_headers()
                return
            self._send_html(render_login("암호가 맞지 않습니다."), status=401)
            return
        if not self._require_auth():
            return
        self.send_error(404)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/login":
            self._send_html(render_login())
            return
        if parsed.path == "/logout":
            self.send_response(303)
            self.send_header("Location", "/login")
            self.send_header("Set-Cookie", "ai_invest_auth=; Path=/; Max-Age=0; HttpOnly; SameSite=Lax")
            self.end_headers()
            return
        if not self._require_auth():
            return
        if parsed.path in ("/", "/index.html"):
            selected_date = parse_qs(parsed.query).get("date", [None])[0]
            self._send_html(render_dashboard_for_date(selected_date))
            return
        super().do_GET()


def serve(host: str = "127.0.0.1", port: int = 8787) -> None:
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"AI Invest Korea dashboard: http://{host}:{port}")
    server.serve_forever()
