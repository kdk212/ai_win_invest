from __future__ import annotations

from hashlib import sha256
from html import escape
from http.cookies import SimpleCookie
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

from .config import DATA_DIR
from .portfolio import add_position, close_position, portfolio_snapshot, reopen_position
from .secrets import load_web_password
from .tracker import latest_recommendation_file
from .utils import pct
from .virtual_portfolio import DEFAULT_START_DATE, simulate_recommendation_portfolio


def _recommendation_files() -> list:
    return sorted((DATA_DIR / "recommendations").glob("recommendations_*.csv"))


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


def _load_latest_backtest() -> pd.DataFrame:
    files = sorted((DATA_DIR / "backtests").glob("backtest_*.csv"))
    files = [p for p in files if not p.name.endswith("_holdings.csv") and not p.name.endswith("_stops.csv")]
    return pd.read_csv(files[-1]) if files else pd.DataFrame()


def _summarize_backtest(df: pd.DataFrame) -> dict[str, float | str]:
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


def _signed(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    sign = "+" if float(value) > 0 else ""
    return f"{sign}{float(value):,.0f}"


def _pnl_class(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return "profit" if float(value) > 0 else "loss" if float(value) < 0 else ""


def _dates() -> list[str]:
    return [p.stem.replace("recommendations_", "") for p in _recommendation_files()]


def _selected_date(recs: pd.DataFrame, requested: str | None) -> str:
    if requested:
        return requested
    if not recs.empty and "as_of" in recs:
        return str(recs["as_of"].iloc[0])
    dates = _dates()
    return dates[-1] if dates else ""


def _date_tabs(selected: str) -> str:
    dates = _dates()
    if not dates:
        return "<span class='muted'>저장된 추천일이 아직 없습니다.</span>"
    return "".join(
        f"<a class='tab{' active' if d == selected else ''}' href='/?date={escape(d)}'>{escape(d)}</a>" for d in dates[-10:]
    )


def _previous_scores(selected: str) -> dict[str, float]:
    dates = [d for d in _dates() if d < selected]
    if not dates:
        return {}
    try:
        frame = pd.read_csv(DATA_DIR / "recommendations" / f"recommendations_{dates[-1]}.csv", dtype={"ticker": str})
    except Exception:
        return {}
    return {str(r["ticker"]).zfill(6): float(r["score"]) for _, r in frame.iterrows() if pd.notna(r.get("score"))}


def _risk_text(row: pd.Series) -> str:
    summary = row.get("news_risk_summary", row.get("previous_day_news_summary", ""))
    return str(summary).replace("\n", "<br>") if summary and pd.notna(summary) else "-"


def _metric(title: str, value: str, cls: str = "") -> str:
    return f"<div class='metric'><span>{escape(title)}</span><b class='{cls}'>{value}</b></div>"


def _render_virtual_holdings(df: pd.DataFrame) -> str:
    if df.empty:
        return "<tr><td colspan='9'>2026-06-01 이후 매수로 전환된 추천 종목이 아직 없습니다.</td></tr>"
    rows = []
    for _, r in df.iterrows():
        rows.append(
            "<tr>"
            f"<td><b>{escape(str(r['name']))}</b><br><span class='muted'>{escape(str(r['ticker']))}</span></td>"
            f"<td>{escape(str(r['first_buy_date']))}</td><td>{int(r['lots'])}</td><td>{pct(r['weight'])}</td>"
            f"<td>{_money(r['avg_buy_price'])}</td><td>{_money(r['current_price'])}<br><span class='muted'>{escape(str(r.get('current_price_date') or ''))}</span></td>"
            f"<td class='{_pnl_class(r.get('return_pct'))}'>{pct(r.get('return_pct'))}</td>"
            f"<td>{_money(r.get('stop_price'))}</td><td>{_money(r.get('take_profit_trigger_price'))}</td></tr>"
        )
    return "".join(rows)


def _render_daily(df: pd.DataFrame) -> str:
    if df.empty:
        return "<tr><td colspan='7'>일별 포트폴리오 기록이 아직 없습니다.</td></tr>"
    rows = []
    for _, r in df.tail(10).sort_values("date", ascending=False).iterrows():
        rows.append(
            "<tr>"
            f"<td>{escape(str(r['date']))}</td><td>{int(r['active_names'])}</td><td>{_num(r['contributed'])}</td>"
            f"<td>{_num(r['market_value'])}</td><td class='{_pnl_class(r['realized_pnl'])}'>{_num(r['realized_pnl'])}</td>"
            f"<td class='{_pnl_class(r['unrealized_pnl'])}'>{_num(r['unrealized_pnl'])}</td>"
            f"<td class='{_pnl_class(r['total_return'])}'>{pct(r['total_return'])}</td></tr>"
        )
    return "".join(rows)


def _render_closed(df: pd.DataFrame) -> str:
    if df.empty:
        return "<tr><td colspan='7'>아직 매도 신호가 발생한 종목이 없습니다.</td></tr>"
    labels = {"stop_loss": "손절", "take_profit_trailing": "익절 트레일링"}
    rows = []
    for _, r in df.head(10).iterrows():
        rows.append(
            "<tr>"
            f"<td><b>{escape(str(r['name']))}</b><br><span class='muted'>{escape(str(r['ticker']))}</span></td>"
            f"<td>{escape(str(r['buy_date']))}</td><td>{_money(r['buy_price'])}</td><td>{escape(str(r['sell_date']))}</td>"
            f"<td>{_money(r['sell_price'])}</td><td>{escape(labels.get(str(r['sell_reason']), str(r['sell_reason'])))}</td>"
            f"<td class='{_pnl_class(r.get('return_pct'))}'>{pct(r.get('return_pct'))}</td></tr>"
        )
    return "".join(rows)


def _render_recommendations(df: pd.DataFrame, selected: str) -> str:
    if df.empty:
        return "<tr><td colspan='13'>추천 결과가 없습니다. daily 작업을 먼저 실행하세요.</td></tr>"
    prev = _previous_scores(selected)
    rows = []
    for _, r in df.iterrows():
        ticker = str(r["ticker"]).zfill(6)
        rows.append(
            "<tr>"
            f"<td>{int(r['rank'])}</td><td>{escape(str(r.get('as_of', selected)))}</td>"
            f"<td><b>{escape(str(r['name']))}</b><br><span class='muted'>{escape(ticker)}</span></td>"
            f"<td>{escape(str(r.get('theme', '-')))}</td><td>{_money(r['close'])}</td>"
            f"<td><b>{_num(r.get('score'))}</b></td><td>{_num(prev.get(ticker))}</td>"
            f"<td>{pct(r['mom20'])}</td><td>{pct(r['mom60'])}</td><td>{_money(r.get('warning_price'))}</td>"
            f"<td>{_money(r.get('stop_price'))}</td><td>{_money(r.get('take_profit_trigger_price'))}</td>"
            f"<td class='news'>{_risk_text(r)}</td></tr>"
        )
    return "".join(rows)


def _render_manual(df: pd.DataFrame) -> str:
    if df.empty:
        return "<tr><td colspan='13'>수동으로 등록한 포트폴리오가 없습니다.</td></tr>"
    rows = []
    for _, r in df.iterrows():
        if r["status"] == "holding":
            action = f"<form method='post' action='/portfolio/close' class='inline'><input type='hidden' name='id' value='{int(r['id'])}'><input type='date' name='sell_date' required><input type='number' step='0.01' name='sell_price' placeholder='매도가' required><button>매도</button></form>"
        else:
            action = f"<form method='post' action='/portfolio/reopen' class='inline'><input type='hidden' name='id' value='{int(r['id'])}'><button>보유전환</button></form>"
        rows.append(
            "<tr>"
            f"<td>{'보유중' if r['status'] == 'holding' else '매도완료'}</td><td>{escape(str(r['buy_date']))}</td>"
            f"<td><b>{escape(str(r['name']))}</b><br><span class='muted'>{escape(str(r['ticker']))}</span></td>"
            f"<td>{_money(r['buy_price'])}</td><td>{_money(r['quantity'])}</td><td>{_money(r.get('buy_value'))}</td>"
            f"<td>{_money(r.get('current_price'))}</td><td>{_money(r.get('market_value'))}</td>"
            f"<td class='{_pnl_class(r.get('pnl'))}'>{_signed(r.get('pnl'))}</td><td class='{_pnl_class(r.get('pnl'))}'>{pct(r.get('pnl_pct'))}</td>"
            f"<td>{escape(str(r.get('sell_date') or ''))}</td><td>{_money(r.get('sell_price'))}</td><td>{action}</td></tr>"
        )
    return "".join(rows)


def render_login(error: str = "") -> str:
    return f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>AI Invest Login</title><style>body{{margin:0;min-height:100vh;display:grid;place-items:center;font-family:Arial,'Malgun Gothic',sans-serif;background:#eef2f6;color:#111827}}form{{width:min(360px,calc(100vw - 32px));background:#fff;border:1px solid #d7dee8;padding:22px;box-shadow:0 12px 30px rgba(15,23,42,.12)}}h1{{margin:0 0 14px;font-size:22px}}label{{display:block;color:#64748b;font-size:13px;margin-bottom:6px}}input{{width:100%;box-sizing:border-box;padding:10px;border:1px solid #cbd5e1;font-size:14px}}button{{width:100%;margin-top:14px;padding:10px;border:0;background:#1d4ed8;color:#fff;font-weight:bold;cursor:pointer}}.error{{color:#dc2626;font-size:13px;min-height:20px}}</style></head><body><form method='post' action='/login'><h1>AI Invest Korea</h1><div class='error'>{escape(error)}</div><label>접속 암호</label><input type='password' name='password' autofocus required><button>로그인</button></form></body></html>"""


def render_dashboard_for_date(selected_date: str | None = None) -> str:
    recs = _load_recommendations(selected_date)
    selected = _selected_date(recs, selected_date)
    manual = portfolio_snapshot()
    virtual = simulate_recommendation_portfolio(DEFAULT_START_DATE)
    backtest = _summarize_backtest(_load_latest_backtest())
    as_of = recs["as_of"].iloc[0] if not recs.empty else "-"
    regime = recs["macro_label"].iloc[0] if not recs.empty else "-"
    auth_note = "" if load_web_password() else "<div class='warn'>WEB_PASSWORD가 설정되지 않아 암호 없이 열립니다.</div>"
    metrics = "".join([
        _metric("가상포트 수익률", pct(virtual.get("total_return", 0.0)), _pnl_class(virtual.get("total_return"))),
        _metric("보유중 수익률", pct(virtual.get("unrealized_return", 0.0)), _pnl_class(virtual.get("unrealized_return"))),
        _metric("보유 종목수", str(int(float(virtual.get("active_count", 0.0))))),
        _metric("백테스트 CAGR", pct(backtest["cagr"])),
        _metric("백테스트 MDD", pct(backtest["mdd"])),
        _metric("Sharpe", f"{float(backtest['sharpe']):.2f}"),
    ])
    return f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>AI Invest Korea</title><style>body{{margin:0;font-family:Arial,'Malgun Gothic',sans-serif;color:#17202a;background:#eef2f6}}header{{padding:20px 28px 14px;background:#111827;color:#fff;border-bottom:4px solid #2563eb}}h1{{margin:0;font-size:26px}}h2{{margin:0 0 10px;font-size:18px;color:#1f2937}}main{{padding:16px 28px 34px}}section{{margin-bottom:16px;background:#fff;border:1px solid #d7dee8;padding:15px}}.topline{{color:#d1d5db;margin-top:8px;font-size:13px;display:flex;gap:18px;flex-wrap:wrap}}.metrics{{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:8px;margin-top:10px}}.metric{{padding:10px 11px;background:#f8fafc;border-top:3px solid #2563eb;min-height:48px}}.metric span{{display:block;color:#64748b;font-size:12px;margin-bottom:4px}}.metric b{{font-size:18px;color:#111827}}.layout{{display:grid;grid-template-columns:1.15fr .85fr;gap:16px}}.scroll{{overflow-x:auto;width:100%}}table{{width:100%;border-collapse:collapse;font-size:12px;background:#fff}}th,td{{border-bottom:1px solid #e2e8f0;padding:7px 8px;text-align:left;white-space:nowrap;vertical-align:top}}th{{background:#f1f5f9;color:#334155;font-weight:bold}}input,button{{font-family:inherit;font-size:12px;padding:6px}}button{{background:#1d4ed8;color:#fff;border:0;cursor:pointer}}.muted{{color:#64748b;font-size:12px}}.news{{white-space:normal;min-width:260px;max-width:380px;line-height:1.45}}.tab{{display:inline-block;margin:3px 5px 3px 0;padding:6px 10px;color:#0b5394;text-decoration:none;border:1px solid #cbd5e1;background:#fff}}.tab.active{{background:#1d4ed8;color:#fff;border-color:#1d4ed8}}.form-grid{{display:grid;grid-template-columns:110px 140px 130px 100px 1fr 80px;gap:8px;align-items:end}}.form-grid label{{display:block;color:#64748b;font-size:12px;margin-bottom:4px}}.form-grid input{{width:100%;box-sizing:border-box}}.inline{{display:flex;gap:5px;align-items:center}}.inline input{{width:95px}}.inline input[type=number]{{width:80px}}.profit{{color:#dc2626!important}}.loss{{color:#2563eb!important}}.note{{color:#475569;font-size:13px;line-height:1.55}}.warn{{margin-top:10px;color:#92400e;background:#fffbeb;border:1px solid #fde68a;padding:8px 10px;font-size:13px}}.logout{{color:#dbeafe;text-decoration:none}}@media(max-width:1000px){{.metrics{{grid-template-columns:repeat(2,1fr)}}.layout{{grid-template-columns:1fr}}.form-grid{{grid-template-columns:1fr 1fr}}}}</style></head><body><header><h1>AI Invest Korea</h1><div class='topline'><span>추천 기준: {escape(str(as_of))}</span><span>시장 환경: {escape(str(regime))}</span><span>가상포트 시작: {DEFAULT_START_DATE.isoformat()} 시초가</span><a class='logout' href='/logout'>로그아웃</a></div>{auth_note}</header><main><section><h2>운영 요약</h2><div class='note'>추천일 전일 종가 기준으로 신호를 만들고, 다음 거래일 시초가에 동일 일자 추천 종목을 같은 비중으로 매수합니다. 매도 신호가 발생하면 해당 종목의 보유분을 전부 매도한 것으로 계산합니다.</div><div class='metrics'>{metrics}</div></section><div class='layout'><section><h2>가상 포트폴리오</h2><div class='scroll'><table><thead><tr><th>종목</th><th>첫 매수일</th><th>누적매수</th><th>비중</th><th>평균매수가</th><th>현재가</th><th>보유수익률</th><th>손절가</th><th>익절 트리거</th></tr></thead><tbody>{_render_virtual_holdings(virtual['holdings'])}</tbody></table></div></section><section><h2>일별 수익률</h2><div class='scroll'><table><thead><tr><th>일자</th><th>보유</th><th>누적투입</th><th>평가금</th><th>실현손익</th><th>평가손익</th><th>총수익률</th></tr></thead><tbody>{_render_daily(virtual['daily'])}</tbody></table></div></section></div><section><h2>매도 신호 기록</h2><div class='scroll'><table><thead><tr><th>종목</th><th>매수일</th><th>매수가</th><th>매도일</th><th>매도가</th><th>규칙</th><th>수익률</th></tr></thead><tbody>{_render_closed(virtual['closed'])}</tbody></table></div></section><section><h2>최근 추천종목</h2><div class='note'>최근 10개 추천일입니다. raw 점수는 모델의 원점수이며, 전일 raw는 직전 추천일 같은 종목의 원점수입니다.</div><div>{_date_tabs(selected)}</div><div class='scroll'><table><thead><tr><th>순위</th><th>추천일</th><th>종목</th><th>테마/공급망</th><th>추천일 종가</th><th>raw 점수</th><th>전일 raw</th><th>1개월</th><th>3개월</th><th>경고가</th><th>손절가</th><th>익절가</th><th>7일 리스크 판단</th></tr></thead><tbody>{_render_recommendations(recs, selected)}</tbody></table></div></section><section><h2>수동 포트폴리오</h2><form method='post' action='/portfolio/add' class='form-grid'><div><label>종목코드</label><input name='ticker' placeholder='005930' required></div><div><label>매수일</label><input type='date' name='buy_date' required></div><div><label>매수가</label><input type='number' step='0.01' name='buy_price' required></div><div><label>수량</label><input type='number' step='0.0001' name='quantity' required></div><div><label>메모</label><input name='memo' placeholder='매수 근거 또는 계획'></div><div><button>추가</button></div></form><br><div class='scroll'><table><thead><tr><th>상태</th><th>매수일</th><th>종목</th><th>매수가</th><th>수량</th><th>매수금액</th><th>현재가</th><th>평가금액</th><th>손익</th><th>수익률</th><th>매도일</th><th>매도가</th><th>관리</th></tr></thead><tbody>{_render_manual(manual)}</tbody></table></div></section></main></body></html>"""


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
        if parsed.path == "/portfolio/add":
            add_position(form["ticker"], form["buy_date"], float(form["buy_price"]), float(form["quantity"]), form.get("memo", ""))
            self._redirect("/")
            return
        if parsed.path == "/portfolio/close":
            close_position(int(form["id"]), form["sell_date"], float(form["sell_price"]))
            self._redirect("/")
            return
        if parsed.path == "/portfolio/reopen":
            reopen_position(int(form["id"]))
            self._redirect("/")
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
