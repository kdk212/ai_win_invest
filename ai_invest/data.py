from __future__ import annotations

from datetime import date, timedelta

import FinanceDataReader as fdr
import pandas as pd
import yfinance as yf
from pykrx import stock

from .config import DATA_DIR
from .utils import safe_to_csv, ymd


def previous_calendar_day(days: int) -> date:
    return date.today() - timedelta(days=days)


def get_market_tickers(market: str = "ALL", as_of: date | None = None) -> list[str]:
    target = ymd(as_of or date.today())
    tickers: list[str] = []
    try:
        if market in ("ALL", "KOSPI"):
            tickers.extend(stock.get_market_ticker_list(target, market="KOSPI"))
        if market in ("ALL", "KOSDAQ"):
            tickers.extend(stock.get_market_ticker_list(target, market="KOSDAQ"))
    except Exception:
        tickers = []
    if tickers:
        return sorted(set(tickers))

    listing = fdr.StockListing("KRX")
    if market == "KOSPI":
        listing = listing[listing["Market"] == "KOSPI"]
    elif market == "KOSDAQ":
        listing = listing[listing["Market"] == "KOSDAQ"]
    return listing["Code"].astype(str).tolist()


def get_name(ticker: str) -> str:
    try:
        name = stock.get_market_ticker_name(ticker)
        if name:
            return name
    except Exception:
        pass
    try:
        listing = fdr.StockListing("KRX")
        row = listing[listing["Code"].astype(str) == ticker]
        if not row.empty:
            return str(row.iloc[0]["Name"])
    except Exception:
        pass
    return ticker


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Change": "change_pct",
        "시가": "open",
        "고가": "high",
        "저가": "low",
        "종가": "close",
        "거래량": "volume",
        "거래대금": "trading_value",
        "등락률": "change_pct",
    }
    out = df.rename(columns=rename_map).copy()
    if "trading_value" not in out.columns and {"close", "volume"}.issubset(out.columns):
        out["trading_value"] = out["close"] * out["volume"]
    out.index = pd.to_datetime(out.index)
    return out


def get_ohlcv(ticker: str, start: str, end: str, cache: bool = True) -> pd.DataFrame:
    cache_path = DATA_DIR / "cache" / f"ohlcv_{ticker}_{start}_{end}.csv"
    if cache and cache_path.exists():
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    try:
        df = stock.get_market_ohlcv_by_date(ymd(start), ymd(end), ticker)
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        df = fdr.DataReader(ticker, start, end)
    if df.empty:
        return df

    out = _normalize_ohlcv(df)
    if cache:
        safe_to_csv(out, cache_path, encoding="utf-8-sig")
    return out


def get_fundamentals(as_of: str) -> pd.DataFrame:
    cache_path = DATA_DIR / "cache" / f"fundamental_{ymd(as_of)}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, index_col=0, dtype=str)

    try:
        df = stock.get_market_fundamental_by_ticker(ymd(as_of), market="ALL")
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df.index = df.index.astype(str)
    safe_to_csv(df, cache_path, encoding="utf-8-sig")
    return df


def get_market_snapshot(as_of: str) -> pd.DataFrame:
    cache_path = DATA_DIR / "cache" / f"market_snapshot_{ymd(as_of)}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, index_col=0, dtype=str)

    frames = []
    try:
        for market in ("KOSPI", "KOSDAQ"):
            df = stock.get_market_cap_by_ticker(ymd(as_of), market=market)
            if not df.empty and "시가총액" in df.columns:
                normalized = df.rename(
                    columns={
                        "종가": "close",
                        "거래량": "volume",
                        "거래대금": "trading_value",
                        "시가총액": "market_cap",
                    }
                )
                normalized["market"] = market
                frames.append(normalized)
    except Exception:
        frames = []

    if frames:
        snapshot = pd.concat(frames)
    else:
        listing = fdr.StockListing("KRX")
        snapshot = listing.set_index("Code").rename(
            columns={
                "Name": "name",
                "Market": "market",
                "Close": "close",
                "Volume": "volume",
                "Amount": "trading_value",
                "Marcap": "market_cap",
            }
        )
    snapshot.index = snapshot.index.astype(str)
    safe_to_csv(snapshot, cache_path, encoding="utf-8-sig")
    return snapshot


def get_price_panel(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    frames: list[pd.Series] = []
    for ticker in tickers:
        try:
            df = get_ohlcv(ticker, start, end)
        except Exception:
            continue
        if not df.empty and "close" in df:
            frames.append(df["close"].rename(ticker))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def get_macro_proxy_prices(symbols: dict[str, str], period: str = "1y") -> pd.DataFrame:
    cache_path = DATA_DIR / "cache" / "macro_proxy_prices.csv"
    try:
        end = date.today()
        start = end - timedelta(days=430)
        frames = []
        for name, symbol in symbols.items():
            df = fdr.DataReader(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            if not df.empty and "Close" in df.columns:
                frames.append(df["Close"].rename(name))
        if frames:
            close = pd.concat(frames, axis=1).sort_index()
            safe_to_csv(close, cache_path, encoding="utf-8-sig")
            return close
    except Exception:
        pass

    try:
        raw = yf.download(
            list(symbols.values()),
            period=period,
            progress=False,
            auto_adjust=True,
            threads=True,
        )
    except Exception:
        if cache_path.exists():
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return pd.DataFrame()

    if raw.empty:
        if cache_path.exists():
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return pd.DataFrame()

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    reverse = {symbol: name for name, symbol in symbols.items()}
    close = close.rename(columns=reverse)
    close.index = pd.to_datetime(close.index)
    safe_to_csv(close, cache_path, encoding="utf-8-sig")
    return close


def get_index_ohlcv(index_code: str, start: str, end: str, cache: bool = True) -> pd.DataFrame:
    cache_path = DATA_DIR / "cache" / f"index_ohlcv_{index_code}_{start}_{end}.csv"
    if cache and cache_path.exists():
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    try:
        df = stock.get_index_ohlcv_by_date(ymd(start), ymd(end), index_code)
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        return df

    out = _normalize_ohlcv(df)
    if cache:
        safe_to_csv(out, cache_path, encoding="utf-8-sig")
    return out


def get_index_returns(start_date: date, end_date: date | None = None) -> dict[str, float | str]:
    end_date = end_date or date.today()
    start = (start_date - timedelta(days=7)).isoformat()
    end = end_date.isoformat()
    indexes = {"kospi": "1001", "kosdaq": "2001"}
    result: dict[str, float | str] = {}
    for key, code in indexes.items():
        frame = get_index_ohlcv(code, start, end)
        if frame.empty or "close" not in frame:
            result[f"{key}_return"] = 0.0
            result[f"{key}_latest_date"] = ""
            continue
        frame = frame[frame.index.date >= start_date].dropna(subset=["close"])
        if frame.empty:
            result[f"{key}_return"] = 0.0
            result[f"{key}_latest_date"] = ""
            continue
        first_close = float(frame.iloc[0]["close"])
        latest_close = float(frame.iloc[-1]["close"])
        result[f"{key}_return"] = latest_close / first_close - 1 if first_close else 0.0
        result[f"{key}_latest_date"] = frame.index[-1].date().isoformat()
    return result
