"""
Stock Analysis Tool - Single stock deep-dive + 3-stock comparison with AI summary
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import os

# ─── Formatting helpers ───────────────────────────────────────────────────────

def fmt_large(n):
    if n is None:
        return "N/A"
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "N/A"
    if abs(n) >= 1e12:
        return f"${n/1e12:.2f}T"
    if abs(n) >= 1e9:
        return f"${n/1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"${n/1e6:.2f}M"
    return f"${n:,.0f}"

def fmt_pct(n, multiply=True):
    if n is None:
        return "N/A"
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "N/A"
    val = n * 100 if multiply else n
    return f"{val:.2f}%"

def fmt_num(n, decimals=2, prefix="", suffix=""):
    if n is None:
        return "N/A"
    try:
        return f"{prefix}{float(n):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"

def safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def calc_period_returns(hist):
    """Return % price change for standard lookback periods from a 1-year history DataFrame."""
    if hist is None or hist.empty:
        return {k: None for k in ["1d", "5d", "1m", "3m", "6m", "1y"]}
    prices = hist["Close"].dropna()
    if len(prices) < 2:
        return {k: None for k in ["1d", "5d", "1m", "3m", "6m", "1y"]}
    last = float(prices.iloc[-1])
    def _ret(n):
        if len(prices) <= n:
            return None
        prev = float(prices.iloc[-n - 1])
        return (last - prev) / prev * 100 if prev != 0 else None
    return {"1d": _ret(1), "5d": _ret(5), "1m": _ret(21), "3m": _ret(63), "6m": _ret(126), "1y": _ret(251)}


# ─── Derived metric calculations ──────────────────────────────────────────────

def calc_roic(financials, balance_sheet):
    try:
        op_income = None
        tax_rate = 0.21
        for key in ["Operating Income", "EBIT"]:
            if key in financials.index:
                op_income = financials.loc[key].iloc[0]
                break
        if op_income is None:
            return None
        if "Income Tax Expense" in financials.index and "Pretax Income" in financials.index:
            taxes = financials.loc["Income Tax Expense"].iloc[0]
            pretax = financials.loc["Pretax Income"].iloc[0]
            if pretax and pretax != 0:
                tax_rate = abs(float(taxes) / float(pretax))
        nopat = float(op_income) * (1 - tax_rate)
        equity = None
        for key in ["Stockholders Equity", "Total Stockholder Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"]:
            if key in balance_sheet.index:
                equity = balance_sheet.loc[key].iloc[0]
                break
        debt = None
        for key in ["Total Debt", "Long Term Debt", "Long Term Debt And Capital Lease Obligation"]:
            if key in balance_sheet.index:
                debt = balance_sheet.loc[key].iloc[0]
                break
        cash = None
        for key in ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash And Short Term Investments"]:
            if key in balance_sheet.index:
                cash = balance_sheet.loc[key].iloc[0]
                break
        invested_capital = float(equity or 0) + float(debt or 0) - float(cash or 0)
        if invested_capital == 0:
            return None
        return nopat / invested_capital
    except Exception:
        return None


def calc_interest_coverage(financials):
    try:
        ebit = None
        for key in ["Operating Income", "EBIT"]:
            if key in financials.index:
                ebit = float(financials.loc[key].iloc[0])
                break
        if ebit is None:
            return None
        interest = None
        for key in ["Interest Expense", "Interest Expense Non Operating", "Net Interest Income"]:
            if key in financials.index:
                val = financials.loc[key].iloc[0]
                if val is not None:
                    interest = abs(float(val))
                    break
        if not interest or interest == 0:
            return None
        return ebit / interest
    except Exception:
        return None


def calc_buyback_yield(info, cashflow):
    try:
        market_cap = safe_float(info.get("marketCap"))
        if not market_cap or cashflow.empty:
            return None
        for key in ["Repurchase Of Capital Stock", "Common Stock Repurchased", "Repurchase Common Stock"]:
            if key in cashflow.index:
                val = cashflow.loc[key].iloc[0]
                if val is not None:
                    return abs(float(val)) / market_cap
        return None
    except Exception:
        return None


def calc_earnings_price_changes(earnings_dates_dict, hist):
    empty = {"date": "N/A", "day": "N/A", "d10": "N/A", "d30": "N/A", "d60": "N/A",
             "_day": None, "_d10": None, "_d30": None, "_d60": None}
    try:
        if not earnings_dates_dict or hist is None or hist.empty:
            return empty
        now = datetime.now().date()
        past_dates = sorted(
            [datetime.fromisoformat(d).date() for d in earnings_dates_dict.keys()
             if datetime.fromisoformat(d).date() < now],
            reverse=True,
        )
        if not past_dates:
            return empty
        earnings_date = past_dates[0]
        hist_dates = hist.index.normalize().date

        def price_near(target):
            for delta in range(5):
                d = target + timedelta(days=delta)
                mask = hist_dates == d
                if mask.any():
                    return float(hist.loc[mask, "Close"].iloc[0])
            return None

        pre = price_near(earnings_date - timedelta(days=1))
        day0 = price_near(earnings_date)
        day10 = price_near(earnings_date + timedelta(days=10))
        day30 = price_near(earnings_date + timedelta(days=30))
        day60 = price_near(earnings_date + timedelta(days=60))

        def raw_chg(after):
            if pre and after:
                return (after - pre) / pre * 100
            return None

        def pct_str(after):
            v = raw_chg(after)
            return f"{v:+.2f}%" if v is not None else "N/A"

        return {
            "date": str(earnings_date),
            "day": pct_str(day0),  "d10": pct_str(day10),
            "d30": pct_str(day30), "d60": pct_str(day60),
            "_day": raw_chg(day0), "_d10": raw_chg(day10),
            "_d30": raw_chg(day30), "_d60": raw_chg(day60),
        }
    except Exception:
        return empty


def calc_technical_indicators(hist, info):
    """RSI, MACD, Bollinger Bands, 52W range position, beta, volume, volatility."""
    out = {k: None for k in ["rsi", "macd_sig", "macd_hist", "bb_pos",
                              "w52_high", "w52_low", "w52_pos",
                              "beta", "avg_vol", "rel_vol", "volatility"]}
    out["beta"]    = safe_float(info.get("beta"))
    out["w52_high"] = safe_float(info.get("fiftyTwoWeekHigh"))
    out["w52_low"]  = safe_float(info.get("fiftyTwoWeekLow"))
    out["avg_vol"]  = safe_float(info.get("averageVolume"))
    price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    curr_vol = safe_float(info.get("regularMarketVolume") or info.get("volume"))
    if out["avg_vol"] and curr_vol and out["avg_vol"] > 0:
        out["rel_vol"] = curr_vol / out["avg_vol"]
    h = out["w52_high"]; l = out["w52_low"]
    if h and l and price and (h - l) > 0:
        out["w52_pos"] = (price - l) / (h - l) * 100

    if hist is None or hist.empty or len(hist) < 30:
        return out
    closes = hist["Close"]

    try:
        rsi_s = _calc_rsi(closes)
        v = rsi_s.dropna()
        if not v.empty:
            out["rsi"] = float(v.iloc[-1])
    except Exception:
        pass

    try:
        ema12 = closes.ewm(span=12).mean()
        ema26 = closes.ewm(span=26).mean()
        macd  = ema12 - ema26
        sig   = macd.ewm(span=9).mean()
        h_val = float((macd - sig).iloc[-1])
        out["macd_sig"]  = "Bullish" if h_val > 0 else "Bearish"
        out["macd_hist"] = h_val
    except Exception:
        pass

    try:
        sma20 = closes.rolling(20).mean()
        std20 = closes.rolling(20).std()
        upper = (sma20 + 2 * std20).iloc[-1]
        lower = (sma20 - 2 * std20).iloc[-1]
        curr  = float(closes.iloc[-1])
        if upper != lower:
            out["bb_pos"] = (curr - lower) / (upper - lower) * 100
    except Exception:
        pass

    try:
        ret = closes.pct_change().dropna()
        if len(ret) >= 20:
            out["volatility"] = float(ret.tail(20).std() * (252 ** 0.5) * 100)
    except Exception:
        pass

    return out


def assess_moat(info):
    score = 0
    signals = []
    gross_margin = safe_float(info.get("grossMargins"))
    if gross_margin:
        if gross_margin > 0.50:
            score += 2
            signals.append(f"High gross margin {gross_margin*100:.0f}%")
        elif gross_margin > 0.30:
            score += 1
            signals.append(f"Moderate gross margin {gross_margin*100:.0f}%")
    roe = safe_float(info.get("returnOnEquity"))
    if roe and roe > 0.20:
        score += 2
        signals.append(f"Strong ROE {roe*100:.0f}%")
    elif roe and roe > 0.10:
        score += 1
    op_margin = safe_float(info.get("operatingMargins"))
    if op_margin and op_margin > 0.20:
        score += 1
        signals.append(f"Strong op margin {op_margin*100:.0f}%")
    revenue_growth = safe_float(info.get("revenueGrowth"))
    if revenue_growth and revenue_growth > 0.15:
        score += 1
        signals.append(f"High revenue growth {revenue_growth*100:.0f}%")
    if score >= 5:
        rating, color = "Strong Moat", "green"
    elif score >= 3:
        rating, color = "Moderate Moat", "orange"
    else:
        rating, color = "Narrow / No Moat", "red"
    detail = " | ".join(signals) if signals else "Insufficient data"
    return rating, color, detail, score


def assess_competitive_position(info):
    market_cap = safe_float(info.get("marketCap")) or 0
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    if market_cap >= 200e9:
        cap_desc = "Mega Cap – Likely sector leader"
    elif market_cap >= 10e9:
        cap_desc = "Large Cap – Major industry player"
    elif market_cap >= 2e9:
        cap_desc = "Mid Cap – Significant niche player"
    elif market_cap > 0:
        cap_desc = "Small / Micro Cap – Smaller player"
    else:
        cap_desc = "Unknown"
    return cap_desc, sector, industry


# ─── Data fetcher (cached 10 min) ────────────────────────────────────────────

def _is_rate_limited(e):
    import json as _j
    msg = str(e)
    return ("429" in msg or "Too Many Requests" in msg
            or (isinstance(e, (ValueError, _j.JSONDecodeError)) and "Expecting value" in msg))

def _try(fn, default=None, retries=2):
    if default is None:
        default = pd.DataFrame()
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            if _is_rate_limited(e):
                if attempt < retries:
                    time.sleep(4 * (attempt + 1))
                else:
                    raise RuntimeError(
                        "Yahoo Finance is rate-limiting this IP. Wait 1-2 minutes and try again."
                    ) from e
            elif attempt == retries:
                return default
    return default

@st.cache_data(ttl=600, show_spinner=False)
def fetch_data(symbol):
    ticker = yf.Ticker(symbol)
    try:
        info = _try(lambda: ticker.info, default=None)
    except RuntimeError as e:
        raise e
    except Exception:
        info = None
    if not info or "symbol" not in info:
        return None, None, None, None, None, {}
    financials    = _try(lambda: ticker.financials)
    balance_sheet = _try(lambda: ticker.balance_sheet)
    cashflow      = _try(lambda: ticker.cashflow)
    hist          = _try(lambda: ticker.history(period="1y"))
    try:
        ed = ticker.earnings_dates
        earnings_dates_dict = ({str(k): v for k, v in ed.to_dict("index").items()}
                               if ed is not None and not ed.empty else {})
    except Exception:
        earnings_dates_dict = {}
    return info, financials, balance_sheet, cashflow, hist, earnings_dates_dict


@st.cache_data(ttl=300, show_spinner=False)
def fetch_macro_snapshot():
    """1-day % change for macro indicators, batch downloaded."""
    syms = list(MACRO_TICKERS.values())
    try:
        raw = yf.download(syms, period="5d", auto_adjust=True, progress=False, threads=True)
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": syms[0]})
        out = {}
        for name, sym in MACRO_TICKERS.items():
            try:
                prices = close[sym].dropna()
                if len(prices) >= 2:
                    last, prev = float(prices.iloc[-1]), float(prices.iloc[-2])
                    out[name] = {"price": last, "change": (last - prev) / prev * 100, "symbol": sym}
                else:
                    out[name] = {"price": None, "change": None, "symbol": sym}
            except Exception:
                out[name] = {"price": None, "change": None, "symbol": sym}
        return out
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_sector_returns():
    """1-day % return for each sector ETF."""
    syms = list(SECTOR_ETFS.values())
    try:
        raw = yf.download(syms, period="5d", auto_adjust=True, progress=False, threads=True)
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": syms[0]})
        out = {}
        for name, sym in SECTOR_ETFS.items():
            try:
                prices = close[sym].dropna()
                out[name] = float((prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2] * 100) if len(prices) >= 2 else None
            except Exception:
                out[name] = None
        return out
    except Exception:
        return {}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_chart_history(symbol, period):
    """OHLCV history for a specific period (for the interactive chart)."""
    try:
        return yf.Ticker(symbol).history(period=period, auto_adjust=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_insider_data(symbol):
    """Recent insider transactions."""
    try:
        df = yf.Ticker(symbol).insider_transactions
        return df if df is not None and not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_earnings_surprise(symbol):
    """Past earnings: EPS estimate vs reported, with surprise %."""
    try:
        ed = yf.Ticker(symbol).earnings_dates
        if ed is None or ed.empty:
            return pd.DataFrame()
        cols = [c for c in ["EPS Estimate", "Reported EPS"] if c in ed.columns]
        if len(cols) < 2:
            return pd.DataFrame()
        ed = ed.dropna(subset=cols).copy()
        now = pd.Timestamp.now(tz="UTC")
        ed = ed[ed.index < now].sort_index(ascending=False).head(8)
        ed["Surprise"] = ed["Reported EPS"] - ed["EPS Estimate"]
        ed["Surprise %"] = (ed["Surprise"] / ed["EPS Estimate"].abs() * 100).round(1)
        ed.index = ed.index.strftime("%Y-%m-%d")
        return ed[["EPS Estimate", "Reported EPS", "Surprise", "Surprise %"]]
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_peer_metrics(symbols_tuple):
    """Key metrics for a tuple of peer symbols (cached per unique set)."""
    results = []
    for sym in symbols_tuple:
        try:
            info, *_ = fetch_data(sym)
            if info is None:
                continue
            price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
            results.append({
                "Symbol": sym,
                "Name": (info.get("longName") or sym)[:22],
                "Price": f"${price:.2f}" if price else "N/A",
                "Mkt Cap": fmt_large(safe_float(info.get("marketCap"))),
                "P/E": fmt_num(safe_float(info.get("trailingPE"))),
                "Fwd P/E": fmt_num(safe_float(info.get("forwardPE"))),
                "EPS": fmt_num(safe_float(info.get("trailingEps")), prefix="$"),
                "ROE": fmt_pct(safe_float(info.get("returnOnEquity"))),
                "Op. Margin": fmt_pct(safe_float(info.get("operatingMargins"))),
                "Rev. Growth": fmt_pct(safe_float(info.get("revenueGrowth"))),
            })
        except Exception:
            continue
    return results


# ─── Metrics extraction (for comparison) ─────────────────────────────────────

def extract_all_metrics(symbol, info, financials, balance_sheet, cashflow, hist, earnings_dates_dict):
    """Return a flat dict of raw numerics + formatted strings for one stock."""
    roic          = (calc_roic(financials, balance_sheet)
                     if financials is not None and not financials.empty
                        and balance_sheet is not None and not balance_sheet.empty else None)
    interest_cov  = (calc_interest_coverage(financials)
                     if financials is not None and not financials.empty else None)
    buyback_yld   = (calc_buyback_yield(info, cashflow)
                     if cashflow is not None and not cashflow.empty else None)
    earnings      = calc_earnings_price_changes(earnings_dates_dict, hist)
    moat_rating, moat_color, moat_detail, moat_score = assess_moat(info)
    cap_desc, sector, industry = assess_competitive_position(info)

    price         = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    ma50          = safe_float(info.get("fiftyDayAverage"))
    ma200         = safe_float(info.get("twoHundredDayAverage"))
    revenue       = safe_float(info.get("totalRevenue"))
    op_margin     = safe_float(info.get("operatingMargins"))
    eps_trail     = safe_float(info.get("trailingEps"))
    roe           = safe_float(info.get("returnOnEquity"))
    pe_trail      = safe_float(info.get("trailingPE"))
    pe_fwd        = safe_float(info.get("forwardPE"))
    peg           = safe_float(info.get("pegRatio"))
    total_debt    = safe_float(info.get("totalDebt"))
    fcf           = safe_float(info.get("freeCashflow"))
    curr_ratio    = safe_float(info.get("currentRatio"))
    quick_ratio   = safe_float(info.get("quickRatio"))
    div_yield     = safe_float(info.get("trailingAnnualDividendYield") or info.get("dividendYield"))
    div_rate      = safe_float(info.get("trailingAnnualDividendRate") or info.get("dividendRate"))
    payout_ratio  = safe_float(info.get("payoutRatio"))
    rec_score     = safe_float(info.get("recommendationMean"))
    target_mean   = safe_float(info.get("targetMeanPrice"))
    target_low    = safe_float(info.get("targetLowPrice"))
    target_high   = safe_float(info.get("targetHighPrice"))
    n_analysts    = info.get("numberOfAnalystOpinions")
    rec_key       = info.get("recommendationKey", "N/A")
    market_cap    = safe_float(info.get("marketCap"))

    vs50  = (price - ma50)  / ma50  * 100 if price and ma50  else None
    vs200 = (price - ma200) / ma200 * 100 if price and ma200 else None
    upside = (target_mean - price) / price * 100 if target_mean and price else None

    rec_labels = {"strongBuy": "Strong Buy", "buy": "Buy", "hold": "Hold",
                  "sell": "Sell", "strongSell": "Strong Sell"}
    rec_label = rec_labels.get(rec_key, str(rec_key).replace("_", " ").title())

    return {
        # identity
        "symbol": symbol,
        "name": info.get("longName", symbol),
        "sector": sector,
        "industry": industry,
        "market_cap_str": fmt_large(market_cap),
        "moat_rating": moat_rating,
        "moat_color": moat_color,
        "moat_detail": moat_detail,
        "cap_desc": cap_desc,
        "rec_label": rec_label,
        "earnings_date": earnings["date"],
        # ── raw numerics (used for comparison coloring) ──
        "_revenue": revenue,
        "_op_margin": op_margin,
        "_eps": eps_trail,
        "_roe": roe,
        "_roic": roic,
        "_pe_trail": pe_trail,
        "_pe_fwd": pe_fwd,
        "_peg": peg,
        "_vs50": vs50,
        "_vs200": vs200,
        "_debt": total_debt,
        "_fcf": fcf,
        "_curr_ratio": curr_ratio,
        "_quick_ratio": quick_ratio,
        "_int_cov": interest_cov,
        "_div_yield": div_yield,
        "_buyback": buyback_yld,
        "_rec_score": rec_score,
        "_upside": upside,
        "_earn_day": earnings["_day"],
        "_earn_d10": earnings["_d10"],
        "_earn_d30": earnings["_d30"],
        "_earn_d60": earnings["_d60"],
        "_moat_score": moat_score,
        # ── formatted strings (displayed in tables) ──
        "Revenue": fmt_large(revenue),
        "Op. Margin": fmt_pct(op_margin),
        "EPS (Trailing)": fmt_num(eps_trail, prefix="$"),
        "ROE": fmt_pct(roe),
        "ROIC": fmt_pct(roic),
        "P/E (Trailing)": fmt_num(pe_trail),
        "P/E (Forward)": fmt_num(pe_fwd),
        "PEG Ratio": fmt_num(peg),
        "vs 50-Day MA": fmt_num(vs50, suffix="%") if vs50 is not None else "N/A",
        "vs 200-Day MA": fmt_num(vs200, suffix="%") if vs200 is not None else "N/A",
        "Total Debt": fmt_large(total_debt),
        "Free Cash Flow": fmt_large(fcf),
        "Current Ratio": fmt_num(curr_ratio),
        "Quick Ratio": fmt_num(quick_ratio),
        "Interest Coverage": fmt_num(interest_cov, suffix="x"),
        "Dividend Yield": fmt_pct(div_yield) if div_yield else "—",
        "Buyback Yield": fmt_pct(buyback_yld) if buyback_yld else "—",
        "Analyst Score": fmt_num(rec_score),
        "Price Target Upside": fmt_num(upside, suffix="%") if upside is not None else "N/A",
        "Earn. Day Return": earnings["day"],
        "Return +10d": earnings["d10"],
        "Return +30d": earnings["d30"],
        "Return +60d": earnings["d60"],
    }


# ─── Comparison table rendering ───────────────────────────────────────────────

# Each entry: (display_label, raw_key, higher_is_better)
COMPARE_SECTIONS = {
    "Profitability": [
        ("Revenue",       "_revenue",  True),
        ("Op. Margin",    "_op_margin", True),
        ("EPS (Trailing)","_eps",       True),
        ("ROE",           "_roe",       True),
        ("ROIC",          "_roic",      True),
    ],
    "Valuation": [
        ("P/E (Trailing)", "_pe_trail", False),
        ("P/E (Forward)",  "_pe_fwd",   False),
        ("PEG Ratio",      "_peg",      False),
        ("vs 50-Day MA",   "_vs50",     True),
        ("vs 200-Day MA",  "_vs200",    True),
    ],
    "Financial Health": [
        ("Total Debt",       "_debt",      False),
        ("Free Cash Flow",   "_fcf",       True),
        ("Current Ratio",    "_curr_ratio", True),
        ("Quick Ratio",      "_quick_ratio", True),
        ("Interest Coverage","_int_cov",   True),
    ],
    "Shareholder Returns": [
        ("Dividend Yield", "_div_yield", True),
        ("Buyback Yield",  "_buyback",   True),
    ],
    "Earnings Impact": [
        ("Earnings Day",  "_earn_day", True),
        ("Return +10d",   "_earn_d10", True),
        ("Return +30d",   "_earn_d30", True),
        ("Return +60d",   "_earn_d60", True),
    ],
    "Analyst Outlook": [
        ("Analyst Score",        "_rec_score", False),  # 1=strong buy, lower=better
        ("Price Target Upside",  "_upside",    True),
    ],
    "Competitive Position": [
        ("Moat Score", "_moat_score", True),
    ],
}

GREEN_BG = "background-color: #0d3b1e; color: #6bff9e"
RED_BG   = "background-color: #3b0d0d; color: #ff6b6b"
MID_BG   = ""

# ─── Stock Screener universe ──────────────────────────────────────────────────

AI_UNIVERSE = list(dict.fromkeys([
    # AI Chips & Hardware
    "NVDA","AMD","AVGO","INTC","QCOM","AMAT","MU","LRCX","KLAC","MRVL","TXN",
    # Big Tech (AI-core)
    "MSFT","GOOGL","META","AMZN","AAPL",
    # AI Cloud & Software
    "ORCL","CRM","ADBE","NOW","SNPS","CDNS","SNOW","PLTR","DDOG",
    # Cybersecurity (AI-driven)
    "PANW","CRWD","ZS","FTNT",
    # AI-adjacent
    "TSLA","TTD","SHOP","COIN","SQ",
]))

NON_AI_UNIVERSE = list(dict.fromkeys([
    # Financials
    "JPM","BAC","WFC","GS","MS","BLK","V","MA","AXP","C","COF","SPGI","MCO",
    # Healthcare
    "JNJ","UNH","PFE","ABBV","MRK","ABT","TMO","DHR","CVS","CI","ISRG","GILD","LLY",
    # Consumer Discretionary
    "HD","WMT","COST","TGT","MCD","SBUX","NKE","TJX","LOW","BKNG","DG",
    # Energy
    "XOM","CVX","COP","EOG","SLB","PSX","MPC","VLO","OXY",
    # Industrials
    "CAT","DE","HON","RTX","LMT","GE","BA","UPS","FDX","EMR","PH","ITW",
    # Communication / Media
    "DIS","NFLX","CMCSA","T","VZ","TMUS",
    # Utilities / REITs
    "NEE","DUK","SO","AMT","PLD","EQIX",
    # Consumer Staples
    "PG","KO","PEP","PM","MO","CL","GIS",
    # Materials
    "LIN","APD","NEM","FCX",
    # Other
    "UBER","ABNB","RBLX","BRK-B","BX","KKR","MSCI",
]))

STOCK_UNIVERSE = list(dict.fromkeys(AI_UNIVERSE + NON_AI_UNIVERSE))

MACRO_TICKERS = {
    "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI",
    "VIX": "^VIX", "10Y Yield": "^TNX", "Gold": "GC=F",
    "Oil (WTI)": "CL=F", "US Dollar": "DX-Y.NYB", "USD/INR": "INR=X",
}

SECTOR_ETFS = {
    "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF",
    "Energy": "XLE", "Industrials": "XLI", "Cons. Disc.": "XLY",
    "Cons. Staples": "XLP", "Utilities": "XLU", "Materials": "XLB",
    "Real Estate": "XLRE", "Comm. Svcs": "XLC",
}

# sector name → list of STOCK_UNIVERSE tickers in that sector (populated lazily)
_SECTOR_PEER_MAP: dict = {}


def build_comparison_df(metrics_list, section_rows):
    """
    Build a styled DataFrame for one section.
    metrics_list: list of extract_all_metrics dicts
    section_rows: list of (label, raw_key, higher_better)
    Returns (display_df, style_df)
    """
    symbols = [m["symbol"] for m in metrics_list]
    rows = []
    styles = []

    for label, raw_key, higher_better in section_rows:
        display_row = {s: m.get(label, "N/A") for s, m in zip(symbols, metrics_list)}
        raw_vals    = [m.get(raw_key) for m in metrics_list]

        valid_pairs = [(i, v) for i, v in enumerate(raw_vals) if v is not None]
        style_row = {s: MID_BG for s in symbols}

        if len(valid_pairs) >= 2:
            vals = [v for _, v in valid_pairs]
            best  = max(vals) if higher_better else min(vals)
            worst = min(vals) if higher_better else max(vals)
            for i, v in valid_pairs:
                s = symbols[i]
                if v == best:
                    style_row[s] = GREEN_BG
                elif v == worst:
                    style_row[s] = RED_BG

        rows.append({"Metric": label, **display_row})
        styles.append(style_row)

    df = pd.DataFrame(rows).set_index("Metric")

    def apply_styles(df_in):
        out = pd.DataFrame("", index=df_in.index, columns=df_in.columns)
        for i, row_style in enumerate(styles):
            for s in symbols:
                out.iloc[i][s] = row_style.get(s, "")
        return out

    styled = df.style.apply(apply_styles, axis=None)
    return styled


# ─── LLM summary builder ─────────────────────────────────────────────────────

def build_llm_prompt(metrics_list):
    lines = [
        "You are a seasoned financial analyst. Below are key metrics for three stocks pulled from Yahoo Finance.",
        "Write a structured comparison covering:",
        "1. Overall winner and why (2-3 sentences)",
        "2. Key strengths and risks for each stock (3-4 bullet points each)",
        "3. A concise investment outlook table (one line per stock: Bullish / Neutral / Cautious + one-line reason)",
        "4. Any red flags or standout metrics worth highlighting",
        "",
        "Be direct, data-driven, and concrete. Do not add disclaimers beyond one sentence at the end.",
        "",
    ]
    for m in metrics_list:
        lines += [
            f"━━━ {m['symbol']} — {m['name']} ━━━",
            f"Sector: {m['sector']} | Industry: {m['industry']}",
            f"Market Cap: {m['market_cap_str']}",
            "",
            "PROFITABILITY",
            f"  Revenue:          {m['Revenue']}",
            f"  Operating Margin: {m['Op. Margin']}",
            f"  EPS (trailing):   {m['EPS (Trailing)']}",
            f"  ROE:              {m['ROE']}",
            f"  ROIC:             {m['ROIC']}",
            "",
            "VALUATION",
            f"  P/E (trailing):   {m['P/E (Trailing)']}",
            f"  P/E (forward):    {m['P/E (Forward)']}",
            f"  PEG ratio:        {m['PEG Ratio']}",
            f"  vs 50-day MA:     {m['vs 50-Day MA']}",
            f"  vs 200-day MA:    {m['vs 200-Day MA']}",
            "",
            "FINANCIAL HEALTH",
            f"  Total Debt:       {m['Total Debt']}",
            f"  Free Cash Flow:   {m['Free Cash Flow']}",
            f"  Current Ratio:    {m['Current Ratio']}",
            f"  Quick Ratio:      {m['Quick Ratio']}",
            f"  Interest Coverage:{m['Interest Coverage']}",
            "",
            "SHAREHOLDER RETURNS",
            f"  Dividend Yield:   {m['Dividend Yield']}",
            f"  Buyback Yield:    {m['Buyback Yield']}",
            "",
            f"EARNINGS IMPACT (last: {m['earnings_date']})",
            f"  Earnings day:     {m['Earn. Day Return']}",
            f"  +10 days:         {m['Return +10d']}",
            f"  +30 days:         {m['Return +30d']}",
            f"  +60 days:         {m['Return +60d']}",
            "",
            "ANALYST OUTLOOK",
            f"  Recommendation:   {m['rec_label']}",
            f"  Consensus score:  {m['Analyst Score']} (1=Strong Buy → 5=Strong Sell)",
            f"  Price target up:  {m['Price Target Upside']}",
            "",
            f"COMPETITIVE POSITION",
            f"  Moat:             {m['moat_rating']} — {m['moat_detail']}",
            f"  Market position:  {m['cap_desc']}",
            "",
        ]
    return "\n".join(lines)


def stream_llm_analysis(prompt, api_key):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=1800,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text


# ─── Single-stock view ────────────────────────────────────────────────────────

def render_single_stock(info, financials, balance_sheet, cashflow, hist, earnings_dates_dict, symbol_input):
    roic         = (calc_roic(financials, balance_sheet)
                    if financials is not None and not financials.empty
                       and balance_sheet is not None and not balance_sheet.empty else None)
    interest_cov = (calc_interest_coverage(financials)
                    if financials is not None and not financials.empty else None)
    buyback_yld  = (calc_buyback_yield(info, cashflow)
                    if cashflow is not None and not cashflow.empty else None)
    earnings     = calc_earnings_price_changes(earnings_dates_dict, hist)
    moat_rating, moat_color, moat_detail, _ = assess_moat(info)
    cap_desc, sector, industry = assess_competitive_position(info)

    price  = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    ma50   = safe_float(info.get("fiftyDayAverage"))
    ma200  = safe_float(info.get("twoHundredDayAverage"))
    market_cap = safe_float(info.get("marketCap"))
    currency   = info.get("currency", "USD")
    exchange   = info.get("exchange", "")

    st.subheader(f"{info.get('longName', symbol_input)}  ({info.get('symbol', symbol_input)})")
    st.caption(f"{exchange} · {sector} · {industry} · {currency} · Market Cap: {fmt_large(market_cap)}")
    st.divider()

    # ── Interactive price chart ───────────────────────────────────────────────
    with st.expander("📊 Price Chart", expanded=True):
        ct1, ct2, _ = st.columns([2, 2, 4])
        chart_period = ct1.selectbox("Period", ["1mo","3mo","6mo","1y","2y","5y"],
                                     index=3, key=f"cp_{symbol_input}")
        chart_type   = ct2.selectbox("Type", ["Line","Candlestick"],
                                     index=0, key=f"ct_{symbol_input}")
        ch = fetch_chart_history(symbol_input, chart_period)
        if ch is not None and not ch.empty:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.03, row_heights=[0.78, 0.22])
            if chart_type == "Candlestick":
                fig.add_trace(go.Candlestick(
                    x=ch.index, open=ch["Open"], high=ch["High"],
                    low=ch["Low"], close=ch["Close"], name="OHLC",
                    increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                ), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=ch.index, y=ch["Close"], name="Close",
                                         line=dict(color="#1f77b4", width=2)), row=1, col=1)
            if ma50:
                fig.add_hline(y=ma50, line_dash="dot", line_color="orange",
                              annotation_text=f"50d ${ma50:.2f}", row=1, col=1)
            if ma200:
                fig.add_hline(y=ma200, line_dash="dot", line_color="#ef5350",
                              annotation_text=f"200d ${ma200:.2f}", row=1, col=1)
            if "Volume" in ch.columns:
                colors_vol = ["#26a69a" if ch["Close"].iloc[i] >= ch["Open"].iloc[i]
                              else "#ef5350" for i in range(len(ch))]
                fig.add_trace(go.Bar(x=ch.index, y=ch["Volume"], name="Volume",
                                     marker_color=colors_vol, showlegend=False), row=2, col=1)
            fig.update_layout(height=440, margin=dict(l=0, r=0, t=10, b=0),
                              xaxis_rangeslider_visible=False,
                              legend=dict(orientation="h"),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Vol", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

    # 0 · Price Performance
    perf = calc_period_returns(hist)
    st.subheader("0 · Price Performance")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    def _pct(v):
        return f"{v:+.2f}%" if v is not None else "N/A"
    c1.metric("1 Day",    _pct(perf["1d"]))
    c2.metric("5 Days",   _pct(perf["5d"]))
    c3.metric("1 Month",  _pct(perf["1m"]))
    c4.metric("3 Months", _pct(perf["3m"]))
    c5.metric("6 Months", _pct(perf["6m"]))
    c6.metric("1 Year",   _pct(perf["1y"]))
    st.divider()

    # 1 · Profitability
    st.subheader("1 · Profitability")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Revenue",         fmt_large(safe_float(info.get("totalRevenue"))))
    c2.metric("Operating Margin",fmt_pct(safe_float(info.get("operatingMargins"))),
              help="Operating income / Revenue")
    c3.metric("EPS (Trailing)",  fmt_num(safe_float(info.get("trailingEps")), prefix="$"))
    c4.metric("ROE",             fmt_pct(safe_float(info.get("returnOnEquity"))),
              help="Net Income / Shareholders' Equity")
    c5.metric("ROIC",            fmt_pct(roic), help="NOPAT / (Equity + Debt − Cash)")
    st.divider()

    # 2 · Valuation
    st.subheader("2 · Valuation")
    c1, c2, c3, c4, c5 = st.columns(5)
    pe_trail = safe_float(info.get("trailingPE"))
    pe_fwd   = safe_float(info.get("forwardPE"))
    peg      = safe_float(info.get("pegRatio"))
    vs50  = f"{(price - ma50)  / ma50  * 100:+.2f}%" if price and ma50  else "N/A"
    vs200 = f"{(price - ma200) / ma200 * 100:+.2f}%" if price and ma200 else "N/A"
    c1.metric("P/E (Trailing)",  fmt_num(pe_trail))
    c2.metric("P/E (Forward)",   fmt_num(pe_fwd))
    c3.metric("PEG Ratio",       fmt_num(peg), help="P/E / EPS growth. <1 = undervalued")
    c4.metric("50-Day MA",       fmt_num(ma50, prefix="$"), delta=vs50)
    c5.metric("200-Day MA",      fmt_num(ma200, prefix="$"), delta=vs200)
    st.divider()

    # 3 · Financial Health
    st.subheader("3 · Financial Health")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Debt",       fmt_large(safe_float(info.get("totalDebt"))))
    c2.metric("Free Cash Flow",   fmt_large(safe_float(info.get("freeCashflow"))))
    c3.metric("Current Ratio",    fmt_num(safe_float(info.get("currentRatio"))),
              help="Current assets / Current liabilities")
    c4.metric("Quick Ratio",      fmt_num(safe_float(info.get("quickRatio"))),
              help="(Current assets − Inventory) / Current liabilities")
    c5.metric("Interest Coverage",fmt_num(interest_cov, suffix="x"),
              help="EBIT / Interest Expense")
    st.divider()

    # 4 · Shareholder Returns
    st.subheader("4 · Shareholder Returns")
    c1, c2, c3, c4 = st.columns(4)
    div_yield = safe_float(info.get("trailingAnnualDividendYield") or info.get("dividendYield"))
    div_rate  = safe_float(info.get("trailingAnnualDividendRate") or info.get("dividendRate"))
    payout    = safe_float(info.get("payoutRatio"))
    c1.metric("Dividend Yield",  fmt_pct(div_yield) if div_yield else "No Dividend")
    c2.metric("Annual Dividend", fmt_num(div_rate, prefix="$") if div_rate else "N/A")
    c3.metric("Payout Ratio",    fmt_pct(payout) if payout else "N/A")
    c4.metric("Buyback Yield",   fmt_pct(buyback_yld) if buyback_yld else "N/A",
              help="Share repurchases / Market cap")
    st.divider()

    # 5 · Earnings Price Impact
    st.subheader("5 · Earnings Price Impact")
    st.caption(f"Most recent earnings: **{earnings['date']}**  ·  % vs. day-before close")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Earnings Day", earnings["day"])
    c2.metric("After 10 Days", earnings["d10"])
    c3.metric("After 30 Days", earnings["d30"])
    c4.metric("After 60 Days", earnings["d60"])
    st.divider()

    # 6 · Analyst Outlook
    st.subheader("6 · Analyst Outlook")
    rec_key   = info.get("recommendationKey", "N/A")
    rec_score = safe_float(info.get("recommendationMean"))
    n_analysts= info.get("numberOfAnalystOpinions", "N/A")
    t_mean    = safe_float(info.get("targetMeanPrice"))
    t_low     = safe_float(info.get("targetLowPrice"))
    t_high    = safe_float(info.get("targetHighPrice"))
    rec_labels = {"strongBuy": "Strong Buy", "buy": "Buy", "hold": "Hold",
                  "sell": "Sell", "strongSell": "Strong Sell"}
    rec_label = rec_labels.get(rec_key, str(rec_key).replace("_", " ").title())
    upside = f"{(t_mean - price) / price * 100:+.2f}%" if t_mean and price else "N/A"
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Recommendation",    rec_label)
    c2.metric("Consensus Score",   fmt_num(rec_score), help="1=Strong Buy → 5=Strong Sell")
    c3.metric("# Analysts",        str(n_analysts))
    c4.metric("Avg Price Target",  fmt_num(t_mean, prefix="$"), delta=upside)
    c5.metric("Target Range",      f"${fmt_num(t_low)} – ${fmt_num(t_high)}"
                                   if t_low and t_high else "N/A")
    st.divider()

    # 7 · Competitive Position
    st.subheader("7 · Competitive Position")
    col_moat, col_comp = st.columns(2)
    moat_icon = {"green": "🟢", "orange": "🟡", "red": "🔴"}.get(moat_color, "")
    with col_moat:
        st.markdown("**Economic Moat**")
        st.markdown(f"{moat_icon} **{moat_rating}**")
        st.caption(f"Signals: {moat_detail}")
        st.caption("_Proxy: gross margin, ROE, operating margin, revenue growth._")
    with col_comp:
        st.markdown("**Market Position**")
        st.markdown(f"**{cap_desc}**")
        st.caption(f"Sector: {sector}  |  Industry: {industry}")
        employees = info.get("fullTimeEmployees")
        if employees:
            st.caption(f"Employees: {employees:,}")
    st.divider()

    # 8 · Earnings Countdown + Next Earnings
    next_earnings_date = None
    try:
        now_ts = pd.Timestamp.now(tz="UTC")
        future_dates = [
            pd.Timestamp(d) if not isinstance(d, pd.Timestamp) else d
            for d in earnings_dates_dict.keys()
            if pd.Timestamp(d).tzinfo is None and pd.Timestamp(d) > now_ts.replace(tzinfo=None)
            or (hasattr(pd.Timestamp(d), "tzinfo") and pd.Timestamp(d).tzinfo is not None and pd.Timestamp(d) > now_ts)
        ]
        if future_dates:
            next_earnings_date = min(future_dates)
    except Exception:
        pass

    if next_earnings_date is not None:
        try:
            days_to_earnings = (next_earnings_date.replace(tzinfo=None) - pd.Timestamp.now().replace(tzinfo=None)).days
        except Exception:
            days_to_earnings = None
        st.subheader("8 · Next Earnings")
        ec1, ec2 = st.columns(2)
        ec1.metric("Earnings Date", next_earnings_date.strftime("%Y-%m-%d"))
        if days_to_earnings is not None:
            urgency = "🔴" if days_to_earnings <= 7 else "🟡" if days_to_earnings <= 30 else "🟢"
            ec2.metric("Days Away", f"{urgency} {days_to_earnings}d")
        st.divider()

    # 9 · Short Interest
    short_pct   = info.get("shortPercentOfFloat")
    short_ratio = info.get("shortRatio")
    shares_short = info.get("sharesShort")
    if any(v is not None for v in [short_pct, short_ratio, shares_short]):
        st.subheader("9 · Short Interest")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Short % of Float", f"{short_pct * 100:.1f}%" if short_pct else "N/A")
        sc2.metric("Short Ratio (days)", f"{short_ratio:.1f}" if short_ratio else "N/A")
        sc3.metric("Shares Short", fmt_large(safe_float(shares_short)) if shares_short else "N/A")
        if short_pct and short_pct > 0.10:
            st.caption(f"⚠️ High short interest ({short_pct*100:.1f}% of float) — elevated bearish sentiment or squeeze candidate.")
        st.divider()

    # 10 · Insider Transactions
    with st.expander("👥 Insider Transactions", expanded=False):
        insider_df = fetch_insider_data(symbol_input)
        if insider_df.empty:
            st.caption("No insider transaction data available.")
        else:
            show_cols = [c for c in ["Date", "Insider", "Position", "Transaction", "Shares", "Value"] if c in insider_df.columns]
            if not show_cols:
                show_cols = list(insider_df.columns[:6])
            st.dataframe(insider_df[show_cols].head(15), use_container_width=True)

    # 11 · Earnings Surprise History
    with st.expander("📈 Earnings Surprise History", expanded=False):
        surp_df = fetch_earnings_surprise(symbol_input)
        if surp_df.empty:
            st.caption("No earnings surprise data available.")
        else:
            def _color_surprise(val):
                try:
                    v = float(val)
                    return "color: #22c55e" if v > 0 else "color: #ef4444"
                except Exception:
                    return ""
            st.dataframe(
                surp_df.style.applymap(_color_surprise, subset=["Surprise %"]),
                use_container_width=True,
            )

    # 12 · Peer Comparison
    with st.expander("🏁 Peer Comparison", expanded=False):
        current_sector = info.get("sector", "")
        if current_sector and not _SECTOR_PEER_MAP.get(current_sector):
            _SECTOR_PEER_MAP[current_sector] = [
                s for s in STOCK_UNIVERSE if s != symbol_input
            ]
        sector_pool = _SECTOR_PEER_MAP.get(current_sector, [])
        # pick up to 8 peers: first the ones in the same industry, then fill with sector
        current_industry = info.get("industry", "")
        industry_peers: list[str] = []
        other_peers: list[str] = []
        for s in sector_pool:
            try:
                s_info_peek, *_ = fetch_data(s)
                if s_info_peek is None:
                    continue
                if s_info_peek.get("industry") == current_industry:
                    industry_peers.append(s)
                else:
                    other_peers.append(s)
            except Exception:
                continue
            if len(industry_peers) >= 5:
                break
        peer_symbols = (industry_peers + other_peers)[:8]
        all_peer_symbols = tuple([symbol_input] + peer_symbols)
        if peer_symbols:
            with st.spinner("Fetching peer data…"):
                peer_rows = fetch_peer_metrics(all_peer_symbols)
            if peer_rows:
                peer_df = pd.DataFrame(peer_rows)
                def _highlight_self(row):
                    return ["font-weight: bold; background-color: rgba(99,102,241,0.15)" if row["Symbol"] == symbol_input else "" for _ in row]
                st.dataframe(
                    peer_df.style.apply(_highlight_self, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("Could not fetch peer data.")
        else:
            st.caption("No peers found in the same sector within the universe.")

    st.divider()

    with st.expander("📄 Raw Info (all fields)"):
        st.json({k: v for k, v in info.items() if v is not None and v != ""})
    if financials is not None and not financials.empty:
        with st.expander("📄 Annual Income Statement"):
            st.dataframe(financials.map(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x))
    if balance_sheet is not None and not balance_sheet.empty:
        with st.expander("📄 Annual Balance Sheet"):
            st.dataframe(balance_sheet.map(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x))
    if cashflow is not None and not cashflow.empty:
        with st.expander("📄 Annual Cash Flow Statement"):
            st.dataframe(cashflow.map(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x))


# ─── Compare view ─────────────────────────────────────────────────────────────

def render_compare(api_key):
    # ── Stock selection ───────────────────────────────────────────────────────
    if "compare_tickers" not in st.session_state:
        st.session_state["compare_tickers"] = ["AAPL", "MSFT", "GOOGL"]

    st.markdown("Add up to **10 tickers**. Check ✓ to include in the comparison.")

    to_remove = None
    selected = []
    h1, h2, h3 = st.columns([4, 1, 1])
    h1.caption("Ticker"); h2.caption("Include"); h3.caption("")

    for i in range(len(st.session_state["compare_tickers"])):
        c1, c2, c3 = st.columns([4, 1, 1])
        sym = c1.text_input(
            f"t{i}", value=st.session_state["compare_tickers"][i],
            placeholder=f"e.g. NVDA", label_visibility="collapsed", key=f"cmp_t_{i}",
        ).strip().upper()
        included = c2.checkbox("✓", value=True, key=f"cmp_s_{i}", label_visibility="collapsed")
        if c3.button("✕", key=f"cmp_r_{i}", help="Remove"):
            to_remove = i
        st.session_state["compare_tickers"][i] = sym
        if included and sym:
            selected.append(sym)

    if to_remove is not None:
        st.session_state["compare_tickers"].pop(to_remove)
        st.rerun()

    ca, cb, _ = st.columns([1, 1, 4])
    if ca.button("＋ Add Ticker", use_container_width=True, key="cmp_add"):
        if len(st.session_state["compare_tickers"]) < 10:
            st.session_state["compare_tickers"].append("")
            st.rerun()
    compare_btn = cb.button("Compare ▶", type="primary", use_container_width=True, key="compare_btn")

    if not compare_btn:
        st.info("Fill in tickers above and click **Compare ▶**.")
        return

    tickers = list(dict.fromkeys(t for t in selected if t))  # dedupe, preserve order
    if len(tickers) < 2:
        st.warning("Select at least 2 tickers.")
        return

    # ── Fetch data ────────────────────────────────────────────────────────────
    all_metrics = []
    all_hists   = {}
    all_tech    = {}
    errors      = []

    progress = st.progress(0, text="Fetching data…")
    for i, sym in enumerate(tickers):
        progress.progress(i / len(tickers), text=f"Fetching {sym}…")
        try:
            info, fin, bs, cf, hist, ed = fetch_data(sym)
            if info is None:
                errors.append(f"{sym}: could not fetch data (check ticker)")
                continue
            all_metrics.append(extract_all_metrics(sym, info, fin, bs, cf, hist, ed))
            all_hists[sym] = hist
            all_tech[sym]  = calc_technical_indicators(hist, info)
        except RuntimeError as e:
            errors.append(f"{sym}: {e}")
        except Exception as e:
            errors.append(f"{sym}: unexpected error — {e}")

    progress.progress(1.0, text="Done")
    progress.empty()

    if errors:
        for err in errors:
            st.warning(err)
    if len(all_metrics) < 2:
        st.error("Need at least 2 valid stocks to compare.")
        return

    symbols = [m["symbol"] for m in all_metrics]
    names   = {m["symbol"]: m["name"] for m in all_metrics}
    colors  = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
               "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]

    # ── Header cards ──────────────────────────────────────────────────────────
    st.divider()
    cols = st.columns(len(all_metrics))
    for col, m in zip(cols, all_metrics):
        col.markdown(f"### {m['symbol']}")
        col.caption(m["name"])
        col.caption(f"{m['sector']} · {m['industry']}")
        col.metric("Market Cap", m["market_cap_str"])

    # ── Normalized price chart ────────────────────────────────────────────────
    st.divider()
    with st.expander("📊 Normalized 1-Year Price Performance (base = 100)", expanded=True):
        fig = go.Figure()
        for i, sym in enumerate(symbols):
            h = all_hists.get(sym)
            if h is not None and not h.empty:
                norm = h["Close"] / h["Close"].iloc[0] * 100
                fig.add_trace(go.Scatter(x=h.index, y=norm,
                    name=f"{sym} — {names[sym]}",
                    line=dict(color=colors[i % len(colors)], width=2)))
        fig.add_hline(y=100, line_dash="dot", line_color="gray", line_width=1)
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title=None, yaxis_title="Indexed (start=100)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    # ── Volume chart ──────────────────────────────────────────────────────────
    with st.expander("📊 Volume (1 Year)", expanded=False):
        fig2 = go.Figure()
        for i, sym in enumerate(symbols):
            h = all_hists.get(sym)
            if h is not None and not h.empty and "Volume" in h.columns:
                fig2.add_trace(go.Scatter(x=h.index, y=h["Volume"], name=sym,
                    line=dict(color=colors[i % len(colors)])))
        fig2.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title=None, yaxis_title="Volume",
            legend=dict(orientation="h"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Fundamental comparison ────────────────────────────────────────────────
    st.divider()
    st.markdown("### Fundamental Comparison")
    st.caption("🟢 Best value in row  ·  🔴 Worst value  ·  Gray = middle or N/A")
    for section_name, sec_rows in COMPARE_SECTIONS.items():
        st.markdown(f"**{section_name}**")
        st.dataframe(build_comparison_df(all_metrics, sec_rows), use_container_width=True)
        st.markdown("")

    moat_icon = {"green": "🟢", "orange": "🟡", "red": "🔴"}
    st.markdown("**Qualitative Summary**")
    qual_rows = [{"Stock": m["symbol"],
                  "Moat": f"{moat_icon.get(m['moat_color'],'')} {m['moat_rating']}",
                  "Market Position": m["cap_desc"],
                  "Analyst Rec.": m["rec_label"],
                  "Earnings Date": m["earnings_date"]} for m in all_metrics]
    st.dataframe(pd.DataFrame(qual_rows).set_index("Stock"), use_container_width=True)

    # ── Technical Analysis ────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Technical Analysis")

    def _rsi_label(v):
        if v is None: return "N/A"
        tag = " ⚠OB" if v > 70 else (" ⚠OS" if v < 30 else "")
        return f"{v:.0f}{tag}"

    def _bb_label(v):
        if v is None: return "N/A"
        note = " (near upper)" if v >= 85 else (" (near lower)" if v <= 15 else "")
        return f"{v:.0f}%{note}"

    tech_spec = [
        ("RSI (14)",           "rsi",       _rsi_label),
        ("MACD Signal",        "macd_sig",  lambda v: v or "N/A"),
        ("Bollinger Position", "bb_pos",    _bb_label),
        ("52W High",           "w52_high",  lambda v: f"${v:.2f}" if v else "N/A"),
        ("52W Low",            "w52_low",   lambda v: f"${v:.2f}" if v else "N/A"),
        ("52W Range %",        "w52_pos",   lambda v: f"{v:.0f}% of range" if v is not None else "N/A"),
        ("Beta",               "beta",      lambda v: f"{v:.2f}" if v else "N/A"),
        ("Avg Volume (50d)",   "avg_vol",   lambda v: f"{v/1e6:.1f}M" if v else "N/A"),
        ("Relative Volume",    "rel_vol",   lambda v: f"{v:.2f}x" if v else "N/A"),
        ("Ann. Volatility",    "volatility",lambda v: f"{v:.1f}%" if v else "N/A"),
    ]

    tech_rows = []
    for label, key, fmt in tech_spec:
        row = {"Metric": label}
        for sym in symbols:
            row[sym] = fmt(all_tech.get(sym, {}).get(key))
        tech_rows.append(row)
    st.dataframe(pd.DataFrame(tech_rows).set_index("Metric"), use_container_width=True)

    # ── AI Analysis ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### AI Analysis")
    if not api_key:
        st.info("Enter your **Anthropic API key** in the sidebar to unlock AI analysis.")
        return
    if st.button("Generate AI Summary", type="primary", key="ai_btn"):
        prompt = build_llm_prompt(all_metrics)
        st.markdown("---")
        with st.spinner("Claude is analyzing the stocks…"):
            try:
                response_container = st.empty()
                full_text = ""
                for chunk in stream_llm_analysis(prompt, api_key):
                    full_text += chunk
                    response_container.markdown(full_text + "▌")
                response_container.markdown(full_text)
            except Exception as e:
                st.error(f"AI analysis failed: {e}")


# ─── Screener scoring helpers ────────────────────────────────────────────────

def _ts(val, breakpoints, scores, na=10):
    """Tier-score: breakpoints descending; returns matching score or 0 if below all."""
    if val is None:
        return na
    for bp, sc in zip(breakpoints, scores):
        if val >= bp:
            return sc
    return 0


def score_growth_topline(info):
    """0-100: future topline growth potential."""
    s = 0
    # 1. Revenue growth YoY
    s += _ts(safe_float(info.get("revenueGrowth")),
             [0.30, 0.20, 0.10, 0.05, 0.0], [25, 20, 15, 10, 5], na=10)
    # 2. Earnings growth
    s += _ts(safe_float(info.get("earningsGrowth")),
             [0.30, 0.20, 0.10, 0.0], [25, 20, 15, 8], na=10)
    # 3. Forward P/E < Trailing P/E → earnings expected to grow
    pe_t = safe_float(info.get("trailingPE"))
    pe_f = safe_float(info.get("forwardPE"))
    if pe_t and pe_f and pe_t > 0 and pe_f > 0:
        ratio = pe_f / pe_t
        if ratio < 0.80:   s += 25
        elif ratio < 0.90: s += 20
        elif ratio < 1.00: s += 15
        elif ratio < 1.10: s += 8
        else:              s += 3
    else:
        s += 10
    # 4. Analyst price-target upside
    price  = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    target = safe_float(info.get("targetMeanPrice"))
    if price and target and price > 0:
        upside = (target - price) / price
        s += _ts(upside, [0.30, 0.20, 0.10, 0.05, 0.0], [25, 20, 15, 10, 5], na=10)
    else:
        s += 10
    return min(100, s)


def score_bottom_line(info):
    """0-100: current profitability."""
    s = 0
    s += _ts(safe_float(info.get("profitMargins")),
             [0.25, 0.15, 0.08, 0.03, 0.0], [25, 20, 15, 10, 5], na=8)
    s += _ts(safe_float(info.get("returnOnEquity")),
             [0.25, 0.15, 0.10, 0.05, 0.0], [25, 20, 15, 10, 5], na=8)
    s += _ts(safe_float(info.get("operatingMargins")),
             [0.25, 0.15, 0.08, 0.03, 0.0], [25, 20, 15, 10, 5], na=8)
    s += _ts(safe_float(info.get("returnOnAssets")),
             [0.12, 0.08, 0.05, 0.02, 0.0], [25, 20, 15, 10, 5], na=8)
    return min(100, s)


def score_mgmt_debt_risk(info):
    """0-100: management quality, low debt, financial stability."""
    s = 0
    # Debt-to-equity (lower = better)
    d2e = safe_float(info.get("debtToEquity"))
    if d2e is not None:
        if d2e < 50:   s += 25
        elif d2e < 100: s += 20
        elif d2e < 150: s += 15
        elif d2e < 250: s += 8
        else:           s += 3
    else:
        s += 15  # many high-quality companies carry zero debt
    s += _ts(safe_float(info.get("currentRatio")),
             [2.5, 1.5, 1.0, 0.5], [25, 20, 15, 8], na=12)
    # FCF yield = FCF / Market Cap
    fcf    = safe_float(info.get("freeCashflow"))
    mktcap = safe_float(info.get("marketCap"))
    if fcf and mktcap and mktcap > 0:
        s += _ts(fcf / mktcap, [0.05, 0.03, 0.01, 0.0], [25, 20, 15, 8], na=8)
    else:
        s += 8
    s += _ts(safe_float(info.get("quickRatio")),
             [2.0, 1.5, 1.0, 0.5], [25, 20, 15, 8], na=12)
    return min(100, s)


def _calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def score_momentum(four_wk_return, info, hist):
    """0-100: price momentum and technical signals."""
    s = 0
    # 1. 4-week return
    r = four_wk_return
    if r is not None:
        if r > 10:   s += 25
        elif r > 5:  s += 20
        elif r > 2:  s += 15
        elif r > 0:  s += 10
        elif r > -2: s += 6
        elif r > -5: s += 3
    else:
        s += 8
    # 2. Price vs 50-day MA
    price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    ma50  = safe_float(info.get("fiftyDayAverage"))
    if price and ma50 and ma50 > 0:
        vs50 = (price - ma50) / ma50 * 100
        if vs50 > 8:   s += 25
        elif vs50 > 3: s += 20
        elif vs50 > 0: s += 15
        elif vs50 > -3: s += 8
        else:           s += 3
    else:
        s += 10
    # 3. Price vs 200-day MA
    ma200 = safe_float(info.get("twoHundredDayAverage"))
    if price and ma200 and ma200 > 0:
        vs200 = (price - ma200) / ma200 * 100
        if vs200 > 15:  s += 25
        elif vs200 > 8: s += 20
        elif vs200 > 0: s += 15
        elif vs200 > -8: s += 8
        else:            s += 3
    else:
        s += 10
    # 4. RSI(14) — sweet spot is 50-65 (trending up, not overbought)
    rsi_val = None
    if hist is not None and not hist.empty and len(hist) >= 20:
        try:
            rsi_s = _calc_rsi(hist["Close"])
            valid = rsi_s.dropna()
            if not valid.empty:
                rsi_val = float(valid.iloc[-1])
        except Exception:
            pass
    if rsi_val is not None:
        if 55 <= rsi_val <= 65:  s += 25
        elif 50 <= rsi_val < 55: s += 20
        elif 65 < rsi_val <= 70: s += 18
        elif 45 <= rsi_val < 50: s += 12
        elif 70 < rsi_val <= 80: s += 8
        elif 35 <= rsi_val < 45: s += 8
        elif rsi_val > 80:       s += 3
        elif 25 <= rsi_val < 35: s += 5
        else:                    s += 2
    else:
        s += 10
    return min(100, s)


def composite_score_val(g, b, m, mom, w_g=1, w_b=1, w_m=1, w_mom=1):
    total = w_g + w_b + w_m + w_mom
    if total == 0:
        return 0.0
    return round((w_g * g + w_b * b + w_m * m + w_mom * mom) / total, 1)


def score_label(score):
    if score >= 80: return "Strong Buy"
    if score >= 65: return "Buy"
    if score >= 50: return "Hold"
    if score >= 35: return "Caution"
    return "Avoid"


def score_badge(score):
    if score >= 80: return "🟢"
    if score >= 65: return "🟩"
    if score >= 50: return "🟡"
    if score >= 35: return "🟠"
    return "🔴"


# ─── Screener data fetchers ───────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def fetch_universe_4week_returns(tickers_tuple):
    """Batch-download ~5 weeks of prices; return {sym: 4w_pct_return}."""
    tickers = list(tickers_tuple)
    end   = datetime.now()
    start = end - timedelta(weeks=5)
    results = {}
    try:
        raw = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if raw.empty:
            return results
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": tickers[0]})
        for sym in tickers:
            try:
                if sym in close.columns:
                    prices = close[sym].dropna()
                    if len(prices) >= 2:
                        results[sym] = float(
                            (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100
                        )
            except Exception:
                pass
    except Exception:
        pass
    return results


def _get_rsi_macd(hist):
    """Return (rsi_float_or_None, 'Bullish'|'Bearish'|None)."""
    if hist is None or hist.empty or len(hist) < 30:
        return None, None
    rsi_val = None
    macd_sig = None
    try:
        rsi_s = _calc_rsi(hist["Close"])
        valid = rsi_s.dropna()
        if not valid.empty:
            rsi_val = float(valid.iloc[-1])
    except Exception:
        pass
    try:
        ema12 = hist["Close"].ewm(span=12).mean()
        ema26 = hist["Close"].ewm(span=26).mean()
        macd  = ema12 - ema26
        sig   = macd.ewm(span=9).mean()
        macd_sig = "Bullish" if (macd - sig).iloc[-1] > 0 else "Bearish"
    except Exception:
        pass
    return rsi_val, macd_sig


# ─── Screener renderer ────────────────────────────────────────────────────────

def _build_screener_df(data):
    display = data.copy()
    for col, key in [("1d%", "_1d"), ("5d%", "_5d"), ("1m%", "_1m"),
                     ("3m%", "_3m"), ("6m%", "_6m"), ("1y%", "_1y")]:
        if key in display.columns:
            display[col] = display[key].apply(
                lambda r: f"{r:+.2f}%" if r is not None else "N/A"
            )
        else:
            display[col] = "N/A"
    display["RSI"] = display["_rsi"].apply(
        lambda r: f"{r:.0f} ({'OB' if r > 70 else ('OS' if r < 30 else 'OK')})"
        if r is not None else "N/A"
    )
    display["Rec"] = display["Score"].apply(
        lambda s: f"{score_badge(s)} {score_label(s)}"
    )
    display["Ticker"] = display["Symbol"].apply(
        lambda s: f"https://finance.yahoo.com/quote/{s}"
    )
    cols = ["Ticker", "Price", "Sector", "1d%", "5d%", "1m%", "3m%", "6m%", "1y%",
            "Growth", "Profit", "Mgmt/Debt", "Momentum", "Score",
            "Rec", "RSI", "MACD", "MA Signal"]
    return display[cols].reset_index(drop=True)


def _show_screener_table(data, key, height, show_notes=True):
    symbols = data["Symbol"].tolist()
    df = _build_screener_df(data)

    col_cfg = {
        "Ticker": st.column_config.LinkColumn(
            "Ticker",
            display_text=r"quote/([^/?]+)",
            help="Ticker symbol — click to open on Yahoo Finance in a new tab",
            width="small",
        ),
        "Price": st.column_config.TextColumn(
            "Price ($)", width="small", disabled=True,
            help="Current share price in USD",
        ),
        "Sector": st.column_config.TextColumn(
            "Sector", disabled=True,
            help="GICS sector classification",
        ),
        "1d%": st.column_config.TextColumn("1d %", width="small", disabled=True,
            help="Price change over 1 trading day"),
        "5d%": st.column_config.TextColumn("5d %", width="small", disabled=True,
            help="Price change over 5 trading days (~1 week)"),
        "1m%": st.column_config.TextColumn("1m %", width="small", disabled=True,
            help="Price change over ~21 trading days (1 month)"),
        "3m%": st.column_config.TextColumn("3m %", width="small", disabled=True,
            help="Price change over ~63 trading days (3 months)"),
        "6m%": st.column_config.TextColumn("6m %", width="small", disabled=True,
            help="Price change over ~126 trading days (6 months)"),
        "1y%": st.column_config.TextColumn("1y %", width="small", disabled=True,
            help="Price change over ~251 trading days (1 year)"),
        "Growth": st.column_config.NumberColumn(
            "Growth", disabled=True, format="%.0f",
            help="Growth score 0–100: revenue growth (25%), EPS growth (25%), "
                 "forward vs trailing P/E ratio (25%), analyst price-target upside (25%)",
        ),
        "Profit": st.column_config.NumberColumn(
            "Profit", disabled=True, format="%.0f",
            help="Profitability score 0–100: net profit margin (25%), "
                 "return on equity ROE (25%), operating margin (25%), return on assets ROA (25%)",
        ),
        "Mgmt/Debt": st.column_config.NumberColumn(
            "Mgmt/Debt", disabled=True, format="%.0f",
            help="Management & balance-sheet score 0–100: debt-to-equity ratio (25%), "
                 "current ratio (25%), free-cash-flow yield (25%), quick ratio (25%)",
        ),
        "Momentum": st.column_config.NumberColumn(
            "Momentum", disabled=True, format="%.0f",
            help="Momentum score 0–100: 4-week price return (25%), "
                 "price vs 50-day MA (25%), price vs 200-day MA (25%), RSI(14) sweet-spot (25%)",
        ),
        "Score": st.column_config.NumberColumn(
            "Score", disabled=True, format="%.0f",
            help="Composite score — equal-weight average of Growth, Profit, Mgmt/Debt, Momentum. "
                 "≥80 Strong Buy · ≥65 Buy · ≥50 Hold · ≥35 Caution · <35 Avoid",
        ),
        "Rec": st.column_config.TextColumn(
            "Signal", disabled=True,
            help="Recommendation signal derived from Score: "
                 "🟢 Strong Buy (≥80) · 🟩 Buy (≥65) · 🟡 Hold (≥50) · 🟠 Caution (≥35) · 🔴 Avoid (<35)",
        ),
        "RSI": st.column_config.TextColumn(
            "RSI", width="small", disabled=True,
            help="14-day Relative Strength Index. "
                 "OB = Overbought (>70, may pull back) · OS = Oversold (<30, may bounce) · OK = Neutral (30–70)",
        ),
        "MACD": st.column_config.TextColumn(
            "MACD", width="small", disabled=True,
            help="MACD momentum indicator: Bullish = MACD line above signal line (upward momentum), "
                 "Bearish = MACD below signal line (downward momentum)",
        ),
        "MA Signal": st.column_config.TextColumn(
            "MA Signal", disabled=True,
            help="Price position vs moving averages. "
                 "↑ 50d = above 50-day MA · ↓ 50d = below · ↑ 200d = above 200-day MA · ↓ 200d = below",
        ),
    }

    if show_notes:
        notes_store = st.session_state.get("screener_notes", {})
        df["Notes"] = [notes_store.get(s, "") for s in symbols]
        col_cfg["Notes"] = st.column_config.TextColumn(
            "📝 Notes / Actions", width="large",
            help="Personal notes saved to disk — type and click away to save",
        )

    edited = st.data_editor(
        df,
        use_container_width=True,
        height=height,
        key=key,
        column_config=col_cfg,
        hide_index=True,
    )

    if show_notes:
        existing = st.session_state.get("screener_notes", {})
        updated = dict(zip(symbols, edited["Notes"]))
        merged = {**existing, **updated}
        merged = {k: v for k, v in merged.items() if v}  # drop blanks
        st.session_state["screener_notes"] = merged
        _save_notes(merged)


import pickle as _pickle
import json as _json

_SCREENER_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screener_cache.pkl")
_NOTES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screener_notes.json")


def _load_notes():
    try:
        if os.path.exists(_NOTES_FILE):
            with open(_NOTES_FILE, "r") as f:
                return _json.load(f)
    except Exception:
        pass
    return {}


def _save_notes(notes):
    try:
        with open(_NOTES_FILE, "w") as f:
            _json.dump({k: v for k, v in notes.items() if v}, f)
    except Exception:
        pass


_ALERTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "price_alerts.json")


def _load_alerts():
    try:
        if os.path.exists(_ALERTS_FILE):
            with open(_ALERTS_FILE, "r") as f:
                return _json.load(f)
    except Exception:
        pass
    return []


def _save_alerts(alerts):
    try:
        with open(_ALERTS_FILE, "w") as f:
            _json.dump(alerts, f)
    except Exception:
        pass


def _load_screener_cache(cache_file=None):
    cf = cache_file or _SCREENER_CACHE_FILE
    try:
        if os.path.exists(cf):
            with open(cf, "rb") as f:
                data = _pickle.load(f)
            return data.get("rows"), data.get("timestamp")
    except Exception:
        pass
    return None, None


def _save_screener_cache(rows, timestamp, cache_file=None):
    cf = cache_file or _SCREENER_CACHE_FILE
    try:
        with open(cf, "wb") as f:
            _pickle.dump({"rows": rows, "timestamp": timestamp}, f)
    except Exception:
        pass


def _clear_screener_cache(cache_file=None):
    cf = cache_file or _SCREENER_CACHE_FILE
    try:
        if os.path.exists(cf):
            os.remove(cf)
    except Exception:
        pass


def render_screener(universe=None, label="all"):
    if universe is None:
        universe = STOCK_UNIVERSE
    cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"screener_cache_{label}.pkl")
    rows_key = f"screener_rows_{label}"
    ts_key   = f"screener_cache_ts_{label}"

    st.markdown(
        f"Scans **{len(universe)} stocks** to find the **top 20 gainers** and "
        "**bottom 20 losers** over the last 4 weeks, then scores each on four "
        "equal-weight pillars."
    )

    if rows_key not in st.session_state:
        cached_rows, cached_ts = _load_screener_cache(cache_file)
        if cached_rows is not None:
            st.session_state[rows_key] = cached_rows
            st.session_state[ts_key]   = cached_ts

    if "screener_notes" not in st.session_state:
        st.session_state["screener_notes"] = _load_notes()

    # ── Scoring weights ───────────────────────────────────────────────────────
    with st.expander("⚙️ Scoring Weights & Categories", expanded=False):
        wc1, wc2, wc3, wc4 = st.columns(4)
        w_g   = wc1.slider("📈 Growth",       0, 100, 25, step=5, key=f"w_g_{label}")
        w_b   = wc2.slider("💰 Profitability", 0, 100, 25, step=5, key=f"w_b_{label}")
        w_m   = wc3.slider("🛡 Mgmt/Debt",    0, 100, 25, step=5, key=f"w_m_{label}")
        w_mom = wc4.slider("⚡ Momentum",     0, 100, 25, step=5, key=f"w_mom_{label}")
        total_w = w_g + w_b + w_m + w_mom
        if total_w > 0:
            st.caption(
                f"Effective weights — "
                f"Growth: {w_g/total_w*100:.0f}%  ·  "
                f"Profitability: {w_b/total_w*100:.0f}%  ·  "
                f"Mgmt/Debt: {w_m/total_w*100:.0f}%  ·  "
                f"Momentum: {w_mom/total_w*100:.0f}%"
            )
        else:
            st.warning("At least one weight must be > 0. Defaulting to equal weights.")
            w_g = w_b = w_m = w_mom = 25
    st.divider()

    col_btn, col_clear, col_hint = st.columns([1, 1, 3])
    run_btn   = col_btn.button("▶ Run Screener",  type="primary",
                               use_container_width=True, key=f"screener_run_{label}")
    clear_btn = col_clear.button("🗑 Clear Cache", use_container_width=True,
                                 key=f"screener_clear_{label}")
    col_hint.caption(
        "⏱ First run ~90 s · Results cached to disk indefinitely · "
        "Scoring: 25% Future Growth  ·  25% Profitability  ·  25% Mgmt/Debt  ·  25% Momentum"
    )

    if clear_btn:
        _clear_screener_cache(cache_file)
        st.session_state.pop(rows_key, None)
        st.session_state.pop(ts_key, None)
        st.rerun()

    if not run_btn and rows_key not in st.session_state:
        c = st.columns(4)
        c[0].info("**📈 Growth (25%)**\nRevenue growth · EPS growth · Fwd vs trailing P/E · Analyst upside")
        c[1].info("**💰 Profitability (25%)**\nNet margin · ROE · Operating margin · ROA")
        c[2].info("**🛡 Mgmt / Debt (25%)**\nDebt-to-equity · Current ratio · FCF yield · Quick ratio")
        c[3].info("**⚡ Momentum (25%)**\n4-week return · vs 50d MA · vs 200d MA · RSI(14)")
        return

    if run_btn:
        st.session_state.pop(rows_key, None)
        st.session_state.pop(ts_key, None)

    if rows_key not in st.session_state:
        tickers_tuple = tuple(sorted(universe))

        prog = st.progress(0, text="Fetching 4-week returns for universe…")
        returns = fetch_universe_4week_returns(tickers_tuple)

        if not returns:
            st.error("Could not download price data. Check internet connection.")
            return

        sorted_rets = sorted(returns.items(), key=lambda x: x[1], reverse=True)
        n = min(20, len(sorted_rets) // 2)
        top20    = [s for s, _ in sorted_rets[:n]]
        bottom20 = [s for s, _ in sorted_rets[-n:]]
        candidates = list(dict.fromkeys(top20 + bottom20))

        prog.progress(0.12, text=f"Identified {len(candidates)} candidates · Fetching fundamentals…")

        rows = []
        for i, sym in enumerate(candidates):
            prog.progress(
                0.12 + 0.85 * (i / len(candidates)),
                text=f"Analyzing {sym}  ({i+1}/{len(candidates)})…",
            )
            try:
                info, fin, bs, cf, hist, ed = fetch_data(sym)
                if info is None:
                    continue
                four_wk = returns.get(sym)

                g   = score_growth_topline(info)
                b   = score_bottom_line(info)
                md  = score_mgmt_debt_risk(info)
                mom = score_momentum(four_wk, info, hist)
                comp = composite_score_val(g, b, md, mom)

                rsi_val, macd_sig = _get_rsi_macd(hist)
                period_rets = calc_period_returns(hist)

                price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
                ma50  = safe_float(info.get("fiftyDayAverage"))
                ma200 = safe_float(info.get("twoHundredDayAverage"))
                vs50  = (price - ma50)  / ma50  * 100 if price and ma50  else None
                vs200 = (price - ma200) / ma200 * 100 if price and ma200 else None

                ma_parts = []
                if vs50  is not None: ma_parts.append(f"{'↑' if vs50  > 0 else '↓'} 50d ({vs50:+.1f}%)")
                if vs200 is not None: ma_parts.append(f"{'↑' if vs200 > 0 else '↓'} 200d ({vs200:+.1f}%)")

                rows.append({
                    "Symbol":    sym,
                    "Name":      (info.get("longName", sym) or sym)[:28],
                    "Price":     f"${price:.2f}" if price else "N/A",
                    "Sector":    info.get("sector", "N/A"),
                    "_4w":       four_wk,
                    "_1d":       period_rets["1d"],
                    "_5d":       period_rets["5d"],
                    "_1m":       period_rets["1m"],
                    "_3m":       period_rets["3m"],
                    "_6m":       period_rets["6m"],
                    "_1y":       period_rets["1y"],
                    "Growth":    round(g,   1),
                    "Profit":    round(b,   1),
                    "Mgmt/Debt": round(md,  1),
                    "Momentum":  round(mom, 1),
                    "Score":     comp,
                    "_rsi":      rsi_val,
                    "MACD":      macd_sig or "N/A",
                    "MA Signal": " | ".join(ma_parts) if ma_parts else "N/A",
                    "_group":    "Top 20" if sym in top20 else "Bottom 20",
                })
            except Exception:
                continue

        prog.progress(1.0, text="Done!")
        prog.empty()

        if not rows:
            st.error("No data fetched. Try again in a moment.")
            return

        now = datetime.now()
        st.session_state[rows_key] = rows
        st.session_state[ts_key]   = now
        _save_screener_cache(rows, now, cache_file)

    # ── Cache age warning ─────────────────────────────────────────────────────
    ts = st.session_state.get(ts_key)
    if ts:
        age   = datetime.now() - ts
        total = int(age.total_seconds())
        days  = total // 86400
        hours = (total % 86400) // 3600
        mins  = (total % 3600)  // 60
        parts = []
        if days:  parts.append(f"{days}d")
        if hours: parts.append(f"{hours}h")
        parts.append(f"{mins}m")
        st.warning(
            f"⚠️ Data cached on **{ts.strftime('%Y-%m-%d %H:%M')}** "
            f"({' '.join(parts)} ago). Prices and scores may be stale. "
            "Click **▶ Run Screener** to refresh or **🗑 Clear Cache** to reset."
        )

    rows = st.session_state[rows_key]
    df   = pd.DataFrame(rows)

    # Re-score with current weights (no re-fetch needed)
    df["Score"] = df.apply(
        lambda r: composite_score_val(r["Growth"], r["Profit"], r["Mgmt/Debt"], r["Momentum"],
                                      w_g, w_b, w_m, w_mom),
        axis=1,
    )

    top_df = df[df["_group"] == "Top 20"].sort_values("Score", ascending=False)
    bot_df = df[df["_group"] == "Bottom 20"].sort_values("Score", ascending=False)

    # ── Top 20 section ──
    st.markdown("---")
    st.markdown("## 🚀 Top 20 Gainers — Last 4 Weeks")
    st.caption("Sorted by composite recommendation score (highest = strongest fundamentals)")
    if not top_df.empty:
        _show_screener_table(top_df, key=f"screener_top_{label}", height=560, show_notes=False)
    else:
        st.info("No top-gainer data.")

    # ── Bottom 20 section ──
    st.markdown("---")
    st.markdown("## 📉 Bottom 20 Losers — Last 4 Weeks")
    st.caption("High score here may signal an oversold buying opportunity; low score confirms weakness")
    if not bot_df.empty:
        _show_screener_table(bot_df, key=f"screener_bot_{label}", height=560, show_notes=False)
    else:
        st.info("No bottom-loser data.")

    # ── Combined all 40, sorted by score ──
    st.markdown("---")
    st.markdown("## 📊 All Candidates — Ranked by Score")
    st.caption("📝 Notes are editable here and saved to disk automatically")
    all_sorted = df.sort_values("Score", ascending=False)

    # CSV export
    export_df = _build_screener_df(all_sorted)
    csv_bytes = export_df.drop(columns=["Notes"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Export CSV",
        data=csv_bytes,
        file_name=f"screener_{label}.csv",
        mime="text/csv",
        key=f"csv_export_{label}",
    )

    _show_screener_table(all_sorted, key=f"screener_all_{label}", height=900, show_notes=True)

    # ── Score legend ──
    st.markdown("---")
    lc = st.columns(5)
    lc[0].success("🟢 Strong Buy ≥ 80")
    lc[1].success("🟩 Buy ≥ 65")
    lc[2].warning("🟡 Hold ≥ 50")
    lc[3].warning("🟠 Caution ≥ 35")
    lc[4].error("🔴 Avoid < 35")
    st.caption(
        "RSI: OB = Overbought (>70) · OS = Oversold (<30) · OK = Neutral  |  "
        "MACD: Bullish = MACD above signal line  |  Not financial advice."
    )


# ─── Market overview tab ──────────────────────────────────────────────────────

def render_market_tab():
    st.subheader("Macro Overview")
    macro = fetch_macro_snapshot()
    if macro:
        cols = st.columns(len(macro))
        for col, (name, data) in zip(cols, macro.items()):
            chg = data.get("change", 0) or 0
            arrow = "▲" if chg >= 0 else "▼"
            color = "#22c55e" if chg >= 0 else "#ef4444"
            col.markdown(
                f"**{name}**<br>"
                f"<span style='font-size:1.1rem'>{data.get('price', 'N/A')}</span><br>"
                f"<span style='color:{color}'>{arrow} {chg:+.2f}%</span>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("Could not load macro data.")

    st.divider()
    st.subheader("Sector Performance (1-Day)")
    sector_data = fetch_sector_returns()
    if sector_data:
        names = list(sector_data.keys())
        values = [sector_data[n] for n in names]
        colors = ["#22c55e" if v >= 0 else "#ef4444" for v in values]
        fig_sector = go.Figure(go.Bar(
            x=names, y=values,
            marker_color=colors,
            text=[f"{v:+.2f}%" for v in values],
            textposition="outside",
        ))
        fig_sector.update_layout(
            yaxis_title="1-Day Return (%)",
            height=360,
            margin=dict(t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig_sector.update_yaxes(zeroline=True, zerolinecolor="#888")
        st.plotly_chart(fig_sector, use_container_width=True)

        # Heat-map grid (colored tiles)
        st.markdown("**Sector Heat Map**")
        tile_cols = st.columns(4)
        for i, (name, val) in enumerate(sector_data.items()):
            col = tile_cols[i % 4]
            bg = f"rgba(34,197,94,{min(abs(val)/3, 1):.2f})" if val >= 0 else f"rgba(239,68,68,{min(abs(val)/3, 1):.2f})"
            col.markdown(
                f"<div style='background:{bg};border-radius:6px;padding:8px 10px;margin-bottom:6px;text-align:center'>"
                f"<b>{name}</b><br><span style='font-size:1.05rem'>{val:+.2f}%</span></div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("Could not load sector data.")


# ─── Alerts tab ───────────────────────────────────────────────────────────────

def render_alerts_tab():
    st.subheader("Price Alerts")
    alerts = _load_alerts()

    # ── Add new alert ─────────────────────────────────────────────────────────
    with st.expander("➕ Add Alert", expanded=not alerts):
        a1, a2, a3, a4 = st.columns([2, 2, 2, 1])
        new_sym   = a1.text_input("Ticker", placeholder="e.g. AAPL", key="alert_sym").strip().upper()
        direction = a2.selectbox("Condition", ["Above", "Below"], key="alert_dir")
        new_price = a3.number_input("Price ($)", min_value=0.01, step=0.01, format="%.2f", key="alert_price")
        a4.markdown("<br>", unsafe_allow_html=True)
        if a4.button("Add", type="primary", key="alert_add"):
            if new_sym and new_price > 0:
                alerts.append({"symbol": new_sym, "direction": direction, "price": new_price, "triggered": False})
                _save_alerts(alerts)
                st.success(f"Alert added: {new_sym} {direction.lower()} ${new_price:.2f}")
                st.rerun()
            else:
                st.warning("Enter a ticker and target price.")

    if not alerts:
        st.info("No alerts set. Add one above.")
        return

    # ── Check current prices against alerts ──────────────────────────────────
    unique_syms = list({a["symbol"] for a in alerts})
    prices: dict[str, float] = {}
    for sym in unique_syms:
        try:
            info_a, *_ = fetch_data(sym)
            p = safe_float(info_a.get("currentPrice") or info_a.get("regularMarketPrice")) if info_a else None
            if p:
                prices[sym] = p
        except Exception:
            pass

    triggered_any = False
    for a in alerts:
        sym = a["symbol"]
        cur = prices.get(sym)
        if cur is None:
            continue
        hit = (a["direction"] == "Above" and cur >= a["price"]) or \
              (a["direction"] == "Below" and cur <= a["price"])
        if hit and not a.get("triggered"):
            a["triggered"] = True
            triggered_any = True
    if triggered_any:
        _save_alerts(alerts)

    # ── Render alerts table ───────────────────────────────────────────────────
    st.markdown(f"**{len(alerts)} alert(s)**")
    to_delete = None
    for i, a in enumerate(alerts):
        sym   = a["symbol"]
        cur   = prices.get(sym)
        hit   = a.get("triggered", False)
        row   = st.columns([2, 2, 2, 2, 1])
        status = "🔔 **TRIGGERED**" if hit else "⏳ Watching"
        row[0].markdown(f"**{sym}**")
        row[1].markdown(f"{a['direction']} **${a['price']:.2f}**")
        row[2].markdown(f"Current: **${cur:.2f}**" if cur else "Current: N/A")
        row[3].markdown(status)
        if row[4].button("🗑", key=f"del_alert_{i}", help="Delete"):
            to_delete = i
    if to_delete is not None:
        alerts.pop(to_delete)
        _save_alerts(alerts)
        st.rerun()

    if any(a.get("triggered") for a in alerts):
        if st.button("Clear Triggered Alerts", key="clear_triggered"):
            alerts = [a for a in alerts if not a.get("triggered")]
            _save_alerts(alerts)
            st.rerun()


# ─── App shell ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Stock Analysis", page_icon="📈", layout="wide")
st.title("📈 Stock Analysis Tool")
st.caption("Data sourced from Yahoo Finance via yfinance")

with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        placeholder="sk-ant-…",
        help="Required for AI analysis in the Compare tab",
    )
    st.divider()
    st.markdown("**Single Stock**")
    symbol_input = st.text_input(
        "Ticker Symbol", value="AAPL",
        placeholder="e.g. AAPL, MSFT, GOOGL",
    ).strip().upper()
    st.button("Analyze", type="primary", use_container_width=True, key="single_analyze")
    st.divider()

    # Compact macro overview in sidebar
    with st.container():
        st.markdown("**Market Pulse**")
        macro_sb = fetch_macro_snapshot()
        KEY_MACRO = ["S&P 500", "NASDAQ", "VIX", "10Y Yield", "USD/INR"]
        for name in KEY_MACRO:
            if name in macro_sb:
                d = macro_sb[name]
                chg = d.get("change", 0) or 0
                arrow = "▲" if chg >= 0 else "▼"
                color = "#22c55e" if chg >= 0 else "#ef4444"
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between'>"
                    f"<span style='color:#999;font-size:0.8rem'>{name}</span>"
                    f"<span style='color:{color};font-size:0.8rem'>{arrow}{chg:+.1f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
    st.divider()
    st.caption("Data cached 10 min · Moat is a proxy metric · Not financial advice")

tab_single, tab_compare, tab_screener, tab_market, tab_alerts = st.tabs(
    ["Single Stock", "Compare Stocks", "Stock Screener", "🌍 Market", "🔔 Alerts"]
)

# ── Single Stock tab ──────────────────────────────────────────────────────────
with tab_single:
    if not symbol_input:
        st.info("Enter a ticker symbol in the sidebar.")
    else:
        with st.spinner(f"Fetching **{symbol_input}**…"):
            try:
                s_info, s_fin, s_bs, s_cf, s_hist, s_ed = fetch_data(symbol_input)
            except RuntimeError as e:
                st.error(str(e))
                s_info = None
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                s_info = None

        if s_info is None:
            st.error(f"Could not fetch data for **{symbol_input}**. Check the ticker and try again.")
        else:
            render_single_stock(s_info, s_fin, s_bs, s_cf, s_hist, s_ed, symbol_input)

    st.caption("Data from Yahoo Finance via yfinance · Not financial advice")

# ── Compare tab ───────────────────────────────────────────────────────────────
with tab_compare:
    render_compare(api_key_input)

# ── Screener tab ──────────────────────────────────────────────────────────────
with tab_screener:
    st_ai, st_nonai, st_all = st.tabs(["🤖 AI & Tech", "🏭 Non-AI", "🌐 All Stocks"])
    with st_ai:
        render_screener(AI_UNIVERSE, "ai")
    with st_nonai:
        render_screener(NON_AI_UNIVERSE, "nonai")
    with st_all:
        render_screener(STOCK_UNIVERSE, "all")

# ── Market tab ────────────────────────────────────────────────────────────────
with tab_market:
    render_market_tab()

# ── Alerts tab ────────────────────────────────────────────────────────────────
with tab_alerts:
    render_alerts_tab()
