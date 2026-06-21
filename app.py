"""
Stock Analysis Tool - Single stock deep-dive + 3-stock comparison with AI summary
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
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

STOCK_UNIVERSE = list(dict.fromkeys([
    # Tech / Semis
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","AVGO","ORCL","CRM",
    "ADBE","AMD","QCOM","INTC","TXN","AMAT","MU","LRCX","KLAC","MRVL",
    "NOW","SNPS","CDNS","PANW","CRWD","SNOW","PLTR","DDOG","ZS","FTNT",
    # Financials
    "JPM","BAC","WFC","GS","MS","BLK","V","MA","AXP","C","COF","SPGI","MCO",
    # Healthcare
    "JNJ","UNH","PFE","ABBV","MRK","ABT","TMO","DHR","CVS","CI","ISRG","GILD","LLY",
    # Consumer Discretionary
    "HD","WMT","COST","TGT","MCD","SBUX","NKE","TJX","LOW","BKNG","DG","AMZN",
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
    # Growth / Other
    "COIN","UBER","ABNB","SHOP","SQ","TTD","RBLX","BRK-B","BX","KKR","MSCI",
]))


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

    if hist is not None and not hist.empty:
        with st.expander("📊 1-Year Price Chart", expanded=True):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Price",
                                     line=dict(color="#1f77b4", width=2)))
            if ma50:
                fig.add_hline(y=ma50, line_dash="dot", line_color="orange",
                              annotation_text=f"50-Day MA ${ma50:.2f}")
            if ma200:
                fig.add_hline(y=ma200, line_dash="dot", line_color="red",
                              annotation_text=f"200-Day MA ${ma200:.2f}")
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0),
                              xaxis_title=None, yaxis_title="Price",
                              legend=dict(orientation="h"),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

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
    st.markdown("Enter up to 3 ticker symbols to compare all metrics side by side.")

    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    t1 = col1.text_input("Stock 1", value="AAPL", key="cmp1").strip().upper()
    t2 = col2.text_input("Stock 2", value="MSFT", key="cmp2").strip().upper()
    t3 = col3.text_input("Stock 3", value="GOOGL", key="cmp3").strip().upper()
    compare_btn = col4.button("Compare", type="primary", use_container_width=True,
                              key="compare_btn")

    if not compare_btn:
        st.info("Fill in the tickers above and click **Compare**.")
        return

    tickers = [t for t in [t1, t2, t3] if t]
    if len(tickers) < 2:
        st.warning("Enter at least 2 ticker symbols.")
        return

    # Fetch all stocks
    all_metrics = []
    all_hists   = {}
    errors = []

    progress = st.progress(0, text="Fetching data…")
    for i, sym in enumerate(tickers):
        progress.progress((i) / len(tickers), text=f"Fetching {sym}…")
        try:
            info, fin, bs, cf, hist, ed = fetch_data(sym)
            if info is None:
                errors.append(f"{sym}: could not fetch data (check ticker)")
                continue
            m = extract_all_metrics(sym, info, fin, bs, cf, hist, ed)
            all_metrics.append(m)
            all_hists[sym] = hist
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
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    with st.expander("📊 Normalized 1-Year Price Performance (base = 100)", expanded=True):
        fig = go.Figure()
        for i, sym in enumerate(symbols):
            hist = all_hists.get(sym)
            if hist is not None and not hist.empty:
                base = hist["Close"].iloc[0]
                normalized = hist["Close"] / base * 100
                fig.add_trace(go.Scatter(
                    x=hist.index, y=normalized,
                    name=f"{sym} — {names[sym]}",
                    line=dict(color=colors[i % len(colors)], width=2),
                ))
        fig.add_hline(y=100, line_dash="dot", line_color="gray", line_width=1)
        fig.update_layout(
            height=380, margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title=None, yaxis_title="Indexed (start = 100)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Metric comparison tables ──────────────────────────────────────────────
    st.divider()
    st.markdown("### Metric Comparison")
    st.caption("🟢 Best value in row  ·  🔴 Worst value  ·  Gray = middle or N/A")

    for section_name, rows in COMPARE_SECTIONS.items():
        st.markdown(f"**{section_name}**")
        styled_df = build_comparison_df(all_metrics, rows)
        st.dataframe(styled_df, use_container_width=True)
        st.markdown("")

    # ── Qualitative row: Moat + Rec ───────────────────────────────────────────
    st.markdown("**Qualitative Summary**")
    moat_icon = {"green": "🟢", "orange": "🟡", "red": "🔴"}
    qual_rows = []
    for m in all_metrics:
        qual_rows.append({
            "Stock": m["symbol"],
            "Moat": f"{moat_icon.get(m['moat_color'],'')} {m['moat_rating']}",
            "Market Position": m["cap_desc"],
            "Analyst Rec.": m["rec_label"],
            "Earnings Date": m["earnings_date"],
        })
    st.dataframe(pd.DataFrame(qual_rows).set_index("Stock"), use_container_width=True)

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


def composite_score_val(g, b, m, mom):
    return round(0.25 * g + 0.25 * b + 0.25 * m + 0.25 * mom, 1)


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
    display["4W Ret%"] = display["_4w"].apply(
        lambda r: f"{r:+.2f}%" if r is not None else "N/A"
    )
    display["RSI"] = display["_rsi"].apply(
        lambda r: f"{r:.0f} ({'OB' if r > 70 else ('OS' if r < 30 else 'OK')})"
        if r is not None else "N/A"
    )
    display["Rec"] = display["Score"].apply(
        lambda s: f"{score_badge(s)} {score_label(s)}"
    )
    display["YF"] = display["Symbol"].apply(
        lambda s: f"https://finance.yahoo.com/quote/{s}"
    )
    cols = ["Symbol", "YF", "Price", "Sector", "4W Ret%",
            "Growth", "Profit", "Mgmt/Debt", "Momentum", "Score",
            "Rec", "RSI", "MACD", "MA Signal"]
    return display[cols].reset_index(drop=True)


def _show_screener_table(data, key, height):
    df = _build_screener_df(data)
    notes_store = st.session_state.get("screener_notes", {})
    df["Notes"] = df["Symbol"].map(lambda s: notes_store.get(s, ""))

    edited = st.data_editor(
        df,
        use_container_width=True,
        height=height,
        key=key,
        column_config={
            "Symbol":    st.column_config.TextColumn("Ticker", width="small", disabled=True),
            "YF":        st.column_config.LinkColumn("Yahoo Finance", display_text="📊 Open", width="small"),
            "Price":     st.column_config.TextColumn("Price", width="small", disabled=True),
            "Sector":    st.column_config.TextColumn("Sector", disabled=True),
            "4W Ret%":   st.column_config.TextColumn("4W Ret%", width="small", disabled=True),
            "Growth":    st.column_config.ProgressColumn("Growth",    min_value=0, max_value=100, format="%.0f"),
            "Profit":    st.column_config.ProgressColumn("Profit",    min_value=0, max_value=100, format="%.0f"),
            "Mgmt/Debt": st.column_config.ProgressColumn("Mgmt/Debt", min_value=0, max_value=100, format="%.0f"),
            "Momentum":  st.column_config.ProgressColumn("Momentum",  min_value=0, max_value=100, format="%.0f"),
            "Score":     st.column_config.ProgressColumn("Score",     min_value=0, max_value=100, format="%.0f"),
            "Rec":       st.column_config.TextColumn("Signal",    disabled=True),
            "RSI":       st.column_config.TextColumn("RSI",       width="small", disabled=True),
            "MACD":      st.column_config.TextColumn("MACD",      width="small", disabled=True),
            "MA Signal": st.column_config.TextColumn("MA Signal", disabled=True),
            "Notes":     st.column_config.TextColumn("📝 Notes / Actions", width="large"),
        },
        hide_index=True,
    )

    updated = dict(zip(edited["Symbol"], edited["Notes"]))
    st.session_state["screener_notes"] = {**notes_store, **updated}


import pickle as _pickle

_SCREENER_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screener_cache.pkl")


def _load_screener_cache():
    try:
        if os.path.exists(_SCREENER_CACHE_FILE):
            with open(_SCREENER_CACHE_FILE, "rb") as f:
                data = _pickle.load(f)
            return data.get("rows"), data.get("timestamp")
    except Exception:
        pass
    return None, None


def _save_screener_cache(rows, timestamp):
    try:
        with open(_SCREENER_CACHE_FILE, "wb") as f:
            _pickle.dump({"rows": rows, "timestamp": timestamp}, f)
    except Exception:
        pass


def _clear_screener_cache():
    try:
        if os.path.exists(_SCREENER_CACHE_FILE):
            os.remove(_SCREENER_CACHE_FILE)
    except Exception:
        pass


def render_screener():
    st.markdown(
        "Scans **~120 popular stocks** to find the **top 20 gainers** and "
        "**bottom 20 losers** over the last 4 weeks, then scores each on four "
        "equal-weight pillars."
    )

    # Load disk cache into session state on first visit this session
    if "screener_rows" not in st.session_state:
        cached_rows, cached_ts = _load_screener_cache()
        if cached_rows is not None:
            st.session_state["screener_rows"]     = cached_rows
            st.session_state["screener_cache_ts"] = cached_ts

    col_btn, col_clear, col_hint = st.columns([1, 1, 3])
    run_btn   = col_btn.button("▶ Run Screener",  type="primary",
                               use_container_width=True, key="screener_run")
    clear_btn = col_clear.button("🗑 Clear Cache", use_container_width=True,
                                 key="screener_clear")
    col_hint.caption(
        "⏱ First run ~90 s · Results cached to disk indefinitely · "
        "Scoring: 25% Future Growth  ·  25% Profitability  ·  25% Mgmt/Debt  ·  25% Momentum"
    )

    if clear_btn:
        _clear_screener_cache()
        st.session_state.pop("screener_rows", None)
        st.session_state.pop("screener_cache_ts", None)
        st.rerun()

    if not run_btn and "screener_rows" not in st.session_state:
        c = st.columns(4)
        c[0].info("**📈 Growth (25%)**\nRevenue growth · EPS growth · Fwd vs trailing P/E · Analyst upside")
        c[1].info("**💰 Profitability (25%)**\nNet margin · ROE · Operating margin · ROA")
        c[2].info("**🛡 Mgmt / Debt (25%)**\nDebt-to-equity · Current ratio · FCF yield · Quick ratio")
        c[3].info("**⚡ Momentum (25%)**\n4-week return · vs 50d MA · vs 200d MA · RSI(14)")
        return

    if run_btn:
        st.session_state.pop("screener_rows", None)
        st.session_state.pop("screener_cache_ts", None)

    if "screener_rows" not in st.session_state:
        tickers_tuple = tuple(sorted(STOCK_UNIVERSE))

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
        st.session_state["screener_rows"]     = rows
        st.session_state["screener_cache_ts"] = now
        _save_screener_cache(rows, now)

    # ── Cache age warning ─────────────────────────────────────────────────────
    ts = st.session_state.get("screener_cache_ts")
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

    rows = st.session_state["screener_rows"]
    df   = pd.DataFrame(rows)

    top_df = df[df["_group"] == "Top 20"].sort_values("Score", ascending=False)
    bot_df = df[df["_group"] == "Bottom 20"].sort_values("Score", ascending=False)

    # ── Top 20 section ──
    st.markdown("---")
    st.markdown("## 🚀 Top 20 Gainers — Last 4 Weeks")
    st.caption("Sorted by composite recommendation score (highest = strongest fundamentals)")
    if not top_df.empty:
        _show_screener_table(top_df, key="screener_top", height=560)
    else:
        st.info("No top-gainer data.")

    # ── Bottom 20 section ──
    st.markdown("---")
    st.markdown("## 📉 Bottom 20 Losers — Last 4 Weeks")
    st.caption("High score here may signal an oversold buying opportunity; low score confirms weakness")
    if not bot_df.empty:
        _show_screener_table(bot_df, key="screener_bot", height=560)
    else:
        st.info("No bottom-loser data.")

    # ── Combined all 40, sorted by score ──
    st.markdown("---")
    st.markdown("## 📊 All Candidates — Ranked by Score")
    all_sorted = df.sort_values("Score", ascending=False)
    _show_screener_table(all_sorted, key="screener_all", height=900)

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
    st.caption("Data cached 10 min · Moat is a proxy metric · Not financial advice")

tab_single, tab_compare, tab_screener = st.tabs(["Single Stock", "Compare 3 Stocks", "Stock Screener"])

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
    render_screener()
