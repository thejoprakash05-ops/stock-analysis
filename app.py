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

tab_single, tab_compare = st.tabs(["Single Stock", "Compare 3 Stocks"])

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
