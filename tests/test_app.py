"""
Unit and integration tests for app.py.
Run: pytest tests/test_app.py -v
Integration tests (hitting Yahoo Finance) are marked with @pytest.mark.integration
and skipped by default. Run them with: pytest -m integration
"""
import os, sys, pickle, tempfile
from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

# ── Make app importable without executing Streamlit page code ──────────────────
def _ctx():
    m = MagicMock()
    m.__enter__ = lambda s: s
    m.__exit__  = MagicMock(return_value=False)
    return m

st_stub = MagicMock()
st_stub.cache_data  = lambda **kw: (lambda fn: fn)
st_stub.session_state = {}
# tabs/columns must return unpackable lists of context-manager-like mocks
st_stub.tabs    = lambda items: [_ctx() for _ in items]
st_stub.columns = lambda spec: [_ctx() for _ in (range(spec) if isinstance(spec, int) else spec)]
st_stub.sidebar = _ctx()
sys.modules["streamlit"] = st_stub

# Stub anthropic so importing app doesn't fail if the package is absent
sys.modules.setdefault("anthropic", MagicMock())

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import app  # noqa: E402  (import after stubs)


# ═══════════════════════════════════════════════════════════════════════════════
# Formatting helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestFmtLarge:
    def test_trillion(self):
        assert app.fmt_large(2.5e12) == "$2.50T"

    def test_billion(self):
        assert app.fmt_large(1.23e9) == "$1.23B"

    def test_million(self):
        assert app.fmt_large(4.56e6) == "$4.56M"

    def test_small(self):
        assert app.fmt_large(999) == "$999"

    def test_none(self):
        assert app.fmt_large(None) == "N/A"

    def test_invalid(self):
        assert app.fmt_large("abc") == "N/A"

    def test_negative_billion(self):
        assert app.fmt_large(-2e9) == "$-2.00B"


class TestFmtPct:
    def test_basic(self):
        assert app.fmt_pct(0.25) == "25.00%"

    def test_no_multiply(self):
        assert app.fmt_pct(25.0, multiply=False) == "25.00%"

    def test_none(self):
        assert app.fmt_pct(None) == "N/A"

    def test_zero(self):
        assert app.fmt_pct(0) == "0.00%"


class TestFmtNum:
    def test_default(self):
        assert app.fmt_num(3.14159) == "3.14"

    def test_prefix(self):
        assert app.fmt_num(5.0, prefix="$") == "$5.00"

    def test_suffix(self):
        assert app.fmt_num(2.5, suffix="x") == "2.50x"

    def test_none(self):
        assert app.fmt_num(None) == "N/A"

    def test_zero_decimals(self):
        assert app.fmt_num(42.9, decimals=0) == "43"


class TestSafeFloat:
    def test_int(self):
        assert app.safe_float(5) == 5.0

    def test_string(self):
        assert app.safe_float("3.14") == 3.14

    def test_none(self):
        assert app.safe_float(None) is None

    def test_invalid(self):
        assert app.safe_float("nope") is None


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestTierScore:
    def test_above_all(self):
        assert app._ts(0.35, [0.30, 0.20, 0.10], [25, 20, 15]) == 25

    def test_middle_tier(self):
        assert app._ts(0.25, [0.30, 0.20, 0.10], [25, 20, 15]) == 20

    def test_below_all(self):
        assert app._ts(0.01, [0.30, 0.20, 0.10], [25, 20, 15]) == 0

    def test_none_returns_na(self):
        assert app._ts(None, [0.30], [25], na=10) == 10


class TestCompositeScore:
    def test_equal_weights(self):
        assert app.composite_score_val(80, 60, 40, 80) == 65.0

    def test_all_same(self):
        assert app.composite_score_val(50, 50, 50, 50) == 50.0


class TestScoreLabel:
    @pytest.mark.parametrize("score,label", [
        (85, "Strong Buy"),
        (70, "Buy"),
        (55, "Hold"),
        (40, "Caution"),
        (20, "Avoid"),
    ])
    def test_labels(self, score, label):
        assert app.score_label(score) == label


# ═══════════════════════════════════════════════════════════════════════════════
# Derived metric calculations
# ═══════════════════════════════════════════════════════════════════════════════

def _make_df(index, values):
    """Build a single-column DataFrame like yfinance returns."""
    return pd.DataFrame({datetime(2024, 1, 1): values}, index=index)


class TestCalcROIC:
    def test_basic(self):
        fin = _make_df(["Operating Income", "Income Tax Expense", "Pretax Income"],
                       [200e6, 40e6, 200e6])
        bs  = _make_df(["Common Stock Equity", "Total Debt", "Cash And Cash Equivalents"],
                       [500e6, 200e6, 100e6])
        roic = app.calc_roic(fin, bs)
        assert roic is not None
        assert 0 < roic < 1

    def test_empty_frames(self):
        assert app.calc_roic(pd.DataFrame(), pd.DataFrame()) is None

    def test_zero_invested_capital(self):
        fin = _make_df(["Operating Income"], [100e6])
        bs  = _make_df(["Common Stock Equity", "Total Debt", "Cash And Cash Equivalents"],
                       [100e6, 0, 100e6])
        assert app.calc_roic(fin, bs) is None


class TestCalcInterestCoverage:
    def test_basic(self):
        fin = _make_df(["Operating Income", "Interest Expense"], [300e6, 50e6])
        assert app.calc_interest_coverage(fin) == pytest.approx(6.0)

    def test_zero_interest(self):
        fin = _make_df(["Operating Income", "Interest Expense"], [100e6, 0])
        assert app.calc_interest_coverage(fin) is None

    def test_empty(self):
        assert app.calc_interest_coverage(pd.DataFrame()) is None


class TestCalcBuybackYield:
    def test_basic(self):
        info = {"marketCap": 1e9}
        cf   = _make_df(["Repurchase Of Capital Stock"], [-50e6])
        result = app.calc_buyback_yield(info, cf)
        assert result == pytest.approx(0.05)

    def test_no_market_cap(self):
        assert app.calc_buyback_yield({}, pd.DataFrame()) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Moat / competitive position
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssessMoat:
    def test_strong_moat(self):
        info = {"grossMargins": 0.60, "returnOnEquity": 0.35, "operatingMargins": 0.30,
                "revenueGrowth": 0.20}
        rating, color, detail, score = app.assess_moat(info)
        assert rating == "Strong Moat"
        assert color == "green"
        assert score >= 5

    def test_narrow_moat(self):
        rating, color, detail, score = app.assess_moat({})
        assert rating == "Narrow / No Moat"
        assert color == "red"

    def test_moderate_moat(self):
        info = {"grossMargins": 0.40, "returnOnEquity": 0.22}
        rating, color, _, score = app.assess_moat(info)
        assert rating == "Moderate Moat"


class TestAssessCompetitivePosition:
    def test_mega_cap(self):
        cap_desc, sector, industry = app.assess_competitive_position(
            {"marketCap": 300e9, "sector": "Tech", "industry": "Software"}
        )
        assert "Mega Cap" in cap_desc

    def test_small_cap(self):
        cap_desc, _, _ = app.assess_competitive_position({"marketCap": 500e6, "sector": "", "industry": ""})
        assert "Small" in cap_desc

    def test_unknown(self):
        cap_desc, _, _ = app.assess_competitive_position({})
        assert cap_desc == "Unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# Score functions
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoreGrowthTopline:
    def test_high_growth(self):
        info = {
            "revenueGrowth": 0.35, "earningsGrowth": 0.35,
            "trailingPE": 30, "forwardPE": 20,
            "currentPrice": 100, "targetMeanPrice": 140,
        }
        assert app.score_growth_topline(info) >= 80

    def test_empty_info(self):
        s = app.score_growth_topline({})
        assert 0 <= s <= 100

    def test_capped_at_100(self):
        info = {
            "revenueGrowth": 1.0, "earningsGrowth": 1.0,
            "trailingPE": 50, "forwardPE": 10,
            "currentPrice": 50, "targetMeanPrice": 100,
        }
        assert app.score_growth_topline(info) <= 100


class TestScoreBottomLine:
    def test_high_profitability(self):
        info = {
            "profitMargins": 0.30, "returnOnEquity": 0.30,
            "operatingMargins": 0.30, "returnOnAssets": 0.15,
        }
        assert app.score_bottom_line(info) >= 80

    def test_empty(self):
        s = app.score_bottom_line({})
        assert 0 <= s <= 100


class TestScoreMgmtDebt:
    def test_low_debt_high_liquidity(self):
        info = {
            "debtToEquity": 20, "currentRatio": 3.0,
            "freeCashflow": 100e6, "marketCap": 1e9,
            "quickRatio": 2.5,
        }
        assert app.score_mgmt_debt_risk(info) >= 80

    def test_high_debt(self):
        info = {"debtToEquity": 500, "currentRatio": 0.4, "quickRatio": 0.3}
        assert app.score_mgmt_debt_risk(info) <= 40


# ═══════════════════════════════════════════════════════════════════════════════
# Disk cache helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestScreenerCache:
    def test_save_and_load(self, tmp_path, monkeypatch):
        cache_file = str(tmp_path / "screener_cache.pkl")
        monkeypatch.setattr(app, "_SCREENER_CACHE_FILE", cache_file)

        rows = [{"Symbol": "AAPL", "Score": 75.0}]
        ts   = datetime(2025, 6, 1, 12, 0, 0)
        app._save_screener_cache(rows, ts)

        loaded_rows, loaded_ts = app._load_screener_cache()
        assert loaded_rows == rows
        assert loaded_ts == ts

    def test_clear(self, tmp_path, monkeypatch):
        cache_file = str(tmp_path / "screener_cache.pkl")
        monkeypatch.setattr(app, "_SCREENER_CACHE_FILE", cache_file)

        app._save_screener_cache([{"Symbol": "MSFT"}], datetime.now())
        assert os.path.exists(cache_file)

        app._clear_screener_cache()
        assert not os.path.exists(cache_file)

    def test_load_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app, "_SCREENER_CACHE_FILE", str(tmp_path / "missing.pkl"))
        rows, ts = app._load_screener_cache()
        assert rows is None
        assert ts is None

    def test_load_corrupt_file(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "bad.pkl"
        cache_file.write_bytes(b"not valid pickle data!!!")
        monkeypatch.setattr(app, "_SCREENER_CACHE_FILE", str(cache_file))
        rows, ts = app._load_screener_cache()
        assert rows is None


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests  (require internet; skipped by default)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestFetchData:
    def test_fetch_aapl(self):
        info, fin, bs, cf, hist, ed = app.fetch_data("AAPL")
        assert info is not None
        assert "symbol" in info
        assert info["symbol"] == "AAPL"

    def test_invalid_ticker(self):
        info, *_ = app.fetch_data("XXXXINVALID999")
        assert info is None

    def test_hist_has_close(self):
        _, _, _, _, hist, _ = app.fetch_data("MSFT")
        assert hist is not None and not hist.empty
        assert "Close" in hist.columns


@pytest.mark.integration
class TestFetchUniverseReturns:
    def test_returns_dict(self):
        results = app.fetch_universe_4week_returns(("AAPL", "MSFT"))
        assert isinstance(results, dict)
        assert len(results) >= 1
        for val in results.values():
            assert isinstance(val, float)
