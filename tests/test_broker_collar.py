# ABOUTME: Tests for collar strategy module with mocked Yahoo Finance.
# ABOUTME: Validates volatility, earnings, and collar analysis.

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from trading_skills.broker.collar import (
    analyze_collar,
    get_call_market_price,
    get_earnings_date,
    get_put_chain,
    get_stock_volatility,
)

MODULE = "trading_skills.broker.collar"


class TestGetEarningsDate:
    """Tests for earnings date fetching."""

    @patch(f"{MODULE}.get_next_earnings_date")
    def test_returns_earnings_date(self, mock_ned):
        mock_ned.return_value = "2025-04-15"

        dt, timing = get_earnings_date("AAPL")
        assert dt is not None
        assert dt.month == 4
        assert dt.day == 15

    @patch(f"{MODULE}.get_next_earnings_date")
    def test_no_earnings_date(self, mock_ned):
        mock_ned.return_value = None

        dt, timing = get_earnings_date("AAPL")
        assert dt is None

    @patch(f"{MODULE}.get_next_earnings_date")
    def test_exception_returns_none(self, mock_ned):
        mock_ned.side_effect = Exception("API Error")
        dt, timing = get_earnings_date("AAPL")
        assert dt is None


class TestGetStockVolatility:
    """Tests for volatility calculation."""

    @patch(f"{MODULE}.yf.Ticker")
    def test_returns_volatility(self, mock_ticker):
        dates = pd.date_range(end=datetime.now(), periods=60, freq="D")
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(60) * 2)
        mock_df = pd.DataFrame(
            {
                "Open": prices - 0.5,
                "High": prices + 1,
                "Low": prices - 1,
                "Close": prices,
                "Volume": np.random.randint(1000000, 5000000, 60),
            },
            index=dates,
        )
        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_instance

        result = get_stock_volatility("AAPL")
        assert "annual_vol" in result
        assert "daily_vol" in result
        assert "vol_class" in result
        assert result["annual_vol"] > 0
        assert result["vol_class"] in ["LOW", "MODERATE", "HIGH", "VERY HIGH", "EXTREME"]

    @patch(f"{MODULE}.yf.Ticker")
    def test_insufficient_data(self, mock_ticker):
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame(
            {"Close": [100, 101]}, index=pd.date_range("2025-01-01", periods=2)
        )
        mock_ticker.return_value = mock_instance

        result = get_stock_volatility("AAPL")
        assert "error" in result

    @patch(f"{MODULE}.yf.Ticker")
    def test_expected_moves(self, mock_ticker):
        dates = pd.date_range(end=datetime.now(), periods=60, freq="D")
        prices = np.linspace(100, 110, 60)
        mock_df = pd.DataFrame(
            {
                "Open": prices,
                "High": prices + 1,
                "Low": prices - 1,
                "Close": prices,
                "Volume": [1000000] * 60,
            },
            index=dates,
        )
        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_instance

        result = get_stock_volatility("AAPL")
        assert "move_1_week" in result
        assert "move_2_weeks" in result
        assert "move_3_weeks" in result
        assert result["move_1_week"] < result["move_2_weeks"] < result["move_3_weeks"]

    @patch(f"{MODULE}.yf.Ticker")
    def test_vol_class_extreme(self, mock_ticker):
        dates = pd.date_range(end=datetime.now(), periods=60, freq="D")
        # Very high daily volatility → annual_vol > 0.80
        np.random.seed(1)
        prices = 100 + np.cumsum(np.random.randn(60) * 15)
        mock_df = pd.DataFrame(
            {
                "Close": prices,
                "Open": prices,
                "High": prices + 5,
                "Low": prices - 5,
                "Volume": [1e6] * 60,
            },
            index=dates,
        )
        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_instance

        result = get_stock_volatility("AAPL")
        assert result.get("vol_class") in ["EXTREME", "VERY HIGH", "HIGH", "MODERATE", "LOW"]

    @patch(f"{MODULE}.yf.Ticker")
    def test_vol_class_very_high(self, mock_ticker):
        dates = pd.date_range(end=datetime.now(), periods=60, freq="D")
        np.random.seed(2)
        prices = 100 + np.cumsum(np.random.randn(60) * 8)
        mock_df = pd.DataFrame(
            {
                "Close": prices,
                "Open": prices,
                "High": prices + 3,
                "Low": prices - 3,
                "Volume": [1e6] * 60,
            },
            index=dates,
        )
        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_instance

        result = get_stock_volatility("AAPL")
        assert "vol_class" in result

    @patch(f"{MODULE}.yf.Ticker")
    def test_vol_exception_returns_error(self, mock_ticker):
        mock_instance = MagicMock()
        mock_instance.history.side_effect = Exception("API error")
        mock_ticker.return_value = mock_instance

        result = get_stock_volatility("AAPL")
        assert "error" in result


class TestGetPutChain:
    """Tests for put chain fetching."""

    @patch(f"{MODULE}.yf.Ticker")
    def test_returns_puts(self, mock_ticker):
        mock_chain = MagicMock()
        puts_df = pd.DataFrame(
            {
                "strike": [90.0, 95.0],
                "bid": [1.0, 1.5],
                "ask": [1.2, 1.8],
                "openInterest": [100, 200],
                "impliedVolatility": [0.35, 0.38],
            }
        )
        mock_chain.puts = puts_df
        mock_instance = MagicMock()
        mock_instance.options = ["2025-06-20"]
        mock_instance.option_chain.return_value = mock_chain
        mock_ticker.return_value = mock_instance

        result = get_put_chain("AAPL", "2025-06-20")
        assert len(result) == 2
        assert result[0]["strike"] == 90.0
        assert result[0]["mid"] == pytest.approx(1.1)
        assert "iv" in result[0]

    @patch(f"{MODULE}.yf.Ticker")
    def test_returns_empty_when_expiry_not_found(self, mock_ticker):
        mock_instance = MagicMock()
        mock_instance.options = ["2025-07-18"]
        mock_ticker.return_value = mock_instance

        result = get_put_chain("AAPL", "2025-06-20")
        assert result == []

    @patch(f"{MODULE}.yf.Ticker")
    def test_returns_empty_on_exception(self, mock_ticker):
        mock_ticker.side_effect = Exception("API error")

        result = get_put_chain("AAPL", "2025-06-20")
        assert result == []


class TestGetCallMarketPrice:
    """Tests for call option market price fetching."""

    @patch(f"{MODULE}.yf.Ticker")
    def test_returns_mid_price(self, mock_ticker):
        calls_df = pd.DataFrame({"strike": [150.0], "bid": [5.0], "ask": [5.4], "lastPrice": [5.2]})
        mock_chain = MagicMock()
        mock_chain.calls = calls_df
        mock_instance = MagicMock()
        mock_instance.options = ["2026-01-16"]
        mock_instance.option_chain.return_value = mock_chain
        mock_ticker.return_value = mock_instance

        result = get_call_market_price("AAPL", 150.0, "20260116")
        assert result == pytest.approx(5.2)

    @patch(f"{MODULE}.yf.Ticker")
    def test_closest_expiry_within_7_days(self, mock_ticker):
        calls_df = pd.DataFrame({"strike": [150.0], "bid": [4.0], "ask": [4.4], "lastPrice": [4.2]})
        mock_chain = MagicMock()
        mock_chain.calls = calls_df
        mock_instance = MagicMock()
        # Expiry is 3 days off from requested
        mock_instance.options = ["2026-01-19"]
        mock_instance.option_chain.return_value = mock_chain
        mock_ticker.return_value = mock_instance

        result = get_call_market_price("AAPL", 150.0, "20260116")
        assert result is not None

    @patch(f"{MODULE}.yf.Ticker")
    def test_expiry_too_far_returns_none(self, mock_ticker):
        mock_instance = MagicMock()
        # No expiry within 7 days of requested
        mock_instance.options = ["2026-03-20"]
        mock_ticker.return_value = mock_instance

        result = get_call_market_price("AAPL", 150.0, "20260116")
        assert result is None

    @patch(f"{MODULE}.yf.Ticker")
    def test_bid_zero_uses_last_price(self, mock_ticker):
        calls_df = pd.DataFrame({"strike": [150.0], "bid": [0.0], "ask": [0.0], "lastPrice": [3.5]})
        mock_chain = MagicMock()
        mock_chain.calls = calls_df
        mock_instance = MagicMock()
        mock_instance.options = ["2026-01-16"]
        mock_instance.option_chain.return_value = mock_chain
        mock_ticker.return_value = mock_instance

        result = get_call_market_price("AAPL", 150.0, "20260116")
        assert result == pytest.approx(3.5)

    @patch(f"{MODULE}.yf.Ticker")
    def test_closest_strike_used(self, mock_ticker):
        calls_df = pd.DataFrame(
            {
                "strike": [148.0, 152.0],
                "bid": [4.0, 3.0],
                "ask": [4.4, 3.4],
                "lastPrice": [4.2, 3.2],
            }
        )
        mock_chain = MagicMock()
        mock_chain.calls = calls_df
        mock_instance = MagicMock()
        mock_instance.options = ["2026-01-16"]
        mock_instance.option_chain.return_value = mock_chain
        mock_ticker.return_value = mock_instance

        # Strike 150 not in chain, but 148 is within $5
        result = get_call_market_price("AAPL", 150.0, "20260116")
        assert result is not None

    @patch(f"{MODULE}.yf.Ticker")
    def test_exception_returns_none(self, mock_ticker):
        mock_ticker.side_effect = Exception("API error")

        result = get_call_market_price("AAPL", 150.0, "20260116")
        assert result is None


class TestAnalyzeCollar:
    """Tests for collar analysis with mocked data."""

    @patch(f"{MODULE}.get_call_market_price")
    @patch(f"{MODULE}.get_put_chain")
    @patch("trading_skills.options.get_expiries")
    @patch(f"{MODULE}.get_stock_volatility")
    def test_basic_analysis(self, mock_vol, mock_expiries, mock_puts, mock_call_price):
        mock_vol.return_value = {
            "annual_vol": 0.35,
            "daily_vol": 0.022,
            "vol_class": "MODERATE",
            "current_price": 150.0,
            "move_1_week": 5.0,
            "move_1_week_pct": 3.3,
            "move_2_weeks": 7.0,
            "move_2_weeks_pct": 4.7,
            "move_3_weeks": 8.5,
            "move_3_weeks_pct": 5.7,
        }

        future = datetime.now() + timedelta(days=30)
        expiry = future.strftime("%Y-%m-%d")
        mock_expiries.return_value = [expiry]

        mock_puts.return_value = [
            {"strike": 135.0, "bid": 1.50, "ask": 2.00, "mid": 1.75, "oi": 100, "iv": 0.40},
            {"strike": 140.0, "bid": 2.50, "ask": 3.00, "mid": 2.75, "oi": 200, "iv": 0.38},
            {"strike": 143.0, "bid": 3.50, "ask": 4.00, "mid": 3.75, "oi": 150, "iv": 0.36},
        ]

        mock_call_price.return_value = 25.0

        earnings_date = datetime.now() + timedelta(days=20)
        result = analyze_collar(
            symbol="AAPL",
            current_price=150.0,
            long_strike=130.0,
            long_expiry="20260121",
            long_qty=5,
            long_cost=25.0,
            short_positions=[{"strike": 160.0, "expiry": "20250321"}],
            earnings_date=earnings_date,
        )

        assert "volatility" in result
        assert "put_analysis" in result
        assert "symbol" in result
        assert result["symbol"] == "AAPL"

    @patch(f"{MODULE}.get_call_market_price")
    @patch(f"{MODULE}.get_put_chain")
    @patch("trading_skills.options.get_expiries")
    @patch(f"{MODULE}.get_stock_volatility")
    def test_no_earnings_date_uses_near_term_expiries(
        self, mock_vol, mock_expiries, mock_puts, mock_call_price
    ):
        mock_vol.return_value = {
            "annual_vol": 0.35,
            "daily_vol": 0.022,
            "vol_class": "MODERATE",
            "current_price": 150.0,
            "move_1_week": 5.0,
            "move_1_week_pct": 3.3,
            "move_2_weeks": 7.0,
            "move_2_weeks_pct": 4.7,
            "move_3_weeks": 8.5,
            "move_3_weeks_pct": 5.7,
        }

        future = datetime.now() + timedelta(days=30)
        expiry = future.strftime("%Y-%m-%d")
        mock_expiries.return_value = [expiry]
        mock_puts.return_value = []
        mock_call_price.return_value = 25.0

        result = analyze_collar(
            symbol="AAPL",
            current_price=150.0,
            long_strike=130.0,
            long_expiry="20260121",
            long_qty=2,
            long_cost=25.0,
            short_positions=[],
            earnings_date=None,  # No earnings date → elif expiries: branch
        )

        assert result["symbol"] == "AAPL"
        assert result["earnings_date"] is None

    @patch(f"{MODULE}.get_call_market_price")
    @patch(f"{MODULE}.get_put_chain")
    @patch("trading_skills.options.get_expiries")
    @patch(f"{MODULE}.get_stock_volatility")
    def test_otm_long_call_scenario(self, mock_vol, mock_expiries, mock_puts, mock_call_price):
        mock_vol.return_value = {
            "annual_vol": 0.35,
            "daily_vol": 0.022,
            "vol_class": "MODERATE",
            "current_price": 90.0,
            "move_1_week": 3.0,
            "move_1_week_pct": 3.3,
            "move_2_weeks": 4.5,
            "move_2_weeks_pct": 5.0,
            "move_3_weeks": 5.5,
            "move_3_weeks_pct": 6.1,
        }
        mock_expiries.return_value = []
        mock_puts.return_value = []
        mock_call_price.return_value = 5.0  # Non-None → uses actual price

        result = analyze_collar(
            symbol="AAPL",
            current_price=90.0,
            long_strike=110.0,  # long_strike > current_price * 1.05 → OTM branch
            long_expiry="20260121",
            long_qty=1,
            long_cost=10.0,
            short_positions=[],
            earnings_date=None,
        )

        assert "unprotected_loss_10" in result
        assert result["long_strike"] == 110.0

    @patch(f"{MODULE}.get_call_market_price")
    @patch(f"{MODULE}.get_put_chain")
    @patch("trading_skills.options.get_expiries")
    @patch(f"{MODULE}.get_stock_volatility")
    def test_no_market_price_uses_black_scholes(
        self, mock_vol, mock_expiries, mock_puts, mock_call_price
    ):
        mock_vol.return_value = {
            "annual_vol": 0.35,
            "daily_vol": 0.022,
            "vol_class": "MODERATE",
            "current_price": 150.0,
            "move_1_week": 5.0,
            "move_1_week_pct": 3.3,
            "move_2_weeks": 7.0,
            "move_2_weeks_pct": 4.7,
            "move_3_weeks": 8.5,
            "move_3_weeks_pct": 5.7,
        }
        mock_expiries.return_value = []
        mock_puts.return_value = []
        mock_call_price.return_value = None  # None → BS fallback

        result = analyze_collar(
            symbol="AAPL",
            current_price=150.0,
            long_strike=130.0,
            long_expiry="20260121",
            long_qty=1,
            long_cost=25.0,
            short_positions=[],
            earnings_date=None,
        )

        assert "unprotected_loss_10" in result
        assert result["long_value_now"] > 0
