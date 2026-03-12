"""
回测引擎和风控模块单元测试
"""
import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine, Portfolio, Position
from backtest.metrics import PerformanceMetrics
from risk.manager import RiskManager
from strategy.momentum import MomentumStrategy


def _generate_mock_data(n_stocks: int = 5, n_days: int = 200) -> dict:
    dates = pd.bdate_range("2023-01-01", periods=n_days, freq="B")
    data = {}
    np.random.seed(42)
    for i in range(n_stocks):
        code = f"TEST_{i:03d}"
        price = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n_days)))
        volume = np.random.randint(1000, 100000, n_days).astype(float)
        df = pd.DataFrame({
            "open": price * 0.99,
            "high": price * 1.02,
            "low": price * 0.98,
            "close": price,
            "volume": volume,
            "amount": price * volume,
            "turnover": np.random.uniform(0.5, 5.0, n_days),
        }, index=dates)
        data[code] = df
    return data


class TestBacktestEngine:
    def test_basic_backtest(self):
        data = _generate_mock_data()
        strategy = MomentumStrategy({"lookback": 20, "top_pct": 0.4})
        engine = BacktestEngine(initial_capital=1_000_000)
        result = engine.run(strategy, data)

        assert result is not None
        assert not result.nav_df.empty
        assert "total_return" in result.metrics
        assert "sharpe_ratio" in result.metrics
        assert "max_drawdown" in result.metrics

    def test_nav_starts_at_one(self):
        data = _generate_mock_data()
        strategy = MomentumStrategy()
        engine = BacktestEngine()
        result = engine.run(strategy, data)

        assert abs(result.nav_df["nav"].iloc[0] - 1.0) < 0.01

    def test_commission_impact(self):
        data = _generate_mock_data()
        strategy = MomentumStrategy({"lookback": 10, "top_pct": 0.5})

        engine_low = BacktestEngine(commission_rate=0.0001)
        engine_high = BacktestEngine(commission_rate=0.003)

        result_low = engine_low.run(strategy, data)
        result_high = engine_high.run(strategy, data)

        assert result_low.metrics["total_return"] >= result_high.metrics["total_return"]


class TestPerformanceMetrics:
    def test_positive_returns(self):
        dates = pd.bdate_range("2023-01-01", periods=252)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        nav = (1 + returns).cumprod()
        nav_df = pd.DataFrame({
            "returns": returns,
            "nav": nav,
            "total_value": nav * 1_000_000,
        })

        metrics = PerformanceMetrics.compute(nav_df)
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert metrics["max_drawdown"] >= 0

    def test_rolling_sharpe(self):
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        rolling_sharpe = PerformanceMetrics.rolling_sharpe(returns, window=60)
        assert len(rolling_sharpe) == 252


class TestPortfolio:
    def test_market_value(self):
        portfolio = Portfolio(cash=100000)
        portfolio.positions["TEST"] = Position(
            code="TEST", shares=1000, avg_cost=10.0, current_price=12.0
        )
        assert portfolio.market_value == 12000
        assert portfolio.total_value == 112000

    def test_pnl(self):
        pos = Position(code="TEST", shares=1000, avg_cost=10.0, current_price=12.0)
        assert pos.pnl == 2000
        assert abs(pos.pnl_pct - 0.2) < 0.001


class TestRiskManager:
    def test_var_calculation(self):
        returns = pd.Series(np.random.normal(0, 0.02, 1000))
        rm = RiskManager()
        var = rm.compute_var(returns, 0.95)
        assert var < 0  # VaR should be negative

    def test_kelly_criterion(self):
        rm = RiskManager()
        kelly = rm.kelly_criterion(win_rate=0.6, avg_win=0.05, avg_loss=0.03)
        assert 0 < kelly <= rm.kelly_fraction

    def test_weight_adjustment(self):
        dates = pd.bdate_range("2023-01-01", periods=100)
        close_panel = pd.DataFrame(
            np.random.uniform(90, 110, (100, 5)),
            index=dates,
            columns=[f"S{i}" for i in range(5)]
        )
        weights = pd.DataFrame(
            np.ones((100, 5)) * 0.3,
            index=dates,
            columns=[f"S{i}" for i in range(5)]
        )

        rm = RiskManager({"max_position_pct": 0.15})
        adjusted = rm.adjust_weights(weights, close_panel)
        assert (adjusted <= 0.15 + 0.001).all().all()

    def test_stop_loss(self):
        rm = RiskManager({"stop_loss_pct": 0.05})
        portfolio = Portfolio(cash=100000)
        portfolio.positions["LOSING"] = Position(
            code="LOSING", shares=1000, avg_cost=100.0, current_price=90.0
        )
        stops = rm.check_stop_loss(portfolio)
        assert "LOSING" in stops


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
