"""
策略模块单元测试
"""
import numpy as np
import pandas as pd
import pytest

from strategy.base import Signal
from strategy.momentum import MomentumStrategy
from strategy.mean_reversion import MeanReversionStrategy
from strategy.multi_factor import MultiFactorStrategy
from data.processor import FeatureEngineer


def _generate_mock_data(n_stocks: int = 5, n_days: int = 200) -> dict:
    """生成模拟行情数据"""
    dates = pd.bdate_range("2023-01-01", periods=n_days, freq="B")
    data = {}
    np.random.seed(42)
    for i in range(n_stocks):
        code = f"TEST_{i:03d}"
        price = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n_days)))
        volume = np.random.randint(1000, 100000, n_days).astype(float)
        turnover = np.random.uniform(0.5, 5.0, n_days)
        df = pd.DataFrame({
            "open": price * (1 + np.random.uniform(-0.01, 0.01, n_days)),
            "high": price * (1 + np.random.uniform(0, 0.03, n_days)),
            "low": price * (1 - np.random.uniform(0, 0.03, n_days)),
            "close": price,
            "volume": volume,
            "amount": price * volume,
            "turnover": turnover,
            "code": code,
        }, index=dates)
        data[code] = df
    return data


class TestMomentumStrategy:
    def test_signal_generation(self):
        data = _generate_mock_data()
        strategy = MomentumStrategy({"lookback": 20, "top_pct": 0.4})
        signals = strategy.generate_signals(data)

        assert not signals.empty
        assert set(signals.columns) == set(data.keys())
        assert signals.shape[0] > 0

    def test_signal_values(self):
        data = _generate_mock_data()
        strategy = MomentumStrategy()
        signals = strategy.generate_signals(data)

        valid_values = {Signal.BUY.value, Signal.SELL.value, Signal.HOLD.value,
                        Signal.STRONG_BUY.value, Signal.STRONG_SELL.value}
        for col in signals.columns:
            assert set(signals[col].unique()).issubset(valid_values)

    def test_weights_sum_to_one(self):
        data = _generate_mock_data()
        strategy = MomentumStrategy()
        signals = strategy.generate_signals(data)
        close_panel = pd.DataFrame({
            code: df["close"] for code, df in data.items()
        })
        weights = strategy.compute_target_weights(signals, close_panel)

        for date in weights.index:
            row_sum = weights.loc[date].sum()
            assert row_sum <= 1.001 or row_sum == 0


class TestMeanReversionStrategy:
    def test_signal_generation(self):
        data = _generate_mock_data()
        strategy = MeanReversionStrategy()
        signals = strategy.generate_signals(data)

        assert not signals.empty

    def test_zscore_based_entry(self):
        dates = pd.bdate_range("2023-01-01", periods=100, freq="B")
        price = np.concatenate([
            np.linspace(100, 80, 50),
            np.linspace(80, 100, 50),
        ])
        data = {
            "TEST": pd.DataFrame({
                "open": price, "high": price * 1.01, "low": price * 0.99,
                "close": price, "volume": np.ones(100) * 10000,
                "amount": price * 10000, "turnover": np.ones(100) * 2.0,
            }, index=dates)
        }

        strategy = MeanReversionStrategy({
            "window": 20,
            "zscore_entry": -1.5,
            "zscore_exit": 0.0,
            "volume_confirm": False,
        })
        signals = strategy.generate_signals(data)
        assert (signals == Signal.BUY.value).any().any()


class TestMultiFactorStrategy:
    def test_factor_scoring(self):
        data = _generate_mock_data(n_stocks=10)
        strategy = MultiFactorStrategy()
        signals = strategy.generate_signals(data)

        assert not signals.empty
        buy_count = (signals == Signal.BUY.value).sum().sum()
        assert buy_count > 0


class TestFeatureEngineer:
    def test_rsi(self):
        prices = pd.Series(np.random.uniform(90, 110, 100))
        fe = FeatureEngineer()
        rsi = fe.rsi(prices, 14)
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    def test_macd(self):
        prices = pd.Series(np.random.uniform(90, 110, 100))
        fe = FeatureEngineer()
        macd_df = fe.macd(prices)
        assert "dif" in macd_df.columns
        assert "dea" in macd_df.columns
        assert "macd_hist" in macd_df.columns

    def test_bollinger_bands(self):
        prices = pd.Series(np.random.uniform(90, 110, 100))
        fe = FeatureEngineer()
        bb = fe.bollinger_bands(prices)
        valid = bb.dropna()
        assert (valid["bb_upper"] >= valid["bb_mid"]).all()
        assert (valid["bb_mid"] >= valid["bb_lower"]).all()

    def test_factor_matrix(self):
        dates = pd.bdate_range("2023-01-01", periods=100, freq="B")
        df = pd.DataFrame({
            "open": np.random.uniform(95, 105, 100),
            "high": np.random.uniform(100, 110, 100),
            "low": np.random.uniform(90, 100, 100),
            "close": np.random.uniform(95, 105, 100),
            "volume": np.random.randint(1000, 100000, 100).astype(float),
            "turnover": np.random.uniform(0.5, 5.0, 100),
        }, index=dates)

        fe = FeatureEngineer()
        factors = fe.build_factor_matrix(df)
        assert "mom_20" in factors.columns
        assert "vol_20" in factors.columns
        assert "rsi_14" in factors.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
