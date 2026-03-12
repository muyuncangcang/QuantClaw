"""
特征工程模块：技术指标、基本面因子、波动率因子计算
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


class FeatureEngineer:
    """计算量化因子并构建因子矩阵"""

    # ────────────────────── 技术指标 ──────────────────────

    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window).mean()

    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        hist = 2 * (dif - dea)
        return pd.DataFrame({"dif": dif, "dea": dea, "macd_hist": hist})

    @staticmethod
    def bollinger_bands(
        series: pd.Series, window: int = 20, num_std: float = 2.0
    ) -> pd.DataFrame:
        mid = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        return pd.DataFrame({
            "bb_upper": mid + num_std * std,
            "bb_mid": mid,
            "bb_lower": mid - num_std * std,
            "bb_width": (num_std * std * 2) / mid,
        })

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    # ────────────────────── 量化因子 ──────────────────────

    @staticmethod
    def momentum_factor(close: pd.Series, window: int = 20) -> pd.Series:
        """动量因子：过去 N 日收益率"""
        return close.pct_change(periods=window)

    @staticmethod
    def volatility_factor(close: pd.Series, window: int = 20) -> pd.Series:
        """波动率因子：过去 N 日收益率标准差的年化值"""
        returns = close.pct_change()
        return returns.rolling(window=window).std() * np.sqrt(252)

    @staticmethod
    def turnover_factor(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """换手率因子：过去 N 日平均换手率"""
        if "turnover" in df.columns:
            return df["turnover"].rolling(window=window).mean()
        return pd.Series(dtype=float)

    @staticmethod
    def volume_price_factor(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """量价相关性因子：收益率与成交量变化的相关系数"""
        ret = df["close"].pct_change()
        vol_chg = df["volume"].pct_change()
        return ret.rolling(window=window).corr(vol_chg)

    @staticmethod
    def reversal_factor(close: pd.Series, window: int = 5) -> pd.Series:
        """短期反转因子"""
        return -close.pct_change(periods=window)

    @staticmethod
    def relative_strength(
        stock_close: pd.Series,
        index_close: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """相对强度因子：个股收益率 / 基准收益率"""
        stock_ret = stock_close.pct_change(periods=window)
        index_ret = index_close.pct_change(periods=window)
        return stock_ret - index_ret

    # ────────────────────── 批量构建 ──────────────────────

    def build_factor_matrix(
        self,
        df: pd.DataFrame,
        index_close: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """为单只标的构建完整因子矩阵"""
        factors = pd.DataFrame(index=df.index)

        factors["return_1d"] = df["close"].pct_change(1)
        factors["return_5d"] = df["close"].pct_change(5)
        factors["return_20d"] = df["close"].pct_change(20)

        factors["mom_10"] = self.momentum_factor(df["close"], 10)
        factors["mom_20"] = self.momentum_factor(df["close"], 20)
        factors["mom_60"] = self.momentum_factor(df["close"], 60)

        factors["vol_20"] = self.volatility_factor(df["close"], 20)
        factors["vol_60"] = self.volatility_factor(df["close"], 60)

        factors["rsi_14"] = self.rsi(df["close"], 14)
        factors["rsi_6"] = self.rsi(df["close"], 6)

        macd_df = self.macd(df["close"])
        factors = factors.join(macd_df)

        bb_df = self.bollinger_bands(df["close"])
        factors = factors.join(bb_df)

        factors["atr_14"] = self.atr(df, 14)

        factors["reversal_5d"] = self.reversal_factor(df["close"], 5)

        factors["vp_corr"] = self.volume_price_factor(df, 20)

        if "turnover" in df.columns:
            factors["turnover_20"] = self.turnover_factor(df, 20)

        if index_close is not None:
            aligned = index_close.reindex(df.index).ffill()
            factors["rel_strength_20"] = self.relative_strength(
                df["close"], aligned, 20
            )

        factors["sma_5"] = self.sma(df["close"], 5)
        factors["sma_20"] = self.sma(df["close"], 20)
        factors["sma_60"] = self.sma(df["close"], 60)
        factors["golden_cross"] = (
            (factors["sma_5"] > factors["sma_20"]) &
            (factors["sma_5"].shift(1) <= factors["sma_20"].shift(1))
        ).astype(int)
        factors["death_cross"] = (
            (factors["sma_5"] < factors["sma_20"]) &
            (factors["sma_5"].shift(1) >= factors["sma_20"].shift(1))
        ).astype(int)

        return factors

    def build_cross_sectional_factors(
        self,
        close_panel: pd.DataFrame,
        window: int = 20,
    ) -> Dict[str, pd.DataFrame]:
        """构建截面因子：用于板块轮动和多因子选股"""
        returns = close_panel.pct_change(periods=window)
        volatility = close_panel.pct_change().rolling(window).std() * np.sqrt(252)

        mom_rank = returns.rank(axis=1, pct=True)
        vol_rank = volatility.rank(axis=1, pct=True, ascending=True)

        return {
            "returns": returns,
            "volatility": volatility,
            "mom_rank": mom_rank,
            "vol_rank": vol_rank,
        }
