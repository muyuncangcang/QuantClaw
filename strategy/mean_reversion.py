"""
均值回归策略：利用价格偏离均值后的回归特性
适用于电网设备、航空航天等波动相对平稳的板块
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from strategy.base import BaseStrategy, Signal


class MeanReversionStrategy(BaseStrategy):
    """
    Z-Score 均值回归策略：
    1. 计算价格相对于 N 日均线的 Z-Score
    2. Z-Score < -threshold 时买入（超卖回归）
    3. Z-Score > 0 时平仓
    4. 加入布林带宽度过滤，避免在震荡收窄时误判
    """

    def __init__(self, params: Optional[Dict] = None):
        default_params = {
            "window": settings.strategy.mean_reversion_window,
            "zscore_entry": settings.strategy.mean_reversion_zscore_entry,
            "zscore_exit": settings.strategy.mean_reversion_zscore_exit,
            "zscore_short_entry": 1.5,
            "bb_width_min": 0.03,
            "volume_confirm": True,
            "volume_ratio_threshold": 1.2,
        }
        if params:
            default_params.update(params)
        super().__init__(name="MeanReversion", params=default_params)

    def _compute_zscore(self, close: pd.Series, window: int) -> pd.Series:
        ma = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()
        return (close - ma) / std.replace(0, np.nan)

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        factors: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        close_panel = pd.DataFrame({
            code: df["close"] for code, df in data.items() if "close" in df.columns
        })
        volume_panel = pd.DataFrame({
            code: df["volume"] for code, df in data.items() if "volume" in df.columns
        })
        close_panel = close_panel.dropna(how="all").ffill()

        window = self.params["window"]
        entry = self.params["zscore_entry"]
        exit_z = self.params["zscore_exit"]
        short_entry = self.params["zscore_short_entry"]
        bb_min = self.params["bb_width_min"]

        signal_matrix = pd.DataFrame(
            Signal.HOLD.value, index=close_panel.index, columns=close_panel.columns
        )

        for code in close_panel.columns:
            series = close_panel[code].dropna()
            if len(series) < window + 5:
                continue

            zscore = self._compute_zscore(series, window)
            ma = series.rolling(window).mean()
            std = series.rolling(window).std()
            bb_width = (2 * std) / ma

            vol_confirm = True
            if self.params["volume_confirm"] and code in volume_panel.columns:
                vol_ratio = volume_panel[code] / volume_panel[code].rolling(window).mean()
            else:
                vol_ratio = pd.Series(2.0, index=series.index)

            in_position = False
            for date in series.index:
                if pd.isna(zscore.get(date)) or pd.isna(bb_width.get(date)):
                    continue

                z = zscore[date]
                bw = bb_width[date]
                vr = vol_ratio.get(date, 2.0)

                if not in_position:
                    if z < entry and bw > bb_min and vr > self.params["volume_ratio_threshold"]:
                        signal_matrix.loc[date, code] = Signal.BUY.value
                        in_position = True
                    elif z > short_entry:
                        signal_matrix.loc[date, code] = Signal.SELL.value
                else:
                    if z >= exit_z:
                        signal_matrix.loc[date, code] = Signal.SELL.value
                        in_position = False
                    else:
                        signal_matrix.loc[date, code] = Signal.BUY.value

        buy_count = int((signal_matrix == Signal.BUY.value).sum().sum())
        sell_count = int((signal_matrix == Signal.SELL.value).sum().sum())
        logger.info(f"[MeanReversion] Signals: BUY={buy_count}, SELL={sell_count}")
        return signal_matrix

    def compute_target_weights(
        self,
        signal_matrix: pd.DataFrame,
        close_panel: pd.DataFrame,
    ) -> pd.DataFrame:
        """等权分配给所有买入信号的标的"""
        weights = pd.DataFrame(0.0, index=signal_matrix.index, columns=signal_matrix.columns)
        for date in signal_matrix.index:
            buy_mask = signal_matrix.loc[date] == Signal.BUY.value
            n_buy = buy_mask.sum()
            if n_buy > 0:
                weights.loc[date, buy_mask] = 1.0 / n_buy
        return weights
