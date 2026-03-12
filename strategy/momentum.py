"""
动量策略：追踪过去 N 日涨幅最大的标的，捕获趋势延续效应
适用于 CPO、半导体等高 beta 板块
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from strategy.base import BaseStrategy, Signal


class MomentumStrategy(BaseStrategy):
    """
    截面动量策略：
    1. 计算过去 lookback 日收益率
    2. 按动量排名，选择排名前 top_pct 的标的
    3. 持有 holding_period 日后轮换
    4. 加入动量衰减过滤和波动率调整
    """

    def __init__(self, params: Optional[Dict] = None):
        default_params = {
            "lookback": settings.strategy.momentum_window,
            "holding_period": settings.strategy.momentum_holding_period,
            "top_pct": 0.3,
            "vol_adjust": True,
            "vol_window": 20,
            "min_momentum": 0.02,
        }
        if params:
            default_params.update(params)
        super().__init__(name="Momentum", params=default_params)

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        factors: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        close_panel = pd.DataFrame({
            code: df["close"] for code, df in data.items() if "close" in df.columns
        })
        close_panel = close_panel.dropna(how="all").ffill()

        lookback = self.params["lookback"]
        momentum = close_panel.pct_change(periods=lookback)

        if self.params["vol_adjust"]:
            vol = close_panel.pct_change().rolling(
                self.params["vol_window"]
            ).std() * np.sqrt(252)
            momentum = momentum / vol.replace(0, np.nan)

        threshold_rank = 1 - self.params["top_pct"]
        min_mom = self.params["min_momentum"]

        signal_matrix = pd.DataFrame(
            Signal.HOLD.value, index=momentum.index, columns=momentum.columns
        )

        for date in momentum.index:
            row = momentum.loc[date].dropna()
            if len(row) < 3:
                continue

            rank_pct = row.rank(pct=True)
            raw_mom = close_panel.pct_change(periods=lookback).loc[date].reindex(row.index)

            for code in row.index:
                if rank_pct[code] >= threshold_rank and raw_mom.get(code, 0) > min_mom:
                    signal_matrix.loc[date, code] = Signal.BUY.value
                elif rank_pct[code] < 0.2:
                    signal_matrix.loc[date, code] = Signal.SELL.value

        holding = self.params["holding_period"]
        signal_matrix = self._apply_holding_period(signal_matrix, holding)

        logger.info(
            f"[Momentum] Generated signals: "
            f"BUY={int((signal_matrix == Signal.BUY.value).sum().sum())}, "
            f"SELL={int((signal_matrix == Signal.SELL.value).sum().sum())}"
        )
        return signal_matrix

    def compute_target_weights(
        self,
        signal_matrix: pd.DataFrame,
        close_panel: pd.DataFrame,
    ) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=signal_matrix.index, columns=signal_matrix.columns)
        lookback = self.params["lookback"]
        momentum = close_panel.pct_change(periods=lookback)

        for date in signal_matrix.index:
            buy_mask = signal_matrix.loc[date] == Signal.BUY.value
            if buy_mask.sum() == 0:
                continue
            mom_scores = momentum.loc[date][buy_mask].dropna()
            if mom_scores.empty:
                continue
            mom_scores = mom_scores.clip(lower=0)
            total = mom_scores.sum()
            if total > 0:
                weights.loc[date, mom_scores.index] = mom_scores / total

        return weights

    @staticmethod
    def _apply_holding_period(
        signals: pd.DataFrame, period: int
    ) -> pd.DataFrame:
        """确保买入信号持续 period 天"""
        result = signals.copy()
        for col in result.columns:
            buy_dates = result.index[result[col] == Signal.BUY.value]
            for bd in buy_dates:
                loc = result.index.get_loc(bd)
                end_loc = min(loc + period, len(result.index))
                for i in range(loc + 1, end_loc):
                    if result.iloc[i][col] != Signal.SELL.value:
                        result.iloc[i, result.columns.get_loc(col)] = Signal.BUY.value
        return result
