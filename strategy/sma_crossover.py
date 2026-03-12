"""
SMA 均线交叉策略：基于 AKShare/Backtrader 经典模式
当短期均线上穿长期均线时买入，下穿时卖出
对基金 NAV 数据特别有效——净值曲线平滑，均线信号噪音小

额外改进：
  - 双均线交叉（短 SMA vs 长 SMA），比单均线更稳健
  - 交叉确认机制：连续 confirm_bars 天维持交叉才出信号，减少假突破
  - 趋势强度过滤：短 SMA 与长 SMA 之间的偏离度作为信号强度
  - 止盈止损：集成百分比止盈止损判定
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from strategy.base import BaseStrategy, Signal


class SMACrossoverStrategy(BaseStrategy):

    def __init__(self, params: Optional[Dict] = None):
        default_params = {
            "short_window": 5,
            "long_window": 20,
            "confirm_bars": 2,
            "stop_profit_pct": 0.08,
            "stop_loss_pct": 0.05,
            "min_deviation": 0.005,
        }
        if params:
            default_params.update(params)
        super().__init__(name="SMACrossover", params=default_params)

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        factors: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        close_panel = pd.DataFrame({
            code: df["close"] for code, df in data.items() if "close" in df.columns
        })
        close_panel = close_panel.dropna(how="all").ffill()

        short_w = self.params["short_window"]
        long_w = self.params["long_window"]
        confirm = self.params["confirm_bars"]
        min_dev = self.params["min_deviation"]
        stop_profit = self.params["stop_profit_pct"]
        stop_loss = self.params["stop_loss_pct"]

        sma_short = close_panel.rolling(short_w, min_periods=short_w).mean()
        sma_long = close_panel.rolling(long_w, min_periods=long_w).mean()

        cross_up = (sma_short > sma_long).astype(int)
        cross_dn = (sma_short < sma_long).astype(int)

        cross_up_confirm = cross_up.rolling(confirm, min_periods=confirm).min()
        cross_dn_confirm = cross_dn.rolling(confirm, min_periods=confirm).min()

        deviation = (sma_short - sma_long) / sma_long.replace(0, np.nan)

        signal_matrix = pd.DataFrame(
            Signal.HOLD.value, index=close_panel.index, columns=close_panel.columns,
        )

        entry_prices: Dict[str, float] = {}

        for i, date in enumerate(close_panel.index):
            for code in close_panel.columns:
                cu = cross_up_confirm.loc[date, code] if not pd.isna(cross_up_confirm.loc[date, code]) else 0
                cd = cross_dn_confirm.loc[date, code] if not pd.isna(cross_dn_confirm.loc[date, code]) else 0
                dev = deviation.loc[date, code] if not pd.isna(deviation.loc[date, code]) else 0
                price = close_panel.loc[date, code]

                if pd.isna(price):
                    continue

                if code in entry_prices:
                    entry_p = entry_prices[code]
                    pnl_pct = (price - entry_p) / entry_p
                    if pnl_pct >= stop_profit:
                        signal_matrix.loc[date, code] = Signal.SELL.value
                        del entry_prices[code]
                        continue
                    if pnl_pct <= -stop_loss:
                        signal_matrix.loc[date, code] = Signal.SELL.value
                        del entry_prices[code]
                        continue

                if cu == 1 and abs(dev) >= min_dev:
                    if dev > min_dev * 3:
                        signal_matrix.loc[date, code] = Signal.STRONG_BUY.value
                    else:
                        signal_matrix.loc[date, code] = Signal.BUY.value
                    if code not in entry_prices:
                        entry_prices[code] = price
                elif cd == 1:
                    if dev < -min_dev * 3:
                        signal_matrix.loc[date, code] = Signal.STRONG_SELL.value
                    else:
                        signal_matrix.loc[date, code] = Signal.SELL.value
                    entry_prices.pop(code, None)

        buy_cnt = int((signal_matrix >= Signal.BUY.value).sum().sum())
        sell_cnt = int((signal_matrix <= Signal.SELL.value).sum().sum())
        logger.info(
            f"[SMACrossover] Generated signals: BUY/STRONG_BUY={buy_cnt}, "
            f"SELL/STRONG_SELL={sell_cnt}"
        )
        return signal_matrix

    def compute_target_weights(
        self,
        signal_matrix: pd.DataFrame,
        close_panel: pd.DataFrame,
    ) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=signal_matrix.index, columns=signal_matrix.columns)

        short_w = self.params["short_window"]
        long_w = self.params["long_window"]
        sma_short = close_panel.rolling(short_w, min_periods=short_w).mean()
        sma_long = close_panel.rolling(long_w, min_periods=long_w).mean()
        deviation = ((sma_short - sma_long) / sma_long.replace(0, np.nan)).abs()

        for date in signal_matrix.index:
            buy_mask = signal_matrix.loc[date] >= Signal.BUY.value
            if buy_mask.sum() == 0:
                continue
            dev_scores = deviation.loc[date][buy_mask].dropna().clip(lower=0.001)
            total = dev_scores.sum()
            if total > 0:
                weights.loc[date, dev_scores.index] = dev_scores / total

        return weights
