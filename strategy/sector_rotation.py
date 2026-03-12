"""
板块轮动策略：在 CPO、半导体、存储芯片、航空航天、电网设备五大板块间动态分配
核心思想：配置动量最强的 top_n 板块，同时考虑波动率和相关性
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import SECTOR_UNIVERSE, Sector, settings
from strategy.base import BaseStrategy, Signal


class SectorRotationStrategy(BaseStrategy):
    """
    板块轮动策略：
    1. 计算每个板块的综合得分（动量 + 反向波动率 + 资金流向）
    2. 选择得分最高的 top_n 板块
    3. 对选中板块内的标的等权或动量加权
    4. 固定周期（如每月）轮换
    """

    def __init__(self, params: Optional[Dict] = None):
        default_params = {
            "lookback": settings.strategy.rotation_lookback,
            "top_n": settings.strategy.rotation_top_n,
            "rebalance_freq": 20,       # 交易日
            "momentum_weight": 0.5,
            "inv_vol_weight": 0.3,
            "turnover_weight": 0.2,
            "min_sector_stocks": 3,
        }
        if params:
            default_params.update(params)
        super().__init__(name="SectorRotation", params=default_params)

    def _compute_sector_scores(
        self,
        sector_close_panels: Dict[Sector, pd.DataFrame],
        date: pd.Timestamp,
    ) -> Dict[Sector, float]:
        """计算各板块在给定日期的综合得分"""
        lookback = self.params["lookback"]
        scores = {}

        for sector, panel in sector_close_panels.items():
            if date not in panel.index:
                continue
            loc = panel.index.get_loc(date)
            if loc < lookback:
                continue

            window = panel.iloc[loc - lookback: loc + 1]
            sector_return = window.iloc[-1] / window.iloc[0] - 1
            avg_return = sector_return.mean()

            daily_ret = panel.iloc[loc - lookback: loc + 1].pct_change().dropna()
            avg_vol = daily_ret.std().mean() * np.sqrt(252)
            inv_vol = 1.0 / max(avg_vol, 0.01)

            score = (
                self.params["momentum_weight"] * avg_return
                + self.params["inv_vol_weight"] * inv_vol * 0.1
            )
            scores[sector] = score

        return scores

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        factors: Optional[Dict[str, pd.DataFrame]] = None,
        sector_mapping: Optional[Dict[Sector, List[str]]] = None,
    ) -> pd.DataFrame:
        """
        data: 所有标的的 DataFrame dict
        sector_mapping: 板块 -> 标的代码列表，如果不提供则从 SECTOR_UNIVERSE 推断
        """
        if sector_mapping is None:
            sector_mapping = {}
            for sector, cfg in SECTOR_UNIVERSE.items():
                codes = [c for c in cfg.stocks.keys() if c in data]
                codes += [f"ETF_{c}" for c in cfg.etfs.keys() if f"ETF_{c}" in data]
                codes += [f"FUND_{c}" for c in cfg.funds.keys() if f"FUND_{c}" in data]
                if codes:
                    sector_mapping[sector] = codes

        all_codes = list(data.keys())
        close_panel = pd.DataFrame({
            code: data[code]["close"] for code in all_codes if "close" in data[code].columns
        }).dropna(how="all").ffill()

        sector_panels = {}
        for sector, codes in sector_mapping.items():
            valid_codes = [c for c in codes if c in close_panel.columns]
            if valid_codes:
                sector_panels[sector] = close_panel[valid_codes]

        signal_matrix = pd.DataFrame(
            Signal.HOLD.value, index=close_panel.index, columns=close_panel.columns
        )

        rebal_freq = self.params["rebalance_freq"]
        top_n = self.params["top_n"]
        rebal_dates = close_panel.index[::rebal_freq]

        active_sectors: List[Sector] = []

        for i, rebal_date in enumerate(rebal_dates):
            scores = self._compute_sector_scores(sector_panels, rebal_date)
            if not scores:
                continue

            sorted_sectors = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            active_sectors = [s[0] for s in sorted_sectors[:top_n]]

            loc = close_panel.index.get_loc(rebal_date)
            next_loc = min(loc + rebal_freq, len(close_panel.index))
            period_dates = close_panel.index[loc:next_loc]

            active_codes = set()
            for sector in active_sectors:
                if sector in sector_mapping:
                    active_codes.update(
                        c for c in sector_mapping[sector] if c in close_panel.columns
                    )

            for date in period_dates:
                for code in close_panel.columns:
                    if code in active_codes:
                        signal_matrix.loc[date, code] = Signal.BUY.value
                    else:
                        signal_matrix.loc[date, code] = Signal.HOLD.value

            logger.debug(
                f"[Rotation] {rebal_date.strftime('%Y-%m-%d')}: "
                f"Top sectors={[s.value for s in active_sectors]}"
            )

        return signal_matrix

    def compute_target_weights(
        self,
        signal_matrix: pd.DataFrame,
        close_panel: pd.DataFrame,
    ) -> pd.DataFrame:
        """被选中板块内等权分配"""
        weights = pd.DataFrame(0.0, index=signal_matrix.index, columns=signal_matrix.columns)
        for date in signal_matrix.index:
            buy_mask = signal_matrix.loc[date] == Signal.BUY.value
            n_buy = buy_mask.sum()
            if n_buy > 0:
                weights.loc[date, buy_mask] = 1.0 / n_buy
        return weights
