"""
多因子选股策略：综合动量、波动率、量价、反转、换手率等因子
采用截面打分排序，选出综合得分最高的标的
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from data.processor import FeatureEngineer
from strategy.base import BaseStrategy, Signal


class MultiFactorStrategy(BaseStrategy):
    """
    多因子选股模型：
    1. 对每个因子做截面 Z-Score 标准化
    2. 用预设权重加权合成综合得分
    3. 选择得分 top_pct 的标的做多
    4. IC/IR 滚动监控因子有效性
    """

    def __init__(self, params: Optional[Dict] = None):
        default_params = {
            "factor_weights": settings.strategy.multi_factor_weights,
            "top_pct": 0.3,
            "rebalance_freq": 10,
            "lookback": 20,
            "ic_window": 60,       # IC 滚动窗口
            "min_ic": 0.02,        # 最低 IC 阈值
            "dynamic_weight": True,
        }
        if params:
            default_params.update(params)
        super().__init__(name="MultiFactor", params=default_params)
        self.fe = FeatureEngineer()
        self._ic_history: Dict[str, List[float]] = {}

    def _compute_factor_scores(
        self,
        data: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
    ) -> Optional[pd.Series]:
        """计算截面因子合成得分"""
        lookback = self.params["lookback"]
        factor_dict: Dict[str, Dict[str, float]] = {}

        for code, df in data.items():
            if date not in df.index:
                continue
            loc = df.index.get_loc(date)
            if loc < lookback + 10:
                continue

            close = df["close"]
            factor_dict.setdefault("momentum", {})[code] = (
                close.iloc[loc] / close.iloc[max(0, loc - lookback)] - 1
            )

            ret = close.pct_change().iloc[max(0, loc - lookback): loc + 1]
            factor_dict.setdefault("volatility", {})[code] = -ret.std() * np.sqrt(252)

            if "volume" in df.columns:
                vol_chg = df["volume"].pct_change()
                ret_series = close.pct_change()
                window = pd.DataFrame({
                    "ret": ret_series, "vol_chg": vol_chg
                }).iloc[max(0, loc - lookback): loc + 1].dropna()
                if len(window) > 5:
                    corr = window["ret"].corr(window["vol_chg"])
                    factor_dict.setdefault("quality", {})[code] = -abs(corr)

            factor_dict.setdefault("reversal", {})[code] = -(
                close.iloc[loc] / close.iloc[max(0, loc - 5)] - 1
            )

            if "turnover" in df.columns and not pd.isna(df["turnover"].iloc[loc]):
                avg_turn = df["turnover"].iloc[max(0, loc - lookback): loc + 1].mean()
                factor_dict.setdefault("value", {})[code] = -avg_turn

        if not factor_dict:
            return None

        factor_df = pd.DataFrame(factor_dict)

        zscore_df = factor_df.apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x)

        weights = self.params["factor_weights"]
        composite = pd.Series(0.0, index=zscore_df.index)
        for factor_name, weight in weights.items():
            if factor_name in zscore_df.columns:
                col = zscore_df[factor_name].fillna(0)
                composite += weight * col

        return composite

    def _compute_ic(
        self,
        data: Dict[str, pd.DataFrame],
        dates: pd.DatetimeIndex,
    ) -> Dict[str, float]:
        """计算因子 IC（信息系数）：因子值与下期收益的秩相关"""
        ic_values: Dict[str, List[float]] = {
            name: [] for name in self.params["factor_weights"]
        }

        for i in range(1, min(len(dates), self.params["ic_window"])):
            date = dates[-i - 1]
            next_date = dates[-i]

            factor_scores = self._compute_factor_scores(data, date)
            if factor_scores is None:
                continue

            next_returns = {}
            for code in factor_scores.index:
                if code in data and next_date in data[code].index and date in data[code].index:
                    next_returns[code] = (
                        data[code].loc[next_date, "close"]
                        / data[code].loc[date, "close"]
                        - 1
                    )

            if len(next_returns) < 5:
                continue

            ret_series = pd.Series(next_returns)
            common = factor_scores.index.intersection(ret_series.index)
            if len(common) < 5:
                continue

            rank_corr = factor_scores[common].corr(ret_series[common], method="spearman")
            if not np.isnan(rank_corr):
                for name in ic_values:
                    ic_values[name].append(rank_corr)

        avg_ic = {}
        for name, values in ic_values.items():
            if values:
                avg_ic[name] = np.mean(values)
        return avg_ic

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        factors: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        all_codes = list(data.keys())
        close_panel = pd.DataFrame({
            code: data[code]["close"] for code in all_codes if "close" in data[code].columns
        }).dropna(how="all").ffill()

        signal_matrix = pd.DataFrame(
            Signal.HOLD.value, index=close_panel.index, columns=close_panel.columns
        )

        rebal_freq = self.params["rebalance_freq"]
        top_pct = self.params["top_pct"]
        rebal_dates = close_panel.index[self.params["lookback"] + 10::rebal_freq]

        if self.params["dynamic_weight"] and len(rebal_dates) > 0:
            ic = self._compute_ic(data, close_panel.index)
            if ic:
                total_ic = sum(abs(v) for v in ic.values())
                if total_ic > 0:
                    new_weights = {k: abs(v) / total_ic for k, v in ic.items()}
                    logger.info(f"[MultiFactor] Dynamic weights from IC: {new_weights}")
                    self.params["factor_weights"].update(new_weights)

        for rebal_date in rebal_dates:
            scores = self._compute_factor_scores(data, rebal_date)
            if scores is None or scores.empty:
                continue

            threshold = scores.quantile(1 - top_pct)
            selected = scores[scores >= threshold].index.tolist()
            rejected = scores[scores < scores.quantile(0.2)].index.tolist()

            loc = close_panel.index.get_loc(rebal_date)
            next_loc = min(loc + rebal_freq, len(close_panel.index))
            period_dates = close_panel.index[loc:next_loc]

            for date in period_dates:
                for code in selected:
                    if code in signal_matrix.columns:
                        signal_matrix.loc[date, code] = Signal.BUY.value
                for code in rejected:
                    if code in signal_matrix.columns:
                        signal_matrix.loc[date, code] = Signal.SELL.value

        buy_count = int((signal_matrix == Signal.BUY.value).sum().sum())
        logger.info(f"[MultiFactor] Generated {buy_count} BUY signals")
        return signal_matrix

    def compute_target_weights(
        self,
        signal_matrix: pd.DataFrame,
        close_panel: pd.DataFrame,
    ) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=signal_matrix.index, columns=signal_matrix.columns)
        for date in signal_matrix.index:
            buy_mask = signal_matrix.loc[date] == Signal.BUY.value
            n_buy = buy_mask.sum()
            if n_buy > 0:
                weights.loc[date, buy_mask] = 1.0 / n_buy
        return weights

    def get_factor_exposure_report(
        self,
        data: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
    ) -> pd.DataFrame:
        """生成因子暴露报告，用于风险归因"""
        scores = self._compute_factor_scores(data, date)
        if scores is None:
            return pd.DataFrame()
        return pd.DataFrame({"composite_score": scores}).sort_values(
            "composite_score", ascending=False
        )
