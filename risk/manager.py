"""
风控管理模块：仓位控制、止损、VaR 计算、相关性监控
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings


class RiskManager:
    """
    多层风控体系：
    1. 事前风控：权重约束、集中度限制
    2. 事中风控：止损、VaR 监控、相关性检测
    3. 事后风控：绩效归因、风险报告
    """

    def __init__(self, params: Optional[Dict] = None):
        rp = settings.risk
        self.max_position_pct = params.get("max_position_pct", rp.max_position_pct) if params else rp.max_position_pct
        self.max_sector_pct = params.get("max_sector_pct", rp.max_sector_pct) if params else rp.max_sector_pct
        self.max_drawdown_pct = params.get("max_drawdown_pct", rp.max_drawdown_pct) if params else rp.max_drawdown_pct
        self.stop_loss_pct = params.get("stop_loss_pct", rp.stop_loss_pct) if params else rp.stop_loss_pct
        self.trailing_stop_pct = params.get("trailing_stop_pct", rp.trailing_stop_pct) if params else rp.trailing_stop_pct
        self.var_confidence = params.get("var_confidence", rp.var_confidence) if params else rp.var_confidence
        self.max_correlation = params.get("max_correlation", rp.max_correlation) if params else rp.max_correlation
        self.kelly_fraction = params.get("kelly_fraction", rp.kelly_fraction) if params else rp.kelly_fraction

        self._peak_values: Dict[str, float] = {}

    def adjust_weights(
        self,
        target_weights: pd.DataFrame,
        close_panel: pd.DataFrame,
    ) -> pd.DataFrame:
        """事前风控：调整目标权重使其满足约束"""
        adjusted = target_weights.copy()

        adjusted = adjusted.clip(upper=self.max_position_pct)

        for date in adjusted.index:
            row = adjusted.loc[date]
            total = row.sum()
            if total > 1.0:
                adjusted.loc[date] = row / total

        adjusted = self._correlation_filter(adjusted, close_panel)

        return adjusted

    def _correlation_filter(
        self,
        weights: pd.DataFrame,
        close_panel: pd.DataFrame,
        lookback: int = 60,
    ) -> pd.DataFrame:
        """过滤掉相关性过高的持仓，保留收益更好的"""
        adjusted = weights.copy()

        for date in adjusted.index[lookback:]:
            active = adjusted.loc[date]
            active_codes = active[active > 0].index.tolist()
            if len(active_codes) < 2:
                continue

            loc = close_panel.index.get_loc(date)
            hist = close_panel[active_codes].iloc[max(0, loc - lookback): loc]
            if hist.empty:
                continue

            corr_matrix = hist.pct_change().dropna().corr()
            returns = hist.iloc[-1] / hist.iloc[0] - 1

            to_remove = set()
            for i, code_a in enumerate(active_codes):
                for code_b in active_codes[i + 1:]:
                    if code_a in corr_matrix.columns and code_b in corr_matrix.columns:
                        if abs(corr_matrix.loc[code_a, code_b]) > self.max_correlation:
                            worse = code_a if returns.get(code_a, 0) < returns.get(code_b, 0) else code_b
                            to_remove.add(worse)

            for code in to_remove:
                adjusted.loc[date, code] = 0

            row = adjusted.loc[date]
            total = row.sum()
            if total > 0:
                adjusted.loc[date] = row / total

        return adjusted

    def check_stop_loss(self, portfolio) -> List[str]:
        """检查持仓是否触发止损"""
        to_close = []
        for code, pos in portfolio.positions.items():
            if pos.pnl_pct < -self.stop_loss_pct:
                logger.warning(
                    f"[RiskManager] Stop-loss triggered: {code} "
                    f"PnL={pos.pnl_pct:.2%} < -{self.stop_loss_pct:.2%}"
                )
                to_close.append(code)
                continue

            peak = self._peak_values.get(code, pos.current_price)
            if pos.current_price > peak:
                self._peak_values[code] = pos.current_price
                peak = pos.current_price

            drawdown_from_peak = (peak - pos.current_price) / peak if peak > 0 else 0
            if drawdown_from_peak > self.trailing_stop_pct and pos.pnl_pct > 0:
                logger.warning(
                    f"[RiskManager] Trailing stop triggered: {code} "
                    f"DD from peak={drawdown_from_peak:.2%}"
                )
                to_close.append(code)

        return to_close

    def compute_var(
        self,
        returns: pd.Series,
        confidence: Optional[float] = None,
        method: str = "historical",
    ) -> float:
        """
        计算 Value at Risk
        method: 'historical' | 'parametric' | 'cornish_fisher'
        """
        conf = confidence or self.var_confidence
        alpha = 1 - conf

        if method == "historical":
            return float(np.percentile(returns.dropna(), alpha * 100))

        elif method == "parametric":
            from scipy import stats
            z = stats.norm.ppf(alpha)
            return float(returns.mean() + z * returns.std())

        elif method == "cornish_fisher":
            from scipy import stats
            z = stats.norm.ppf(alpha)
            s = returns.skew()
            k = returns.kurtosis()
            z_cf = (
                z
                + (z**2 - 1) * s / 6
                + (z**3 - 3 * z) * k / 24
                - (2 * z**3 - 5 * z) * s**2 / 36
            )
            return float(returns.mean() + z_cf * returns.std())

        return 0.0

    def compute_cvar(
        self,
        returns: pd.Series,
        confidence: Optional[float] = None,
    ) -> float:
        """Conditional VaR (Expected Shortfall)"""
        var = self.compute_var(returns, confidence)
        tail = returns[returns <= var]
        return float(tail.mean()) if len(tail) > 0 else var

    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """凯利公式计算最优仓位"""
        if avg_loss == 0:
            return 0.0
        b = avg_win / avg_loss
        f = (win_rate * (b + 1) - 1) / b
        return max(0.0, min(self.kelly_fraction, f * self.kelly_fraction))

    def risk_report(
        self,
        returns: pd.Series,
        positions: Dict[str, float],
    ) -> Dict[str, float]:
        """生成风险报告"""
        return {
            "var_95_hist": self.compute_var(returns, 0.95, "historical"),
            "var_95_param": self.compute_var(returns, 0.95, "parametric"),
            "var_95_cf": self.compute_var(returns, 0.95, "cornish_fisher"),
            "cvar_95": self.compute_cvar(returns, 0.95),
            "max_position": max(positions.values()) if positions else 0.0,
            "n_positions": len(positions),
            "hhi": sum(v**2 for v in positions.values()) if positions else 0.0,
        }
