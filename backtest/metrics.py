"""
绩效评估模块：计算回测的各类绩效指标
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


class PerformanceMetrics:
    """量化策略绩效指标计算"""

    @staticmethod
    def compute(
        nav_df: pd.DataFrame,
        initial_capital: float = 1_000_000.0,
        risk_free_rate: float = 0.025,
        benchmark_col: Optional[str] = None,
    ) -> Dict[str, float]:
        returns = nav_df["returns"].dropna()
        if returns.empty:
            return {}

        total_days = len(returns)
        annual_factor = 252

        total_return = nav_df["nav"].iloc[-1] - 1
        annual_return = (1 + total_return) ** (annual_factor / total_days) - 1

        vol = returns.std() * np.sqrt(annual_factor)
        daily_rf = risk_free_rate / annual_factor
        excess_ret = returns - daily_rf
        sharpe = excess_ret.mean() / returns.std() * np.sqrt(annual_factor) if returns.std() > 0 else 0

        downside_ret = returns[returns < 0]
        downside_std = downside_ret.std() * np.sqrt(annual_factor) if len(downside_ret) > 0 else 1e-10
        sortino = (annual_return - risk_free_rate) / downside_std

        cummax = nav_df["nav"].cummax()
        drawdown = (nav_df["nav"] - cummax) / cummax
        max_drawdown = abs(drawdown.min())

        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0

        dd_end = drawdown.idxmin()
        dd_start = nav_df["nav"][:dd_end].idxmax() if dd_end == dd_end else dd_end
        max_dd_duration = (dd_end - dd_start).days if dd_start != dd_end else 0

        positive_days = (returns > 0).sum()
        win_rate = positive_days / total_days if total_days > 0 else 0

        avg_gain = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1e-10
        profit_loss_ratio = avg_gain / avg_loss

        skew = returns.skew()
        kurtosis = returns.kurtosis()

        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else var_95

        metrics = {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "max_dd_duration_days": max_dd_duration,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "skewness": skew,
            "kurtosis": kurtosis,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "total_trades": 0,
            "annual_turnover": 0.0,
        }

        if benchmark_col and benchmark_col in nav_df.columns:
            bench_returns = nav_df.get("benchmark_returns", pd.Series(dtype=float)).dropna()
            if not bench_returns.empty:
                excess = returns.reindex(bench_returns.index) - bench_returns
                tracking_error = excess.std() * np.sqrt(annual_factor)
                info_ratio = excess.mean() / excess.std() * np.sqrt(annual_factor) if excess.std() > 0 else 0
                alpha, beta = PerformanceMetrics._compute_alpha_beta(
                    returns, bench_returns, risk_free_rate
                )
                metrics.update({
                    "alpha": alpha,
                    "beta": beta,
                    "tracking_error": tracking_error,
                    "information_ratio": info_ratio,
                })

        return metrics

    @staticmethod
    def _compute_alpha_beta(
        returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.025,
    ) -> tuple:
        common = returns.index.intersection(benchmark_returns.index)
        if len(common) < 10:
            return 0.0, 1.0
        r = returns[common].values
        b = benchmark_returns[common].values
        rf_daily = risk_free_rate / 252

        cov = np.cov(r - rf_daily, b - rf_daily)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0
        alpha = (np.mean(r) - rf_daily) - beta * (np.mean(b) - rf_daily)
        alpha_annual = alpha * 252
        return alpha_annual, beta

    @staticmethod
    def rolling_sharpe(
        returns: pd.Series, window: int = 60, risk_free_rate: float = 0.025
    ) -> pd.Series:
        daily_rf = risk_free_rate / 252
        excess = returns - daily_rf
        rolling_mean = excess.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        return (rolling_mean / rolling_std) * np.sqrt(252)

    @staticmethod
    def rolling_max_drawdown(nav: pd.Series, window: int = 60) -> pd.Series:
        def _max_dd(x):
            peak = x.cummax()
            dd = (x - peak) / peak
            return dd.min()
        return nav.rolling(window).apply(_max_dd, raw=False)
