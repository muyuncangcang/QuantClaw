"""
策略权重动态调整器
根据各策略近期回测表现（夏普比率/胜率）动态分配信号权重
近期表现好的策略获得更高权重，表现差的策略权重降低
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Type

import numpy as np
import pandas as pd
from loguru import logger

from backtest.engine import BacktestEngine, BacktestResult
from backtest.metrics import PerformanceMetrics
from strategy.base import BaseStrategy
from strategy.momentum import MomentumStrategy
from strategy.mean_reversion import MeanReversionStrategy
from strategy.multi_factor import MultiFactorStrategy
from strategy.sector_rotation import SectorRotationStrategy
from strategy.sma_crossover import SMACrossoverStrategy


STRATEGY_CLASSES: Dict[str, Type[BaseStrategy]] = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "multi_factor": MultiFactorStrategy,
    "sector_rotation": SectorRotationStrategy,
    "sma_crossover": SMACrossoverStrategy,
}


@dataclass
class StrategyPerformance:
    name: str
    sharpe: float
    win_rate: float
    max_drawdown: float
    total_return: float
    calmar: float
    raw_weight: float
    final_weight: float


def _run_recent_backtest(
    strategy_name: str,
    data: Dict[str, pd.DataFrame],
    lookback_days: int = 80,
    params: Optional[Dict] = None,
) -> Optional[BacktestResult]:
    """对策略在最近 N 天数据上运行回测"""
    cls = STRATEGY_CLASSES.get(strategy_name)
    if cls is None:
        return None

    trimmed = {}
    for code, df in data.items():
        if len(df) > lookback_days:
            trimmed[code] = df.iloc[-lookback_days:].copy()
        else:
            trimmed[code] = df.copy()

    if not trimmed:
        return None

    try:
        strategy = cls(params=params) if params else cls()
        engine = BacktestEngine()
        return engine.run(strategy, trimmed)
    except Exception as e:
        logger.debug(f"Recent backtest failed for {strategy_name}: {e}")
        return None


def compute_strategy_weights(
    data: Dict[str, pd.DataFrame],
    strategy_names: List[str],
    optimized_params: Optional[Dict[str, Dict]] = None,
    lookback_days: int = 80,
    min_weight: float = 0.05,
    method: str = "sharpe",
) -> Dict[str, StrategyPerformance]:
    """
    根据近期回测表现计算策略动态权重

    Args:
        data: 全量行情数据
        strategy_names: 要评估的策略列表
        optimized_params: 经优化后的参数 {strategy_name: params}
        lookback_days: 回测回看天数
        min_weight: 最低权重（保底分配）
        method: 权重计算方式 "sharpe" | "composite"

    Returns:
        {strategy_name: StrategyPerformance}
    """
    performances = {}
    raw_scores = {}
    total = len(strategy_names)

    for idx, name in enumerate(strategy_names):
        print(f"  [Weighter] 回测策略 {idx+1}/{total}: {name}", flush=True)
        params = optimized_params.get(name) if optimized_params else None
        result = _run_recent_backtest(name, data, lookback_days, params)

        if result is None:
            performances[name] = StrategyPerformance(
                name=name, sharpe=0, win_rate=0, max_drawdown=0,
                total_return=0, calmar=0, raw_weight=0, final_weight=0,
            )
            raw_scores[name] = 0.0
            continue

        m = result.metrics
        sharpe = m.get("sharpe_ratio", 0)
        win_rate = m.get("win_rate", 0)
        max_dd = m.get("max_drawdown", 0)
        total_ret = m.get("total_return", 0)
        calmar = m.get("calmar_ratio", 0)

        if method == "sharpe":
            score = max(sharpe, 0)
        else:
            score = max(0, sharpe) * 0.4 + max(0, win_rate - 0.4) * 0.3 + max(0, calmar) * 0.2 + max(0, 1 - max_dd) * 0.1

        raw_scores[name] = score
        performances[name] = StrategyPerformance(
            name=name,
            sharpe=round(sharpe, 3),
            win_rate=round(win_rate, 3),
            max_drawdown=round(max_dd, 4),
            total_return=round(total_ret, 4),
            calmar=round(calmar, 3),
            raw_weight=0,
            final_weight=0,
        )

    total_score = sum(raw_scores.values())

    if total_score > 0:
        for name in strategy_names:
            raw_w = raw_scores[name] / total_score
            performances[name].raw_weight = round(raw_w, 4)
    else:
        equal = 1.0 / len(strategy_names) if strategy_names else 0
        for name in strategy_names:
            performances[name].raw_weight = round(equal, 4)

    # 应用最低权重保底
    n = len(strategy_names)
    remaining = 1.0 - min_weight * n
    if remaining < 0:
        min_weight = 1.0 / n
        remaining = 0

    for name in strategy_names:
        raw_w = performances[name].raw_weight
        final_w = min_weight + raw_w * remaining
        performances[name].final_weight = round(final_w, 4)

    # 归一化
    total_final = sum(p.final_weight for p in performances.values())
    if total_final > 0:
        for p in performances.values():
            p.final_weight = round(p.final_weight / total_final, 4)

    logger.info(
        f"[StrategyWeighter] Weights ({lookback_days}d): "
        + " | ".join(f"{n}={p.final_weight:.1%}" for n, p in performances.items())
    )

    return performances


def weighted_signal_aggregation(
    signals_by_strategy: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
    buy_threshold: float = 0.3,
    sell_threshold: float = -0.3,
) -> pd.DataFrame:
    """
    加权聚合多策略信号矩阵

    Args:
        signals_by_strategy: {strategy_name: signal_matrix(DataFrame, values in {-1,0,1})}
        weights: {strategy_name: weight}
        buy_threshold: 加权分数超过此值发出BUY
        sell_threshold: 加权分数低于此值发出SELL

    Returns:
        综合信号 DataFrame (values: 1=BUY, 0=HOLD, -1=SELL)
    """
    if not signals_by_strategy:
        return pd.DataFrame()

    ref = list(signals_by_strategy.values())[0]
    weighted_sum = pd.DataFrame(0.0, index=ref.index, columns=ref.columns)

    for name, sig_matrix in signals_by_strategy.items():
        w = weights.get(name, 0)
        aligned = sig_matrix.reindex(index=ref.index, columns=ref.columns).fillna(0)
        weighted_sum += aligned * w

    result = pd.DataFrame(0, index=ref.index, columns=ref.columns)
    result[weighted_sum >= buy_threshold] = 1
    result[weighted_sum <= sell_threshold] = -1

    return result


def format_weight_report(performances: Dict[str, StrategyPerformance]) -> str:
    lines = [
        f"\n{'=' * 60}",
        "  策略动态权重分配（基于近期回测）",
        f"{'=' * 60}",
    ]
    for name, p in performances.items():
        lines.append(f"  [{name}] 权重: {p.final_weight:.1%}")
        lines.append(f"    夏普: {p.sharpe:.2f} | 胜率: {p.win_rate:.1%} | "
                     f"最大回撤: {p.max_drawdown:.1%} | 收益: {p.total_return:.2%}")
    return "\n".join(lines)
