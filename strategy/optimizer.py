"""
回测驱动的策略参数优化器
对每个策略的关键参数进行网格搜索，用回测夏普比率选出最优参数组合
支持缓存优化结果，避免重复计算
"""
from __future__ import annotations

import hashlib
import itertools
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import pandas as pd
from loguru import logger

from backtest.engine import BacktestEngine, BacktestResult
from config.settings import settings
from strategy.base import BaseStrategy
from strategy.momentum import MomentumStrategy
from strategy.mean_reversion import MeanReversionStrategy
from strategy.multi_factor import MultiFactorStrategy
from strategy.sector_rotation import SectorRotationStrategy
from strategy.sma_crossover import SMACrossoverStrategy


PARAM_SEARCH_SPACE: Dict[str, Dict[str, List[Any]]] = {
    "momentum": {
        "lookback": [10, 15, 20, 30],
        "holding_period": [5, 10, 15],
        "top_pct": [0.2, 0.3, 0.4],
        "min_momentum": [0.01, 0.02, 0.03],
    },
    "mean_reversion": {
        "window": [10, 15, 20, 30],
        "zscore_entry": [-2.0, -1.5, -1.0],
        "zscore_exit": [-0.2, 0.0, 0.2],
        "bb_width_min": [0.02, 0.03, 0.05],
    },
    "multi_factor": {
        "top_pct": [0.2, 0.3, 0.4],
        "rebalance_freq": [5, 10, 20],
        "lookback": [15, 20, 30],
    },
    "sector_rotation": {
        "lookback": [10, 15, 20, 30],
        "top_n": [1, 2, 3],
        "rebalance_freq": [10, 15, 20],
    },
    "sma_crossover": {
        "short_window": [3, 5, 8, 10],
        "long_window": [15, 20, 25, 30],
        "confirm_bars": [1, 2, 3],
        "stop_profit_pct": [0.06, 0.08, 0.10],
    },
}

STRATEGY_CLASSES: Dict[str, Type[BaseStrategy]] = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "multi_factor": MultiFactorStrategy,
    "sector_rotation": SectorRotationStrategy,
    "sma_crossover": SMACrossoverStrategy,
}


@dataclass
class OptimizationResult:
    strategy_name: str
    best_params: Dict[str, Any]
    best_sharpe: float
    best_metrics: Dict[str, float]
    all_results: List[Dict]
    data_hash: str


def _data_fingerprint(data: Dict[str, pd.DataFrame]) -> str:
    """对数据集生成唯一指纹用于缓存"""
    parts = []
    for code in sorted(data.keys())[:5]:
        df = data[code]
        if not df.empty and "close" in df.columns:
            parts.append(f"{code}:{len(df)}:{df['close'].iloc[-1]:.2f}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


def _cache_path(strategy_name: str) -> Path:
    p = Path(settings.data_cache_dir) / "optimizer"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{strategy_name}_optimal.pkl"


def optimize_strategy(
    strategy_name: str,
    data: Dict[str, pd.DataFrame],
    search_space: Optional[Dict[str, List[Any]]] = None,
    metric: str = "sharpe_ratio",
    max_combos: int = 200,
    use_cache: bool = True,
) -> OptimizationResult:
    """
    对单个策略进行参数网格搜索优化

    Args:
        strategy_name: 策略名称
        data: 行情数据
        search_space: 参数搜索空间，None 则使用默认
        metric: 优化目标指标
        max_combos: 最大参数组合数
        use_cache: 是否使用缓存
    """
    cls = STRATEGY_CLASSES.get(strategy_name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    space = search_space or PARAM_SEARCH_SPACE.get(strategy_name, {})
    if not space:
        logger.warning(f"No search space defined for {strategy_name}")
        return OptimizationResult(
            strategy_name=strategy_name, best_params={},
            best_sharpe=0.0, best_metrics={}, all_results=[],
            data_hash="",
        )

    data_hash = _data_fingerprint(data)

    if use_cache:
        cache = _cache_path(strategy_name)
        if cache.exists():
            try:
                cached: OptimizationResult = pickle.loads(cache.read_bytes())
                if cached.data_hash == data_hash:
                    logger.info(f"[Optimizer] Using cached result for {strategy_name}")
                    return cached
            except Exception:
                pass

    param_names = list(space.keys())
    param_values = list(space.values())
    combos = list(itertools.product(*param_values))

    if len(combos) > max_combos:
        import random
        random.seed(42)
        combos = random.sample(combos, max_combos)

    total = len(combos)
    logger.info(f"[Optimizer] {strategy_name}: testing {total} param combos")
    print(f"[Optimizer] {strategy_name}: 0/{total} ...", flush=True)
    engine = BacktestEngine()
    all_results = []
    best_sharpe = -999
    best_params = {}
    best_metrics = {}

    for i, combo in enumerate(combos):
        if (i + 1) % max(1, total // 5) == 0 or i + 1 == total:
            pct = (i + 1) / total * 100
            print(f"[Optimizer] {strategy_name}: {i+1}/{total} ({pct:.0f}%)", flush=True)

        params = dict(zip(param_names, combo))
        try:
            strategy = cls(params=params)
            result = engine.run(strategy, data)
            sharpe = result.metrics.get(metric, -999)
            win_rate = result.metrics.get("win_rate", 0)
            max_dd = result.metrics.get("max_drawdown", 1)

            composite = sharpe - max(0, max_dd - 0.2) * 2 + (win_rate - 0.45) * 0.5

            entry = {
                "params": params,
                "sharpe": round(sharpe, 4),
                "win_rate": round(win_rate, 4),
                "max_dd": round(max_dd, 4),
                "total_return": round(result.metrics.get("total_return", 0), 4),
                "composite": round(composite, 4),
            }
            all_results.append(entry)

            if composite > best_sharpe:
                best_sharpe = composite
                best_params = params
                best_metrics = result.metrics

        except Exception as e:
            logger.debug(f"[Optimizer] Combo {i} failed: {e}")

    all_results.sort(key=lambda x: x["composite"], reverse=True)

    opt_result = OptimizationResult(
        strategy_name=strategy_name,
        best_params=best_params,
        best_sharpe=best_sharpe,
        best_metrics=best_metrics,
        all_results=all_results[:20],
        data_hash=data_hash,
    )

    try:
        _cache_path(strategy_name).write_bytes(pickle.dumps(opt_result))
    except Exception as e:
        logger.warning(f"Failed to cache optimization result: {e}")

    logger.info(
        f"[Optimizer] {strategy_name} best: {best_params} | "
        f"Sharpe={best_metrics.get('sharpe_ratio', 0):.2f} | "
        f"WinRate={best_metrics.get('win_rate', 0):.1%} | "
        f"MaxDD={best_metrics.get('max_drawdown', 0):.1%}"
    )
    return opt_result


def optimize_all(
    data: Dict[str, pd.DataFrame],
    strategies: Optional[List[str]] = None,
    use_cache: bool = True,
) -> Dict[str, OptimizationResult]:
    """优化所有策略并返回最优参数"""
    names = strategies or list(STRATEGY_CLASSES.keys())
    results = {}
    for name in names:
        try:
            results[name] = optimize_strategy(name, data, use_cache=use_cache)
        except Exception as e:
            logger.error(f"Optimization failed for {name}: {e}")
    return results


def format_optimization_report(results: Dict[str, OptimizationResult]) -> str:
    """格式化优化结果为可读报告"""
    lines = [
        f"\n{'=' * 60}",
        "  参数优化结果",
        f"{'=' * 60}",
    ]
    for name, r in results.items():
        lines.append(f"\n  [{name}]")
        lines.append(f"    最优参数: {r.best_params}")
        lines.append(f"    夏普比率: {r.best_metrics.get('sharpe_ratio', 0):.3f}")
        lines.append(f"    胜率: {r.best_metrics.get('win_rate', 0):.1%}")
        lines.append(f"    最大回撤: {r.best_metrics.get('max_drawdown', 0):.1%}")
        lines.append(f"    总收益: {r.best_metrics.get('total_return', 0):.2%}")
        lines.append(f"    测试组合数: {len(r.all_results)}")
    return "\n".join(lines)
