"""
走出样本验证 (Walk-Forward Analysis)
将历史数据分成滚动的训练/测试窗口:
  - 训练窗口: 优化策略参数
  - 测试窗口: 用优化后参数做样本外回测
拼接所有测试窗口的结果，得到纯样本外绩效
同时计算过拟合比率 = 样本内Sharpe / 样本外Sharpe
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

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
from strategy.optimizer import optimize_strategy, PARAM_SEARCH_SPACE

STRATEGY_CLASSES: Dict[str, Type[BaseStrategy]] = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "multi_factor": MultiFactorStrategy,
    "sector_rotation": SectorRotationStrategy,
    "sma_crossover": SMACrossoverStrategy,
}


@dataclass
class WalkForwardFold:
    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: Dict[str, Any]
    in_sample_sharpe: float
    out_sample_sharpe: float
    out_sample_return: float
    out_sample_max_dd: float
    out_sample_win_rate: float


@dataclass
class WalkForwardResult:
    strategy_name: str
    folds: List[WalkForwardFold]
    aggregate_oos_sharpe: float
    aggregate_oos_return: float
    aggregate_oos_max_dd: float
    aggregate_oos_win_rate: float
    avg_is_sharpe: float
    avg_oos_sharpe: float
    overfit_ratio: float
    recommended_params: Dict[str, Any]


def _split_data(
    data: Dict[str, pd.DataFrame],
    start_idx: int,
    end_idx: int,
    reference_index: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """从数据中切出 [start_idx, end_idx) 范围"""
    start_date = reference_index[start_idx]
    end_date = reference_index[min(end_idx, len(reference_index) - 1)]

    trimmed = {}
    for code, df in data.items():
        mask = (df.index >= start_date) & (df.index <= end_date)
        sub = df.loc[mask]
        if not sub.empty:
            trimmed[code] = sub.copy()
    return trimmed


def walk_forward_analysis(
    strategy_name: str,
    data: Dict[str, pd.DataFrame],
    train_days: int = 80,
    test_days: int = 40,
    step_days: int = 30,
    max_folds: int = 8,
) -> WalkForwardResult:
    """
    执行走出样本分析

    Args:
        strategy_name: 策略名称
        data: 完整行情数据
        train_days: 训练窗口天数
        test_days: 测试窗口天数
        step_days: 滚动步长
        max_folds: 最大折叠数
    """
    cls = STRATEGY_CLASSES.get(strategy_name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # 建立统一时间索引
    all_dates = set()
    for df in data.values():
        all_dates.update(df.index)
    ref_index = pd.DatetimeIndex(sorted(all_dates))
    total_days = len(ref_index)

    min_required = train_days + test_days
    if total_days < min_required:
        logger.warning(f"Insufficient data ({total_days} < {min_required}) for walk-forward")
        return WalkForwardResult(
            strategy_name=strategy_name, folds=[], aggregate_oos_sharpe=0,
            aggregate_oos_return=0, aggregate_oos_max_dd=0, aggregate_oos_win_rate=0,
            avg_is_sharpe=0, avg_oos_sharpe=0, overfit_ratio=1.0,
            recommended_params={},
        )

    # 预计算总折数
    est_folds = min(max_folds, max(0, (total_days - train_days - test_days) // step_days + 1))
    print(f"[WalkForward] {strategy_name}: 预计 {est_folds} 折", flush=True)

    folds = []
    fold_idx = 0
    start = 0

    while start + train_days + test_days <= total_days and fold_idx < max_folds:
        train_end = start + train_days
        test_end = train_end + test_days

        train_data = _split_data(data, start, train_end, ref_index)
        test_data = _split_data(data, train_end, test_end, ref_index)

        if not train_data or not test_data:
            start += step_days
            continue

        print(
            f"[WalkForward] {strategy_name}: 折 {fold_idx+1}/{est_folds} "
            f"({ref_index[start].strftime('%m-%d')}~{ref_index[min(test_end-1, total_days-1)].strftime('%m-%d')})",
            flush=True,
        )
        logger.info(
            f"[WalkForward] {strategy_name} fold {fold_idx}: "
            f"train={ref_index[start].strftime('%Y-%m-%d')}~{ref_index[train_end-1].strftime('%Y-%m-%d')} "
            f"test={ref_index[train_end].strftime('%Y-%m-%d')}~{ref_index[min(test_end-1, total_days-1)].strftime('%Y-%m-%d')}"
        )

        try:
            # 训练阶段：优化参数
            opt_result = optimize_strategy(
                strategy_name, train_data,
                max_combos=80, use_cache=False,
            )
            best_params = opt_result.best_params
            is_sharpe = opt_result.best_metrics.get("sharpe_ratio", 0)

            # 测试阶段：用优化后参数做样本外回测
            strategy = cls(params=best_params)
            engine = BacktestEngine()
            oos_result = engine.run(strategy, test_data)

            oos_sharpe = oos_result.metrics.get("sharpe_ratio", 0)
            oos_return = oos_result.metrics.get("total_return", 0)
            oos_max_dd = oos_result.metrics.get("max_drawdown", 0)
            oos_win_rate = oos_result.metrics.get("win_rate", 0)

            fold = WalkForwardFold(
                fold_idx=fold_idx,
                train_start=ref_index[start].strftime("%Y-%m-%d"),
                train_end=ref_index[train_end - 1].strftime("%Y-%m-%d"),
                test_start=ref_index[train_end].strftime("%Y-%m-%d"),
                test_end=ref_index[min(test_end - 1, total_days - 1)].strftime("%Y-%m-%d"),
                best_params=best_params,
                in_sample_sharpe=round(is_sharpe, 3),
                out_sample_sharpe=round(oos_sharpe, 3),
                out_sample_return=round(oos_return, 4),
                out_sample_max_dd=round(oos_max_dd, 4),
                out_sample_win_rate=round(oos_win_rate, 4),
            )
            folds.append(fold)

        except Exception as e:
            logger.warning(f"[WalkForward] Fold {fold_idx} failed: {e}")

        fold_idx += 1
        start += step_days

    if not folds:
        return WalkForwardResult(
            strategy_name=strategy_name, folds=[], aggregate_oos_sharpe=0,
            aggregate_oos_return=0, aggregate_oos_max_dd=0, aggregate_oos_win_rate=0,
            avg_is_sharpe=0, avg_oos_sharpe=0, overfit_ratio=1.0,
            recommended_params={},
        )

    avg_is = np.mean([f.in_sample_sharpe for f in folds])
    avg_oos = np.mean([f.out_sample_sharpe for f in folds])
    agg_return = np.mean([f.out_sample_return for f in folds])
    agg_dd = np.max([f.out_sample_max_dd for f in folds])
    agg_wr = np.mean([f.out_sample_win_rate for f in folds])

    overfit = avg_is / avg_oos if avg_oos != 0 else float("inf")

    # 推荐参数：选择 OOS 夏普最高的 fold 的参数
    best_fold = max(folds, key=lambda f: f.out_sample_sharpe)
    recommended = best_fold.best_params

    result = WalkForwardResult(
        strategy_name=strategy_name,
        folds=folds,
        aggregate_oos_sharpe=round(avg_oos, 3),
        aggregate_oos_return=round(agg_return, 4),
        aggregate_oos_max_dd=round(agg_dd, 4),
        aggregate_oos_win_rate=round(agg_wr, 4),
        avg_is_sharpe=round(avg_is, 3),
        avg_oos_sharpe=round(avg_oos, 3),
        overfit_ratio=round(overfit, 2),
        recommended_params=recommended,
    )

    logger.info(
        f"[WalkForward] {strategy_name}: {len(folds)} folds | "
        f"IS Sharpe={avg_is:.2f} | OOS Sharpe={avg_oos:.2f} | "
        f"Overfit={overfit:.2f}x | OOS Return={agg_return:.2%}"
    )
    return result


def walk_forward_all(
    data: Dict[str, pd.DataFrame],
    strategies: Optional[List[str]] = None,
    train_days: int = 80,
    test_days: int = 40,
) -> Dict[str, WalkForwardResult]:
    """对所有策略执行走出样本分析"""
    names = strategies or list(STRATEGY_CLASSES.keys())
    results = {}
    for name in names:
        try:
            results[name] = walk_forward_analysis(name, data, train_days, test_days)
        except Exception as e:
            logger.error(f"Walk-forward failed for {name}: {e}")
    return results


def format_walk_forward_report(results: Dict[str, WalkForwardResult]) -> str:
    lines = [
        f"\n{'=' * 60}",
        "  走出样本验证 (Walk-Forward Analysis)",
        f"{'=' * 60}",
    ]
    for name, r in results.items():
        health = "健康" if r.overfit_ratio < 2.0 else ("注意" if r.overfit_ratio < 3.0 else "过拟合风险")
        lines.append(f"\n  [{name}] ({len(r.folds)} 折)")
        lines.append(f"    样本内夏普:  {r.avg_is_sharpe:.3f}")
        lines.append(f"    样本外夏普:  {r.avg_oos_sharpe:.3f}")
        lines.append(f"    过拟合比率:  {r.overfit_ratio:.2f}x  [{health}]")
        lines.append(f"    样本外收益:  {r.aggregate_oos_return:.2%}")
        lines.append(f"    样本外最大回撤: {r.aggregate_oos_max_dd:.1%}")
        lines.append(f"    样本外胜率:  {r.aggregate_oos_win_rate:.1%}")
        lines.append(f"    推荐参数:    {r.recommended_params}")

        if r.folds:
            lines.append("    各折明细:")
            for f in r.folds:
                lines.append(
                    f"      Fold{f.fold_idx}: "
                    f"[{f.train_start}~{f.test_end}] "
                    f"IS={f.in_sample_sharpe:.2f} OOS={f.out_sample_sharpe:.2f} "
                    f"Ret={f.out_sample_return:.1%}"
                )
    return "\n".join(lines)
