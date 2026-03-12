"""
离线模型训练模块
对所有基金进行半年回测训练 + 最近7天验证，保存最优策略参数供 OpenClaw 推理使用
"""
from __future__ import annotations

import itertools
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from loguru import logger

from backtest.engine import BacktestEngine
from config.settings import SECTOR_UNIVERSE, Sector
from data.fetcher import DataFetcher
from strategy.optimizer import PARAM_SEARCH_SPACE, STRATEGY_CLASSES

TRAIN_MONTHS = 6
TEST_DAYS = 7
MAX_COMBOS_PER_STRATEGY = 80  # 每个策略最多测试参数组合数，加速训练

# 保存路径：使用 quant_openclaw 项目下的 cache 目录
_CACHE_ROOT = Path(__file__).resolve().parent.parent / "cache"
TRAINED_PARAMS_PATH = _CACHE_ROOT / "trained_params.json"

STRATEGY_NAMES = [
    "momentum", "mean_reversion", "multi_factor", "sector_rotation", "sma_crossover"
]


def _get_fund_data(fetcher: DataFetcher, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """仅拉取基金数据"""
    result = {}
    for sector in Sector:
        cfg = SECTOR_UNIVERSE[sector]
        for code in cfg.funds.keys():
            df = fetcher.fetch_fund_daily(code, start_date, end_date)
            if not df.empty and "close" in df.columns:
                result[f"FUND_{code}"] = df
    return result


def _optimize_on_data(
    data: Dict[str, pd.DataFrame],
    strategy_name: str,
    max_combos: int,
) -> Dict[str, Any]:
    """在指定数据上做网格搜索，返回最优参数及指标"""
    cls = STRATEGY_CLASSES.get(strategy_name)
    if cls is None:
        return {}

    space = PARAM_SEARCH_SPACE.get(strategy_name, {})
    if not space:
        return {}

    param_names = list(space.keys())
    param_values = list(space.values())
    combos = list(itertools.product(*param_values))

    if len(combos) > max_combos:
        import random
        random.seed(42)
        combos = random.sample(combos, max_combos)

    engine = BacktestEngine()
    best_composite = -999
    best_params = {}
    best_metrics = {}

    for params in combos:
        try:
            kwargs = dict(zip(param_names, params))
            strategy = cls(params=kwargs)
            result = engine.run(strategy, data)
            sharpe = result.metrics.get("sharpe_ratio", -999)
            win_rate = result.metrics.get("win_rate", 0)
            max_dd = result.metrics.get("max_drawdown", 1)
            composite = sharpe - max(0, max_dd - 0.2) * 2 + (win_rate - 0.45) * 0.5

            if composite > best_composite:
                best_composite = composite
                best_params = kwargs
                best_metrics = dict(result.metrics)

        except Exception as e:
            logger.debug(f"Combo {params} failed: {e}")

    return {"params": best_params, "metrics": best_metrics}


def _compute_strategy_weights(
    data: Dict[str, pd.DataFrame],
    optimized_params: Dict[str, Dict],
) -> Dict[str, float]:
    """基于近期回测计算策略权重"""
    performances = {}
    engine = BacktestEngine()

    for name in STRATEGY_NAMES:
        cls = STRATEGY_CLASSES.get(name)
        if cls is None:
            continue
        try:
            params = optimized_params.get(name)
            strategy = cls(params=params) if params else cls()
            result = engine.run(strategy, data)
            sharpe = result.metrics.get("sharpe_ratio", 0)
            win_rate = result.metrics.get("win_rate", 0)
            calmar = result.metrics.get("calmar_ratio", 0)
            max_dd = result.metrics.get("max_drawdown", 1)
            score = max(0, sharpe) * 0.4 + max(0, win_rate - 0.4) * 0.3 + max(0, calmar) * 0.2
            performances[name] = max(score, 0)
        except Exception as e:
            logger.debug(f"Weight backtest failed for {name}: {e}")
            performances[name] = 0.0

    total = sum(performances.values())
    if total <= 0:
        return {n: 1.0 / len(STRATEGY_NAMES) for n in STRATEGY_NAMES}
    return {n: round(v / total, 4) for n, v in performances.items()}


def run_training(
    train_months: int = TRAIN_MONTHS,
    test_days: int = TEST_DAYS,
    max_combos: int = MAX_COMBOS_PER_STRATEGY,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    执行离线训练：6个月回测 + 7天验证，保存最优参数

    Returns:
        保存的参数字典（含 optimized_params, strategy_weights, trained_at 等）
    """
    def _log(msg: str):
        if verbose:
            print(msg, flush=True)

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=train_months * 31)
    start_date = start_dt.strftime("%Y%m%d")
    end_date = end_dt.strftime("%Y%m%d")

    _log("[Train] 1. 加载基金数据 (6个月)...")
    fetcher = DataFetcher()
    fund_data = _get_fund_data(fetcher, start_date, end_date)
    if not fund_data:
        raise RuntimeError("未获取到任何基金数据，请检查网络或基金代码")

    _log(f"  -> 已加载 {len(fund_data)} 只基金")

    # 训练集：全量，验证集：最后7天（用于报告）
    test_end = end_dt
    test_start = end_dt - timedelta(days=test_days)
    train_data = fund_data  # 全量用于优化
    test_data = {}
    for code, df in fund_data.items():
        mask = (df.index >= pd.Timestamp(test_start)) & (df.index <= pd.Timestamp(test_end))
        trimmed = df.loc[mask]
        if len(trimmed) >= 2:  # 至少2个交易日才有意义
            # 验证时需要前面若干日做指标预热，保留到 test_start 前 60 天的数据
            warm_start = test_start - timedelta(days=60)
            full = df[df.index >= pd.Timestamp(warm_start)]
            if len(full) >= 5:
                test_data[code] = full

    _log("[Train] 2. 参数优化 (各策略网格搜索)...")
    optimized_params = {}
    opt_details = {}

    for i, name in enumerate(STRATEGY_NAMES):
        _log(f"  -> [{i+1}/{len(STRATEGY_NAMES)}] {name} ...")
        res = _optimize_on_data(train_data, name, max_combos)
        if res.get("params"):
            optimized_params[name] = res["params"]
            opt_details[name] = {
                "sharpe": res["metrics"].get("sharpe_ratio", 0),
                "win_rate": res["metrics"].get("win_rate", 0),
                "max_drawdown": res["metrics"].get("max_drawdown", 0),
            }

    _log("[Train] 3. 计算策略权重...")
    strategy_weights = _compute_strategy_weights(train_data, optimized_params)
    _log(f"  -> 权重: {' | '.join(f'{k}={v:.0%}' for k, v in strategy_weights.items())}")

    # 在验证集上跑一遍，记录7天表现（可选）
    validation_7d = {}
    if test_data:
        _log("[Train] 4. 7日验证集回测...")
        engine = BacktestEngine()
        for name in STRATEGY_NAMES:
            cls = STRATEGY_CLASSES.get(name)
            params = optimized_params.get(name)
            if cls is None or not params:
                continue
            try:
                strategy = cls(params=params)
                result = engine.run(strategy, test_data)
                validation_7d[name] = {
                    "sharpe": result.metrics.get("sharpe_ratio", 0),
                    "total_return": result.metrics.get("total_return", 0),
                }
            except Exception:
                pass

    payload = {
        "optimized_params": optimized_params,
        "strategy_weights": strategy_weights,
        "trained_at": datetime.now().isoformat(),
        "train_months": train_months,
        "test_days": test_days,
        "fund_count": len(fund_data),
        "opt_details": opt_details,
        "validation_7d": validation_7d,
    }

    out_path = TRAINED_PARAMS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    _log(f"[Train] 5. 参数已保存至 {out_path}")
    return payload


def load_trained_params() -> Dict[str, Any] | None:
    """
    加载已训练参数，若不存在或格式错误返回 None
    """
    if not TRAINED_PARAMS_PATH.exists():
        return None
    try:
        with open(TRAINED_PARAMS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load trained params: {e}")
        return None
