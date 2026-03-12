"""
信号置信度评分模块
综合回测胜率、策略共识度、市场状态、新闻情绪，
为每个信号生成 [0, 1] 的置信度分数和文字标签
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class SignalConfidence:
    code: str
    name: str
    signal: str            # "BUY" / "SELL" / "HOLD"
    confidence: float      # [0, 1]
    label: str             # "极高/高/中/低/极低"
    components: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)


# --------------- 子评分器 ---------------

def _backtest_score(
    code: str,
    signal_type: str,
    strategy_performances: Dict[str, "StrategyPerformance"],
    code_signals: List[Dict],
) -> Tuple[float, str]:
    """
    回测胜率维度：基于发出该信号的策略的历史胜率加权计算

    Args:
        strategy_performances: {strategy_name: StrategyPerformance} 来自 strategy_weighter
        code_signals: 该标的的信号列表 [{strategy, signal, ...}]
    """
    if not code_signals or not strategy_performances:
        return 0.5, "回测数据不足"

    relevant = [s for s in code_signals if s.get("signal") == signal_type]
    if not relevant:
        return 0.5, "无匹配信号"

    weighted_wr = 0.0
    total_w = 0.0
    for s in relevant:
        strat = s.get("strategy", "")
        perf = strategy_performances.get(strat)
        if perf:
            w = perf.final_weight
            wr = perf.win_rate
            weighted_wr += w * wr
            total_w += w

    if total_w > 0:
        score = weighted_wr / total_w
        return score, f"加权胜率 {score:.0%}"
    return 0.5, "无性能数据"


def _consensus_score(
    code: str,
    signal_type: str,
    code_signals: List[Dict],
    total_strategies: int,
) -> Tuple[float, str]:
    """
    策略共识维度：同方向信号占策略总数的比例

    3/4 策略看多 → 0.75
    1/4 策略看多 → 0.25
    """
    if total_strategies <= 0:
        return 0.5, "无策略"

    agree_count = sum(1 for s in code_signals if s.get("signal") == signal_type)
    score = agree_count / total_strategies

    if agree_count >= 3:
        label = f"{agree_count}/{total_strategies} 策略共识，强信号"
    elif agree_count == 2:
        label = f"{agree_count}/{total_strategies} 策略一致"
    elif agree_count == 1:
        label = f"仅 1 个策略，信号较弱"
    else:
        label = "无策略支持"

    return score, label


def _market_regime_score(
    code: str,
    signal_type: str,
    close_data: pd.Series,
    vol_window: int = 20,
) -> Tuple[float, str]:
    """
    市场状态维度：根据波动率环境和趋势判断信号可靠性

    - 低波动 + 上升趋势 → 动量/买入信号更可靠
    - 高波动 + 超卖 → 均值回归/买入信号更可靠
    - 高波动 + 下跌趋势 → 卖出信号更可靠
    """
    if close_data is None or len(close_data) < vol_window + 5:
        return 0.5, "数据不足"

    returns = close_data.pct_change().dropna()
    recent_vol = returns.iloc[-vol_window:].std() * np.sqrt(252)
    long_vol = returns.std() * np.sqrt(252)
    vol_ratio = recent_vol / long_vol if long_vol > 0 else 1.0

    # 趋势判断
    sma20 = close_data.iloc[-vol_window:].mean()
    sma60 = close_data.iloc[-min(len(close_data), 60):].mean()
    current = close_data.iloc[-1]
    uptrend = current > sma20 > sma60

    # 超买超卖
    zscore = (current - sma20) / (close_data.iloc[-vol_window:].std() or 1)

    if signal_type == "BUY":
        if uptrend and vol_ratio < 1.2:
            return 0.75, "趋势向上+波动率低，买入信号可靠"
        elif zscore < -1.5:
            return 0.7, "深度超卖，反弹概率较高"
        elif not uptrend and vol_ratio > 1.5:
            return 0.3, "下跌趋势+高波动，买入风险大"
        else:
            return 0.5, "市场状态中性"
    elif signal_type == "SELL":
        if not uptrend and vol_ratio > 1.2:
            return 0.75, "趋势向下+波动升高，卖出合理"
        elif zscore > 2.0:
            return 0.7, "严重超买，回调风险大"
        elif uptrend and vol_ratio < 0.8:
            return 0.3, "强势上涨中，卖出可能过早"
        else:
            return 0.5, "市场状态中性"

    return 0.5, "HOLD 信号"


def _news_sentiment_score(
    code: str,
    signal_type: str,
    news_digest: Optional[Dict] = None,
) -> Tuple[float, str]:
    """
    新闻情绪维度：新闻方向与信号方向一致时加分

    买入 + 利好新闻 → 高置信
    买入 + 利空新闻 → 低置信
    """
    if news_digest is None:
        return 0.5, "无新闻数据"

    mood_score = news_digest.get("score", 0.0)
    mood = news_digest.get("mood", "中性")

    if signal_type == "BUY":
        if mood_score > 0.2:
            return 0.75, f"市场情绪偏多({mood})，利于买入"
        elif mood_score < -0.2:
            return 0.3, f"市场情绪偏空({mood})，买入需谨慎"
        else:
            return 0.5, "情绪中性"
    elif signal_type == "SELL":
        if mood_score < -0.2:
            return 0.75, f"市场情绪偏空({mood})，卖出合理"
        elif mood_score > 0.2:
            return 0.3, f"市场情绪偏多({mood})，卖出可能过早"
        else:
            return 0.5, "情绪中性"

    return 0.5, "HOLD"


# --------------- 主评分函数 ---------------

CONFIDENCE_WEIGHTS = {
    "backtest": 0.30,
    "consensus": 0.30,
    "market_regime": 0.25,
    "news_sentiment": 0.15,
}

CONFIDENCE_LABELS = [
    (0.8, "极高"),
    (0.65, "高"),
    (0.5, "中"),
    (0.35, "低"),
    (0.0, "极低"),
]


def score_signal(
    code: str,
    name: str,
    signal_type: str,
    code_signals: List[Dict],
    total_strategies: int,
    close_data: Optional[pd.Series],
    strategy_performances: Optional[Dict] = None,
    news_digest: Optional[Dict] = None,
) -> SignalConfidence:
    """
    为单个信号计算综合置信度

    Returns:
        SignalConfidence 数据对象
    """
    components = {}
    reasons = []

    # 1. 回测胜率
    bt_score, bt_reason = _backtest_score(
        code, signal_type, strategy_performances or {}, code_signals
    )
    components["backtest"] = bt_score
    reasons.append(f"回测: {bt_reason}")

    # 2. 策略共识
    cons_score, cons_reason = _consensus_score(
        code, signal_type, code_signals, total_strategies
    )
    components["consensus"] = cons_score
    reasons.append(f"共识: {cons_reason}")

    # 3. 市场状态
    mkt_score, mkt_reason = _market_regime_score(
        code, signal_type, close_data
    )
    components["market_regime"] = mkt_score
    reasons.append(f"市场: {mkt_reason}")

    # 4. 新闻情绪
    news_score, news_reason = _news_sentiment_score(
        code, signal_type, news_digest
    )
    components["news_sentiment"] = news_score
    reasons.append(f"新闻: {news_reason}")

    # 加权合成
    confidence = sum(
        components[k] * CONFIDENCE_WEIGHTS[k] for k in CONFIDENCE_WEIGHTS
    )
    confidence = round(max(0.0, min(1.0, confidence)), 3)

    label = "极低"
    for threshold, lbl in CONFIDENCE_LABELS:
        if confidence >= threshold:
            label = lbl
            break

    return SignalConfidence(
        code=code,
        name=name,
        signal=signal_type,
        confidence=confidence,
        label=label,
        components=components,
        reasons=reasons,
    )


def score_all_signals(
    signals_by_strategy: Dict[str, List[Dict]],
    all_data: Dict[str, pd.DataFrame],
    code_names: Dict[str, str],
    strategy_performances: Optional[Dict] = None,
    news_digest: Optional[Dict] = None,
) -> List[SignalConfidence]:
    """
    对所有信号批量计算置信度

    Args:
        signals_by_strategy: {strategy_name: [{code, name, signal, price, strategy}]}
    """
    # 按 code 聚合信号
    code_signal_map: Dict[str, List[Dict]] = {}
    for strat_name, sigs in signals_by_strategy.items():
        for s in sigs:
            code_signal_map.setdefault(s["code"], []).append(s)

    total_strats = len(signals_by_strategy)
    results = []

    for code, sigs in code_signal_map.items():
        buy_sigs = [s for s in sigs if s.get("signal") == "BUY"]
        sell_sigs = [s for s in sigs if s.get("signal") == "SELL"]

        close = all_data.get(code, pd.DataFrame()).get("close")

        if buy_sigs:
            sc = score_signal(
                code, code_names.get(code, code), "BUY", sigs,
                total_strats, close, strategy_performances, news_digest,
            )
            results.append(sc)

        if sell_sigs:
            sc = score_signal(
                code, code_names.get(code, code), "SELL", sigs,
                total_strats, close, strategy_performances, news_digest,
            )
            results.append(sc)

    results.sort(key=lambda x: x.confidence, reverse=True)
    return results


def format_confidence_report(scored: List[SignalConfidence], top_n: int = 15) -> str:
    lines = [
        f"\n{'=' * 60}",
        "  信号置信度排行",
        f"{'=' * 60}",
    ]
    for sc in scored[:top_n]:
        bar = "\u2588" * int(sc.confidence * 10) + "\u2591" * (10 - int(sc.confidence * 10))
        lines.append(
            f"  {sc.signal:4s} {sc.name}({sc.code}) "
            f"[{bar}] {sc.confidence:.0%} ({sc.label})"
        )
        for r in sc.reasons:
            lines.append(f"       {r}")
    return "\n".join(lines)
