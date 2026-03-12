"""
策略基类：定义统一接口，所有策略继承此类
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd


class Signal(int, Enum):
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class TradeSignal:
    date: pd.Timestamp
    code: str
    signal: Signal
    price: float
    weight: float = 1.0
    reason: str = ""


class BaseStrategy(ABC):
    """所有策略的抽象基类"""

    def __init__(self, name: str, params: Optional[Dict] = None):
        self.name = name
        self.params = params or {}
        self.signals: List[TradeSignal] = []

    @abstractmethod
    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        factors: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        生成交易信号矩阵

        Returns:
            DataFrame with columns=stock codes, index=dates, values=Signal enum
        """
        ...

    @abstractmethod
    def compute_target_weights(
        self,
        signal_matrix: pd.DataFrame,
        close_panel: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        基于信号矩阵计算目标持仓权重

        Returns:
            DataFrame with same shape as signal_matrix, values are target weights [0, 1]
        """
        ...

    def _rank_normalize(self, series: pd.Series) -> pd.Series:
        """截面排名归一化到 [0, 1]"""
        ranked = series.rank(pct=True)
        return ranked

    def _zscore(self, series: pd.Series) -> pd.Series:
        """截面标准化"""
        return (series - series.mean()) / series.std()
