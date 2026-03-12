"""
回测引擎：模拟策略在历史数据上的实际交易过程
考虑手续费、滑点、印花税、涨跌停限制
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from backtest.metrics import PerformanceMetrics
from config.settings import settings
from risk.manager import RiskManager
from strategy.base import BaseStrategy, Signal


@dataclass
class Position:
    code: str
    shares: int
    avg_cost: float
    current_price: float

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def pnl(self) -> float:
        return self.shares * (self.current_price - self.avg_cost)

    @property
    def pnl_pct(self) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price - self.avg_cost) / self.avg_cost


@dataclass
class TradeRecord:
    date: pd.Timestamp
    code: str
    direction: str      # "BUY" / "SELL"
    price: float
    shares: int
    amount: float
    commission: float
    tax: float


@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    trade_history: List[TradeRecord] = field(default_factory=list)

    @property
    def market_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def total_value(self) -> float:
        return self.cash + self.market_value

    @property
    def position_pct(self) -> float:
        tv = self.total_value
        return self.market_value / tv if tv > 0 else 0.0


class BacktestEngine:
    """
    事件驱动回测引擎：
    - 按日遍历交易信号
    - 根据目标权重下单
    - 模拟成交（考虑交易成本）
    - 记录每日净值和持仓
    """

    def __init__(
        self,
        initial_capital: Optional[float] = None,
        commission_rate: Optional[float] = None,
        slippage_pct: Optional[float] = None,
        stamp_tax: Optional[float] = None,
        risk_manager: Optional[RiskManager] = None,
    ):
        bp = settings.backtest
        self.initial_capital = initial_capital or bp.initial_capital
        self.commission_rate = commission_rate or bp.commission_rate
        self.slippage_pct = slippage_pct or bp.slippage_pct
        self.stamp_tax = stamp_tax or bp.stamp_tax
        self.risk_manager = risk_manager or RiskManager()

    def run(
        self,
        strategy: BaseStrategy,
        data: Dict[str, pd.DataFrame],
        benchmark_close: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """运行回测"""
        logger.info(f"Starting backtest: strategy={strategy.name}")

        signal_matrix = strategy.generate_signals(data)
        close_panel = pd.DataFrame({
            code: data[code]["close"] for code in data if "close" in data[code].columns
        }).dropna(how="all").ffill()

        target_weights = strategy.compute_target_weights(signal_matrix, close_panel)
        target_weights = self.risk_manager.adjust_weights(target_weights, close_panel)

        portfolio = Portfolio(cash=self.initial_capital)
        daily_values = []
        daily_positions = []

        dates = close_panel.index
        for date in dates:
            self._update_prices(portfolio, close_panel, date)

            if date in target_weights.index:
                tw = target_weights.loc[date]
                self._rebalance(portfolio, tw, close_panel, date)

            stop_codes = self.risk_manager.check_stop_loss(portfolio)
            for code in stop_codes:
                self._close_position(portfolio, code, close_panel, date)

            daily_values.append({
                "date": date,
                "total_value": portfolio.total_value,
                "cash": portfolio.cash,
                "market_value": portfolio.market_value,
                "n_positions": len(portfolio.positions),
            })

            daily_positions.append({
                "date": date,
                "positions": {
                    code: {
                        "shares": p.shares,
                        "avg_cost": p.avg_cost,
                        "current_price": p.current_price,
                        "pnl_pct": p.pnl_pct,
                    }
                    for code, p in portfolio.positions.items()
                },
            })

        nav_df = pd.DataFrame(daily_values).set_index("date")
        nav_df["returns"] = nav_df["total_value"].pct_change()
        nav_df["cumulative_returns"] = (1 + nav_df["returns"]).cumprod() - 1
        nav_df["nav"] = nav_df["total_value"] / self.initial_capital

        if benchmark_close is not None:
            bench = benchmark_close.reindex(nav_df.index).ffill()
            nav_df["benchmark_returns"] = bench.pct_change()
            nav_df["benchmark_nav"] = bench / bench.iloc[0]
            nav_df["excess_returns"] = nav_df["returns"] - nav_df["benchmark_returns"]

        metrics = PerformanceMetrics.compute(
            nav_df,
            initial_capital=self.initial_capital,
            benchmark_col="benchmark_nav" if benchmark_close is not None else None,
        )

        trade_df = pd.DataFrame([
            {
                "date": t.date,
                "code": t.code,
                "direction": t.direction,
                "price": t.price,
                "shares": t.shares,
                "amount": t.amount,
                "commission": t.commission,
                "tax": t.tax,
            }
            for t in portfolio.trade_history
        ])

        result = BacktestResult(
            strategy_name=strategy.name,
            nav_df=nav_df,
            trade_df=trade_df,
            metrics=metrics,
            daily_positions=daily_positions,
        )

        logger.info(
            f"Backtest complete: {strategy.name} | "
            f"Return={metrics.get('total_return', 0):.2%} | "
            f"Sharpe={metrics.get('sharpe_ratio', 0):.2f} | "
            f"MaxDD={metrics.get('max_drawdown', 0):.2%}"
        )
        return result

    def _update_prices(
        self, portfolio: Portfolio, close_panel: pd.DataFrame, date: pd.Timestamp
    ):
        for code, pos in list(portfolio.positions.items()):
            if code in close_panel.columns and date in close_panel.index:
                pos.current_price = close_panel.loc[date, code]

    def _rebalance(
        self,
        portfolio: Portfolio,
        target_weights: pd.Series,
        close_panel: pd.DataFrame,
        date: pd.Timestamp,
    ):
        total_value = portfolio.total_value
        for code in target_weights.index:
            if code not in close_panel.columns or date not in close_panel.index:
                continue

            target_w = target_weights[code]
            price = close_panel.loc[date, code]
            if pd.isna(price) or price <= 0:
                continue

            price_with_slippage = price * (1 + self.slippage_pct)
            target_value = total_value * target_w
            target_shares = int(target_value / price_with_slippage / 100) * 100  # A股100股整手

            current_shares = portfolio.positions.get(code, Position(code, 0, 0, price)).shares

            delta = target_shares - current_shares
            if abs(delta) < 100:
                continue

            if delta > 0:
                self._buy(portfolio, code, price_with_slippage, delta, date)
            elif delta < 0:
                sell_price = price * (1 - self.slippage_pct)
                self._sell(portfolio, code, sell_price, abs(delta), date)

        for code in list(portfolio.positions.keys()):
            if code not in target_weights.index or target_weights.get(code, 0) == 0:
                if code in close_panel.columns and date in close_panel.index:
                    sell_price = close_panel.loc[date, code] * (1 - self.slippage_pct)
                    self._sell(
                        portfolio, code, sell_price,
                        portfolio.positions[code].shares, date
                    )

    def _buy(
        self, portfolio: Portfolio, code: str, price: float, shares: int, date: pd.Timestamp
    ):
        amount = price * shares
        commission = max(amount * self.commission_rate, 5.0)
        total_cost = amount + commission

        if total_cost > portfolio.cash:
            shares = int(portfolio.cash / (price * (1 + self.commission_rate)) / 100) * 100
            if shares <= 0:
                return
            amount = price * shares
            commission = max(amount * self.commission_rate, 5.0)
            total_cost = amount + commission

        portfolio.cash -= total_cost

        if code in portfolio.positions:
            pos = portfolio.positions[code]
            total_shares = pos.shares + shares
            pos.avg_cost = (pos.avg_cost * pos.shares + price * shares) / total_shares
            pos.shares = total_shares
            pos.current_price = price
        else:
            portfolio.positions[code] = Position(
                code=code, shares=shares, avg_cost=price, current_price=price
            )

        portfolio.trade_history.append(TradeRecord(
            date=date, code=code, direction="BUY",
            price=price, shares=shares, amount=amount,
            commission=commission, tax=0.0,
        ))

    def _sell(
        self, portfolio: Portfolio, code: str, price: float, shares: int, date: pd.Timestamp
    ):
        if code not in portfolio.positions:
            return
        pos = portfolio.positions[code]
        shares = min(shares, pos.shares)
        if shares <= 0:
            return

        amount = price * shares
        commission = max(amount * self.commission_rate, 5.0)
        tax = amount * self.stamp_tax
        net_proceeds = amount - commission - tax

        portfolio.cash += net_proceeds
        pos.shares -= shares

        if pos.shares <= 0:
            del portfolio.positions[code]

        portfolio.trade_history.append(TradeRecord(
            date=date, code=code, direction="SELL",
            price=price, shares=shares, amount=amount,
            commission=commission, tax=tax,
        ))

    def _close_position(
        self, portfolio: Portfolio, code: str, close_panel: pd.DataFrame, date: pd.Timestamp
    ):
        if code not in portfolio.positions:
            return
        if code in close_panel.columns and date in close_panel.index:
            price = close_panel.loc[date, code] * (1 - self.slippage_pct)
            self._sell(portfolio, code, price, portfolio.positions[code].shares, date)


@dataclass
class BacktestResult:
    strategy_name: str
    nav_df: pd.DataFrame
    trade_df: pd.DataFrame
    metrics: Dict[str, float]
    daily_positions: List[Dict]

    def summary(self) -> str:
        lines = [
            f"{'='*60}",
            f"策略: {self.strategy_name}",
            f"{'='*60}",
        ]
        key_metrics = [
            ("总收益率", "total_return", "{:.2%}"),
            ("年化收益率", "annual_return", "{:.2%}"),
            ("夏普比率", "sharpe_ratio", "{:.2f}"),
            ("索提诺比率", "sortino_ratio", "{:.2f}"),
            ("最大回撤", "max_drawdown", "{:.2%}"),
            ("卡尔马比率", "calmar_ratio", "{:.2f}"),
            ("胜率", "win_rate", "{:.2%}"),
            ("盈亏比", "profit_loss_ratio", "{:.2f}"),
            ("交易次数", "total_trades", "{:.0f}"),
            ("年化换手率", "annual_turnover", "{:.2f}"),
        ]
        for label, key, fmt in key_metrics:
            val = self.metrics.get(key, 0)
            lines.append(f"  {label:12s}: {fmt.format(val)}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)
