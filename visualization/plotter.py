"""
可视化模块：净值曲线、回撤图、因子热力图、板块对比图
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path("output/charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class QuantPlotter:
    """量化策略可视化工具"""

    @staticmethod
    def plot_nav_curve(
        nav_df: pd.DataFrame,
        strategy_name: str,
        benchmark_col: Optional[str] = "benchmark_nav",
        save_path: Optional[str] = None,
    ) -> str:
        """绘制净值曲线与基准对比"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]})

        ax1 = axes[0]
        ax1.plot(nav_df.index, nav_df["nav"], label=f"{strategy_name}", linewidth=2, color="#2196F3")
        if benchmark_col and benchmark_col in nav_df.columns:
            ax1.plot(nav_df.index, nav_df[benchmark_col], label="沪深300", linewidth=1.5, color="#9E9E9E", linestyle="--")
        ax1.fill_between(nav_df.index, 1, nav_df["nav"], alpha=0.1, color="#2196F3")
        ax1.set_ylabel("净值")
        ax1.set_title(f"{strategy_name} 策略净值曲线", fontsize=14, fontweight="bold")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        cummax = nav_df["nav"].cummax()
        drawdown = (nav_df["nav"] - cummax) / cummax
        ax2 = axes[1]
        ax2.fill_between(nav_df.index, 0, drawdown, color="#F44336", alpha=0.5)
        ax2.set_ylabel("回撤")
        ax2.set_title("动态回撤", fontsize=11)
        ax2.grid(True, alpha=0.3)

        ax3 = axes[2]
        if "returns" in nav_df.columns:
            colors = ["#4CAF50" if r >= 0 else "#F44336" for r in nav_df["returns"].fillna(0)]
            ax3.bar(nav_df.index, nav_df["returns"].fillna(0), color=colors, alpha=0.6, width=1)
        ax3.set_ylabel("日收益率")
        ax3.set_title("每日收益率", fontsize=11)
        ax3.grid(True, alpha=0.3)

        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        plt.tight_layout()
        path = save_path or str(OUTPUT_DIR / f"{strategy_name}_nav.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Nav chart saved: {path}")
        return path

    @staticmethod
    def plot_sector_comparison(
        sector_returns: Dict[str, pd.Series],
        title: str = "板块收益对比",
        save_path: Optional[str] = None,
    ) -> str:
        """绘制板块累计收益对比图"""
        fig, ax = plt.subplots(figsize=(14, 7))
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

        for i, (name, returns) in enumerate(sector_returns.items()):
            cum_ret = (1 + returns).cumprod()
            color = colors[i % len(colors)]
            ax.plot(cum_ret.index, cum_ret, label=name, linewidth=2, color=color)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("累计收益")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        plt.tight_layout()
        path = save_path or str(OUTPUT_DIR / "sector_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    @staticmethod
    def plot_correlation_heatmap(
        close_panel: pd.DataFrame,
        title: str = "持仓相关性矩阵",
        save_path: Optional[str] = None,
    ) -> str:
        """绘制持仓相关性热力图"""
        corr = close_panel.pct_change().dropna().corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlBu_r", center=0,
            square=True, linewidths=0.5,
            ax=ax, vmin=-1, vmax=1,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = save_path or str(OUTPUT_DIR / "correlation_heatmap.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    @staticmethod
    def plot_factor_ic_series(
        ic_series: pd.Series,
        factor_name: str,
        save_path: Optional[str] = None,
    ) -> str:
        """绘制因子 IC 时序图"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 7))

        ax1 = axes[0]
        colors = ["#4CAF50" if v >= 0 else "#F44336" for v in ic_series]
        ax1.bar(ic_series.index, ic_series, color=colors, alpha=0.7)
        ax1.axhline(y=ic_series.mean(), color="#2196F3", linestyle="--", label=f"均值={ic_series.mean():.4f}")
        ax1.set_title(f"{factor_name} 因子IC序列", fontsize=14, fontweight="bold")
        ax1.set_ylabel("IC")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        cum_ic = ic_series.cumsum()
        ax2.plot(cum_ic.index, cum_ic, color="#2196F3", linewidth=2)
        ax2.fill_between(cum_ic.index, 0, cum_ic, alpha=0.1, color="#2196F3")
        ax2.set_title("IC累计值", fontsize=11)
        ax2.set_ylabel("累计IC")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = save_path or str(OUTPUT_DIR / f"{factor_name}_ic.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    @staticmethod
    def plot_metrics_dashboard(
        metrics: Dict[str, float],
        strategy_name: str,
        save_path: Optional[str] = None,
    ) -> str:
        """绘制绩效指标仪表盘"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        fig.suptitle(f"{strategy_name} 绩效仪表盘", fontsize=16, fontweight="bold")

        gauge_items = [
            ("年化收益率", metrics.get("annual_return", 0), "{:.2%}", (-0.3, 0.5), "#4CAF50"),
            ("夏普比率", metrics.get("sharpe_ratio", 0), "{:.2f}", (-1, 3), "#2196F3"),
            ("最大回撤", metrics.get("max_drawdown", 0), "{:.2%}", (0, 0.3), "#F44336"),
            ("胜率", metrics.get("win_rate", 0), "{:.2%}", (0, 1), "#FF9800"),
            ("索提诺比率", metrics.get("sortino_ratio", 0), "{:.2f}", (-1, 5), "#9C27B0"),
            ("卡尔马比率", metrics.get("calmar_ratio", 0), "{:.2f}", (-1, 5), "#00BCD4"),
        ]

        for ax, (name, value, fmt, (vmin, vmax), color) in zip(axes.flat, gauge_items):
            normalized = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            normalized = max(0, min(1, normalized))

            ax.barh([0], [normalized], color=color, alpha=0.7, height=0.4)
            ax.barh([0], [1], color="#E0E0E0", alpha=0.3, height=0.4)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_title(name, fontsize=12)
            ax.text(0.5, -0.3, fmt.format(value), ha="center", fontsize=14, fontweight="bold", transform=ax.transAxes)
            ax.set_xticks([])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        path = save_path or str(OUTPUT_DIR / f"{strategy_name}_dashboard.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    @staticmethod
    def plot_trade_analysis(
        trade_df: pd.DataFrame,
        strategy_name: str,
        save_path: Optional[str] = None,
    ) -> str:
        """绘制交易分析图"""
        if trade_df.empty:
            return ""

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{strategy_name} 交易分析", fontsize=14, fontweight="bold")

        ax1 = axes[0, 0]
        if "direction" in trade_df.columns:
            direction_counts = trade_df["direction"].value_counts()
            ax1.pie(direction_counts, labels=direction_counts.index, autopct="%1.1f%%",
                    colors=["#4CAF50", "#F44336"])
            ax1.set_title("买卖方向分布")

        ax2 = axes[0, 1]
        if "code" in trade_df.columns:
            code_counts = trade_df["code"].value_counts().head(10)
            ax2.barh(code_counts.index, code_counts.values, color="#2196F3")
            ax2.set_title("交易频次 Top10")
            ax2.invert_yaxis()

        ax3 = axes[1, 0]
        if "date" in trade_df.columns:
            trade_df["date"] = pd.to_datetime(trade_df["date"])
            monthly = trade_df.set_index("date").resample("M").size()
            ax3.bar(monthly.index, monthly.values, color="#FF9800", width=20)
            ax3.set_title("月度交易次数")
            ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        ax4 = axes[1, 1]
        if "amount" in trade_df.columns:
            ax4.hist(trade_df["amount"], bins=30, color="#9C27B0", alpha=0.7, edgecolor="white")
            ax4.set_title("交易金额分布")
            ax4.set_xlabel("金额")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        path = save_path or str(OUTPUT_DIR / f"{strategy_name}_trades.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path
