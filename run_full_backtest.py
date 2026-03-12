"""
全量回测脚本：对所有板块（含基金）运行全部策略，输出训练/测试/验证结果
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from loguru import logger

from backtest.engine import BacktestEngine
from config.settings import SECTOR_UNIVERSE, Sector, settings
from data.fetcher import DataFetcher
from risk.manager import RiskManager
from strategy.mean_reversion import MeanReversionStrategy
from strategy.momentum import MomentumStrategy
from strategy.multi_factor import MultiFactorStrategy
from strategy.sector_rotation import SectorRotationStrategy
from visualization.plotter import QuantPlotter

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:7s}</level> | {message}")

STRATEGY_MAP = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "sector_rotation": SectorRotationStrategy,
    "multi_factor": MultiFactorStrategy,
}

START_DATE = "20240101"
END_DATE = None


def load_all_data(fetcher, start_date, end_date):
    """加载所有板块的数据（股票 + ETF + 基金）"""
    all_data = {}
    sector_stats = {}

    for sector in Sector:
        cfg = SECTOR_UNIVERSE[sector]
        sector_data = fetcher.fetch_sector_data(sector, start_date, end_date)
        all_data.update(sector_data)

        n_stocks = sum(1 for k in sector_data if not k.startswith("ETF_") and not k.startswith("FUND_"))
        n_etfs = sum(1 for k in sector_data if k.startswith("ETF_"))
        n_funds = sum(1 for k in sector_data if k.startswith("FUND_"))
        sector_stats[cfg.label] = {"stocks": n_stocks, "etfs": n_etfs, "funds": n_funds}

    return all_data, sector_stats


def run_single_strategy(strategy_name, strategy_cls, all_data, benchmark_close):
    """运行单个策略的回测"""
    strategy = strategy_cls()
    risk_mgr = RiskManager()
    engine = BacktestEngine(risk_manager=risk_mgr)
    result = engine.run(strategy, all_data, benchmark_close)
    return result


def print_separator(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def main():
    print_separator("A 股多板块量化系统 - 全量回测（含基金）")
    print(f"  回测区间: {START_DATE} ~ {END_DATE or '最新'}")
    print(f"  初始资金: {settings.backtest.initial_capital:,.0f} 元")
    print(f"  手续费率: {settings.backtest.commission_rate:.4f}")
    print(f"  滑点: {settings.backtest.slippage_pct:.4f}")
    print(f"  印花税: {settings.backtest.stamp_tax:.4f}")

    fetcher = DataFetcher()

    print_separator("第一阶段：数据加载")
    all_data, sector_stats = load_all_data(fetcher, START_DATE, END_DATE)

    print(f"\n  数据加载完成，共 {len(all_data)} 个标的:")
    for label, stats in sector_stats.items():
        print(f"    [{label}] 股票={stats['stocks']}, ETF={stats['etfs']}, 基金={stats['funds']}")

    fund_keys = sorted([k for k in all_data if k.startswith("FUND_")])
    if fund_keys:
        print(f"\n  基金标的明细:")
        for k in fund_keys:
            df = all_data[k]
            code = df["code"].iloc[0]
            print(f"    {k} ({code}): {len(df)} 交易日, "
                  f"净值区间 {df['close'].min():.4f} ~ {df['close'].max():.4f}, "
                  f"最新 {df['close'].iloc[-1]:.4f}")

    benchmark = fetcher.fetch_index_daily("000300", START_DATE, END_DATE)
    bench_close = benchmark["close"] if not benchmark.empty and "close" in benchmark.columns else None
    if bench_close is not None:
        print(f"\n  基准(沪深300): {len(benchmark)} 交易日")

    print_separator("第二阶段：策略训练与回测")
    results = {}
    plotter = QuantPlotter()

    for name, cls in STRATEGY_MAP.items():
        print(f"\n  >>> 正在回测: {name}")
        try:
            result = run_single_strategy(name, cls, all_data, bench_close)
            results[name] = result
            print(result.summary())

            plotter.plot_nav_curve(result.nav_df, name)
            plotter.plot_metrics_dashboard(result.metrics, name)
            if not result.trade_df.empty:
                plotter.plot_trade_analysis(result.trade_df, name)
        except Exception as e:
            logger.error(f"  Strategy {name} failed: {e}")
            import traceback
            traceback.print_exc()

    print_separator("第三阶段：策略对比验证")

    if results:
        header = f"  {'策略':<22s}{'总收益':>10s}{'年化收益':>10s}{'夏普':>8s}{'最大回撤':>10s}{'胜率':>8s}{'索提诺':>8s}{'卡尔马':>8s}{'交易数':>8s}"
        print(header)
        print("  " + "-" * 86)

        best_sharpe = ("", -999)
        best_return = ("", -999)
        min_drawdown = ("", 999)

        for name, r in results.items():
            m = r.metrics
            total_ret = m.get("total_return", 0)
            annual_ret = m.get("annual_return", 0)
            sharpe = m.get("sharpe_ratio", 0)
            max_dd = m.get("max_drawdown", 0)
            win_rate = m.get("win_rate", 0)
            sortino = m.get("sortino_ratio", 0)
            calmar = m.get("calmar_ratio", 0)
            trades = m.get("total_trades", 0)

            print(
                f"  {name:<22s}"
                f"{total_ret:>9.2%}"
                f"{annual_ret:>10.2%}"
                f"{sharpe:>8.2f}"
                f"{max_dd:>10.2%}"
                f"{win_rate:>8.2%}"
                f"{sortino:>8.2f}"
                f"{calmar:>8.2f}"
                f"{trades:>8.0f}"
            )

            if sharpe > best_sharpe[1]:
                best_sharpe = (name, sharpe)
            if total_ret > best_return[1]:
                best_return = (name, total_ret)
            if max_dd < min_drawdown[1]:
                min_drawdown = (name, max_dd)

        print("  " + "-" * 86)
        print(f"\n  ** 最高夏普:  {best_sharpe[0]} ({best_sharpe[1]:.2f})")
        print(f"  ** 最高收益:  {best_return[0]} ({best_return[1]:.2%})")
        print(f"  ** 最小回撤:  {min_drawdown[0]} ({min_drawdown[1]:.2%})")

    if len(results) > 1:
        try:
            panel = fetcher.build_close_panel(all_data)
            plotter.plot_correlation_heatmap(panel, title="全标的相关性矩阵（含基金）")
            logger.info("Correlation heatmap saved")
        except Exception as e:
            logger.warning(f"Heatmap failed: {e}")

    print_separator("回测完成")
    print(f"  图表保存至: output/charts/")
    print(f"  包含: 净值曲线、回撤分析、绩效仪表盘、交易分析、相关性热力图")


if __name__ == "__main__":
    main()
