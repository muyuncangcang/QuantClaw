"""
主入口：策略回测、信号生成、可视化、API 服务的统一入口
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from typing import List, Optional

from loguru import logger

from backtest.engine import BacktestEngine
from config.settings import SECTOR_UNIVERSE, Sector, settings
from data.fetcher import DataFetcher
from data.processor import FeatureEngineer
from risk.manager import RiskManager
from strategy.mean_reversion import MeanReversionStrategy
from strategy.momentum import MomentumStrategy
from strategy.multi_factor import MultiFactorStrategy
from strategy.sector_rotation import SectorRotationStrategy
from visualization.plotter import QuantPlotter


STRATEGY_MAP = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "sector_rotation": SectorRotationStrategy,
    "multi_factor": MultiFactorStrategy,
}


def run_backtest(
    strategy_name: str,
    sectors: Optional[List[str]] = None,
    start_date: str = "20230101",
    end_date: Optional[str] = None,
    plot: bool = True,
):
    """运行单策略回测"""
    logger.info(f"Running backtest: {strategy_name}")

    fetcher = DataFetcher()
    target_sectors = [Sector(s) for s in sectors] if sectors else list(Sector)

    all_data = {}
    for sector in target_sectors:
        sector_data = fetcher.fetch_sector_data(sector, start_date, end_date)
        all_data.update(sector_data)
        logger.info(f"  Loaded {sector.value}: {len(sector_data)} instruments")

    if not all_data:
        logger.error("No data loaded")
        return

    strategy_cls = STRATEGY_MAP.get(strategy_name)
    if strategy_cls is None:
        logger.error(f"Unknown strategy: {strategy_name}")
        return

    strategy = strategy_cls()
    risk_mgr = RiskManager()
    engine = BacktestEngine(risk_manager=risk_mgr)

    benchmark = fetcher.fetch_index_daily("000300", start_date, end_date)
    bench_close = benchmark["close"] if not benchmark.empty and "close" in benchmark.columns else None

    result = engine.run(strategy, all_data, bench_close)

    print(result.summary())

    if plot:
        plotter = QuantPlotter()
        plotter.plot_nav_curve(result.nav_df, strategy_name)
        plotter.plot_metrics_dashboard(result.metrics, strategy_name)
        if not result.trade_df.empty:
            plotter.plot_trade_analysis(result.trade_df, strategy_name)
        logger.info("Charts saved to output/charts/")

    return result


def run_all_strategies(
    sectors: Optional[List[str]] = None,
    start_date: str = "20230101",
):
    """运行所有策略并对比"""
    results = {}
    for name in STRATEGY_MAP:
        logger.info(f"\n{'='*60}\n  Strategy: {name}\n{'='*60}")
        result = run_backtest(name, sectors, start_date, plot=True)
        if result:
            results[name] = result

    if results:
        print("\n" + "=" * 80)
        print("  策略对比汇总")
        print("=" * 80)
        header = f"{'策略':20s}{'收益率':>12s}{'夏普':>10s}{'回撤':>10s}{'胜率':>10s}"
        print(header)
        print("-" * 62)
        for name, r in results.items():
            m = r.metrics
            print(
                f"{name:20s}"
                f"{m.get('total_return', 0):>11.2%}"
                f"{m.get('sharpe_ratio', 0):>10.2f}"
                f"{m.get('max_drawdown', 0):>10.2%}"
                f"{m.get('win_rate', 0):>10.2%}"
            )
        print("=" * 80)

    return results


def run_signals(
    strategy_name: str = "multi_factor",
    sectors: Optional[List[str]] = None,
):
    """获取最新交易信号"""
    from agent.openclaw_agent import QuantAgent

    agent = QuantAgent()
    signals = agent.get_latest_signals(strategy_name, sectors)

    print(f"\n{'='*60}")
    print(f"  {strategy_name} 最新交易信号")
    print(f"{'='*60}")
    for s in signals:
        print(f"  [{s['signal']}] {s['name']}({s['code']}) @ {s['price']:.2f}")
    print(f"{'='*60}")

    return signals


def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """启动 FastAPI 服务"""
    from agent.openclaw_agent import create_quant_api

    app = create_quant_api()
    if app is None:
        logger.error("Failed to create API app")
        return

    import uvicorn
    logger.info(f"Starting API server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


async def run_agent_pipeline(query: str):
    """运行 OpenClaw Agent Pipeline"""
    from agent.openclaw_agent import QuantAgent

    agent = QuantAgent()
    await agent.connect()
    try:
        result = await agent.run_research_pipeline(query)
        print(result)
    finally:
        await agent.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="A股多板块量化策略系统 (基于OpenClaw)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py backtest --strategy momentum --sectors cpo,semiconductor
  python main.py backtest --strategy multi_factor --start 20230101
  python main.py compare --sectors cpo,semiconductor,storage_chip
  python main.py signals --strategy multi_factor
  python main.py api --port 8000
  python main.py agent --query "分析CPO板块近期走势"
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    bt_parser = subparsers.add_parser("backtest", help="运行策略回测")
    bt_parser.add_argument("--strategy", "-s", required=True, choices=list(STRATEGY_MAP.keys()))
    bt_parser.add_argument("--sectors", type=str, default=None, help="逗号分隔的板块列表")
    bt_parser.add_argument("--start", type=str, default="20230101", help="开始日期 YYYYMMDD")
    bt_parser.add_argument("--end", type=str, default=None, help="结束日期 YYYYMMDD")
    bt_parser.add_argument("--no-plot", action="store_true", help="不生成图表")

    cmp_parser = subparsers.add_parser("compare", help="对比所有策略")
    cmp_parser.add_argument("--sectors", type=str, default=None)
    cmp_parser.add_argument("--start", type=str, default="20230101")

    sig_parser = subparsers.add_parser("signals", help="获取最新信号")
    sig_parser.add_argument("--strategy", "-s", default="multi_factor")
    sig_parser.add_argument("--sectors", type=str, default=None)

    api_parser = subparsers.add_parser("api", help="启动API服务")
    api_parser.add_argument("--host", type=str, default="0.0.0.0")
    api_parser.add_argument("--port", type=int, default=8000)

    agent_parser = subparsers.add_parser("agent", help="OpenClaw Agent查询")
    agent_parser.add_argument("--query", "-q", required=True)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    sectors = args.sectors.split(",") if hasattr(args, "sectors") and args.sectors else None

    if args.command == "backtest":
        run_backtest(args.strategy, sectors, args.start, args.end, not args.no_plot)
    elif args.command == "compare":
        run_all_strategies(sectors, args.start)
    elif args.command == "signals":
        run_signals(args.strategy, sectors)
    elif args.command == "api":
        run_api_server(args.host, args.port)
    elif args.command == "agent":
        asyncio.run(run_agent_pipeline(args.query))


if __name__ == "__main__":
    main()
