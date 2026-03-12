"""
OpenClaw Agent 集成模块：
- 通过 openclaw-sdk 将策略逻辑封装为 AI Agent
- 支持自然语言交互进行策略配置、回测、信号查询
- 对接 FastAPI 提供 HTTP 接口
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel

try:
    from openclaw_sdk import OpenClawClient
    from openclaw_sdk.callbacks.handler import CallbackHandler
    from openclaw_sdk.core.types import ExecutionResult
    from openclaw_sdk.output.structured import StructuredOutput
    from openclaw_sdk.pipeline.pipeline import Pipeline
    from openclaw_sdk.tracking.cost import CostTracker
    OPENCLAW_AVAILABLE = True
except ImportError:
    OPENCLAW_AVAILABLE = False
    logger.warning("openclaw-sdk not installed, agent features disabled")

from backtest.engine import BacktestEngine, BacktestResult
from config.settings import SECTOR_UNIVERSE, Sector, settings
from data.fetcher import DataFetcher
from data.processor import FeatureEngineer
from risk.manager import RiskManager
from strategy.mean_reversion import MeanReversionStrategy
from strategy.momentum import MomentumStrategy
from strategy.multi_factor import MultiFactorStrategy
from strategy.sector_rotation import SectorRotationStrategy


# ─── Pydantic 结构化输出模型 ───

class StrategySignal(BaseModel):
    code: str
    name: str
    signal: str
    score: float
    reason: str


class BacktestSummary(BaseModel):
    strategy: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    top_holdings: List[str]
    recommendation: str


class RiskAlert(BaseModel):
    level: str
    message: str
    affected_positions: List[str]
    suggested_action: str


# ─── Agent 回调 ───

class QuantCallback(CallbackHandler):
    async def on_execution_start(self, agent_id: str, query: str) -> None:
        logger.info(f"[Agent:{agent_id}] Query: {query[:80]}")

    async def on_execution_end(self, agent_id: str, result: "ExecutionResult") -> None:
        logger.info(f"[Agent:{agent_id}] Done in {result.latency_ms}ms")


# ─── 核心 Agent ───

class QuantAgent:
    """
    量化交易 AI Agent：
    - research-bot: 研究分析，生成因子报告
    - strategy-bot: 策略回测与信号生成
    - risk-bot: 风险监控与预警
    - pipeline: 串联 research -> strategy -> risk 完整工作流
    """

    STRATEGY_MAP = {
        "momentum": MomentumStrategy,
        "mean_reversion": MeanReversionStrategy,
        "sector_rotation": SectorRotationStrategy,
        "multi_factor": MultiFactorStrategy,
    }

    def __init__(self):
        self.fetcher = DataFetcher()
        self.fe = FeatureEngineer()
        self.risk_mgr = RiskManager()
        self.engine = BacktestEngine(risk_manager=self.risk_mgr)
        self._client: Optional[Any] = None
        self._cost_tracker: Optional[Any] = None

    async def connect(self):
        if not OPENCLAW_AVAILABLE:
            logger.warning("OpenClaw SDK not available, running in offline mode")
            return
        self._cost_tracker = CostTracker()
        self._client = await OpenClawClient.connect(
            gateway_ws_url=settings.openclaw_ws_url,
            api_key=settings.openclaw_api_key,
            callbacks=[QuantCallback()],
        ).__aenter__()
        logger.info("Connected to OpenClaw gateway")

    async def disconnect(self):
        if self._client:
            await self._client.__aexit__(None, None, None)

    # ─── 核心功能 ───

    def run_backtest(
        self,
        strategy_name: str,
        sectors: Optional[List[str]] = None,
        start_date: str = "20230101",
        end_date: Optional[str] = None,
        params: Optional[Dict] = None,
    ) -> BacktestResult:
        """运行策略回测"""
        target_sectors = [Sector(s) for s in sectors] if sectors else list(Sector)

        all_data: Dict[str, pd.DataFrame] = {}
        for sector in target_sectors:
            sector_data = self.fetcher.fetch_sector_data(sector, start_date, end_date)
            all_data.update(sector_data)

        if not all_data:
            raise ValueError("No data fetched for any sector")

        strategy_cls = self.STRATEGY_MAP.get(strategy_name)
        if strategy_cls is None:
            raise ValueError(f"Unknown strategy: {strategy_name}. Choose from {list(self.STRATEGY_MAP.keys())}")

        strategy = strategy_cls(params=params)

        benchmark = self.fetcher.fetch_index_daily("000300", start_date, end_date)
        bench_close = benchmark["close"] if not benchmark.empty and "close" in benchmark.columns else None

        result = self.engine.run(strategy, all_data, bench_close)
        return result

    def get_latest_signals(
        self,
        strategy_name: str = "multi_factor",
        sectors: Optional[List[str]] = None,
        top_n: int = 10,
    ) -> List[Dict]:
        """获取最新交易信号"""
        target_sectors = [Sector(s) for s in sectors] if sectors else list(Sector)

        all_data: Dict[str, pd.DataFrame] = {}
        code_names: Dict[str, str] = {}
        for sector in target_sectors:
            cfg = SECTOR_UNIVERSE[sector]
            sector_data = self.fetcher.fetch_sector_data(sector)
            all_data.update(sector_data)
            code_names.update(cfg.stocks)
            for etf_code, etf_name in cfg.etfs.items():
                code_names[f"ETF_{etf_code}"] = etf_name

        strategy_cls = self.STRATEGY_MAP.get(strategy_name)
        if strategy_cls is None:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        strategy = strategy_cls()
        signals = strategy.generate_signals(all_data)

        if signals.empty:
            return []

        latest = signals.iloc[-1]
        buy_signals = latest[latest == 1].index.tolist()

        close_panel = pd.DataFrame({
            code: all_data[code]["close"] for code in all_data if "close" in all_data[code].columns
        })

        result = []
        for code in buy_signals[:top_n]:
            name = code_names.get(code, code)
            price = close_panel[code].iloc[-1] if code in close_panel.columns else 0
            result.append({
                "code": code,
                "name": name,
                "signal": "BUY",
                "price": float(price),
                "strategy": strategy_name,
            })

        return result

    # ─── OpenClaw Pipeline ───

    async def run_research_pipeline(
        self,
        query: str,
    ) -> str:
        """
        通过 OpenClaw Pipeline 串联执行：
        研究分析 -> 策略生成 -> 风险评估
        """
        if not self._client:
            return self._offline_research(query)

        result = await (
            Pipeline(self._client)
            .add_step(
                "researcher",
                "research-bot",
                f"分析以下量化交易问题并提供数据支持: {query}",
            )
            .add_step(
                "strategist",
                "strategy-bot",
                "基于以下研究结果，生成具体的交易策略建议: {researcher}",
            )
            .add_step(
                "risk_analyst",
                "risk-bot",
                "评估以下策略的风险并给出风控建议: {strategist}",
            )
            .run()
        )

        if self._cost_tracker:
            self._cost_tracker.record(result.final_result)

        return result.final_result.content

    async def ask_agent(self, agent_id: str, query: str) -> str:
        """向指定 Agent 提问"""
        if not self._client:
            return f"[Offline] 无法连接 OpenClaw，请确保服务已启动。查询: {query}"

        agent = self._client.get_agent(agent_id)
        result = await agent.execute(query)

        if self._cost_tracker:
            self._cost_tracker.record(result)

        return result.content

    def get_cost_summary(self) -> Dict[str, Any]:
        if self._cost_tracker:
            return self._cost_tracker.summary()
        return {"total_cost_usd": 0, "total_tokens": 0}

    def _offline_research(self, query: str) -> str:
        """离线模式下的本地分析"""
        signals = self.get_latest_signals()
        signal_text = "\n".join(
            f"  - {s['name']}({s['code']}): {s['signal']} @ {s['price']:.2f}"
            for s in signals[:5]
        )
        return (
            f"[离线分析结果]\n"
            f"查询: {query}\n"
            f"当前信号:\n{signal_text}\n"
            f"注意: OpenClaw 未连接，以上为本地策略计算结果。"
        )


# ─── FastAPI 集成 ───

def create_quant_api():
    """创建 FastAPI 应用，提供策略回测和信号查询的 HTTP 接口"""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        logger.error("fastapi not installed")
        return None

    app = FastAPI(
        title="Quant OpenClaw API",
        description="A股多板块量化策略回测与信号系统",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    agent = QuantAgent()

    @app.get("/api/signals/{strategy}")
    async def get_signals(
        strategy: str,
        sectors: Optional[str] = None,
        top_n: int = 10,
    ):
        try:
            sector_list = sectors.split(",") if sectors else None
            signals = agent.get_latest_signals(strategy, sector_list, top_n)
            return {"signals": signals}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/backtest")
    async def run_backtest(
        strategy: str,
        sectors: Optional[str] = None,
        start_date: str = "20230101",
    ):
        try:
            sector_list = sectors.split(",") if sectors else None
            result = agent.run_backtest(strategy, sector_list, start_date)
            return {
                "strategy": result.strategy_name,
                "metrics": result.metrics,
                "summary": result.summary(),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/research")
    async def research(query: str):
        try:
            result = await agent.run_research_pipeline(query)
            return {"result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/sectors")
    async def list_sectors():
        return {
            sector.value: {
                "label": cfg.label,
                "stocks": cfg.stocks,
                "etfs": cfg.etfs,
            }
            for sector, cfg in SECTOR_UNIVERSE.items()
        }

    return app


# 需要在文件顶部额外导入 pandas（仅在需要时使用）
import pandas as pd
