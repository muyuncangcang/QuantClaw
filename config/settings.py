"""
全局配置模块：板块定义、股票池、策略参数、风控阈值
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()


class Sector(str, Enum):
    CPO = "cpo"
    SEMICONDUCTOR = "semiconductor"
    STORAGE_CHIP = "storage_chip"
    AEROSPACE = "aerospace"
    POWER_GRID = "power_grid"
    ROBOT = "robot"
    AI_APPLICATION = "ai_application"
    FUND = "fund"


@dataclass
class SectorConfig:
    name: str
    label: str
    stocks: Dict[str, str]      # code -> name
    etfs: Dict[str, str]        # code -> name
    funds: Dict[str, str] = field(default_factory=dict)  # code -> name
    weight: float = 0.2         # 默认等权


SECTOR_UNIVERSE: Dict[Sector, SectorConfig] = {
    Sector.CPO: SectorConfig(
        name="cpo",
        label="CPO/光模块",
        stocks={
            "300308": "中际旭创",
            "300502": "新易盛",
            "002281": "光迅科技",
            "300394": "天孚通信",
            "688498": "源杰科技",
            "300620": "光库科技",
        },
        etfs={
            "159853": "通信ETF",
        },
        funds={
            "002112": "德邦新星价值灵活配置混合C",
            "002062": "国泰国策驱动混合A",
        },
    ),
    Sector.SEMICONDUCTOR: SectorConfig(
        name="semiconductor",
        label="半导体",
        stocks={
            "688981": "中芯国际",
            "688041": "海光信息",
            "688256": "寒武纪",
            "002371": "北方华创",
            "688012": "中微公司",
            "603501": "韦尔股份",
        },
        etfs={
            "159813": "半导体ETF",
            "512760": "芯片ETF",
            "159516": "半导体设备ETF",
        },
        funds={
            "014320": "德邦半导体产业混合C",
            "025209": "永赢先锋智选混合C",
        },
    ),
    Sector.STORAGE_CHIP: SectorConfig(
        name="storage_chip",
        label="存储芯片",
        stocks={
            "603986": "兆易创新",
            "300223": "北京君正",
            "688396": "华峰测控",
            "603690": "至纯科技",
            "688120": "华海清科",
            "688082": "盛美上海",
        },
        etfs={
            "512760": "芯片ETF",
        },
    ),
    Sector.AEROSPACE: SectorConfig(
        name="aerospace",
        label="航空航天",
        stocks={
            "600118": "中国卫星",
            "600893": "航发动力",
            "600760": "中航沈飞",
            "002025": "航天电器",
            "002179": "中航光电",
            "002414": "高德红外",
        },
        etfs={
            "159227": "航空航天ETF",
            "563380": "航空航天ETF",
            "512670": "国防ETF",
        },
        funds={
            "015790": "永赢高端制造混合C",
        },
    ),
    Sector.POWER_GRID: SectorConfig(
        name="power_grid",
        label="电网设备",
        stocks={
            "601126": "四方股份",
            "600406": "国电南瑞",
            "300341": "麦克奥迪",
            "002028": "思源电气",
            "603169": "兰石重装",
            "688390": "固德威",
        },
        etfs={
            "159870": "电力ETF",
        },
        funds={
            "025833": "天弘电网设备特高压指数C",
            "001665": "平安鑫安混合A",
        },
    ),
    Sector.ROBOT: SectorConfig(
        name="robot",
        label="机器人",
        stocks={},
        etfs={},
        funds={
            "020973": "易方达机器人ETF联接C",
        },
    ),
    Sector.AI_APPLICATION: SectorConfig(
        name="ai_application",
        label="AI应用",
        stocks={},
        etfs={},
        funds={
            "017736": "融通明锐混合C",
        },
    ),
    Sector.FUND: SectorConfig(
        name="fund",
        label="主题基金",
        stocks={},
        etfs={},
        funds={
            "016665": "天弘全球新能源汽车C",
        },
    ),
}


@dataclass
class StrategyParams:
    momentum_window: int = 20
    momentum_holding_period: int = 10
    mean_reversion_window: int = 20
    mean_reversion_zscore_entry: float = -1.5
    mean_reversion_zscore_exit: float = 0.0
    rotation_lookback: int = 20
    rotation_top_n: int = 2
    multi_factor_weights: Dict[str, float] = field(default_factory=lambda: {
        "momentum": 0.3,
        "value": 0.2,
        "volatility": 0.2,
        "quality": 0.15,
        "reversal": 0.15,
    })


@dataclass
class RiskParams:
    max_position_pct: float = 0.15
    max_sector_pct: float = 0.35
    max_drawdown_pct: float = 0.10
    stop_loss_pct: float = 0.05
    trailing_stop_pct: float = 0.08
    var_confidence: float = 0.95
    max_correlation: float = 0.85
    kelly_fraction: float = 0.25


@dataclass
class BacktestParams:
    initial_capital: float = 1_000_000.0
    commission_rate: float = 0.0003
    slippage_pct: float = 0.001
    stamp_tax: float = 0.001        # 印花税（卖出）
    benchmark: str = "000300"        # 沪深300


@dataclass
class Settings:
    tushare_token: str = os.getenv("TUSHARE_TOKEN", "")
    openclaw_ws_url: str = os.getenv("OPENCLAW_GATEWAY_WS_URL", "ws://127.0.0.1:18789/gateway")
    openclaw_api_key: str = os.getenv("OPENCLAW_API_KEY", "")
    data_cache_dir: str = os.getenv("DATA_CACHE_DIR", "./cache")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    strategy: StrategyParams = field(default_factory=StrategyParams)
    risk: RiskParams = field(default_factory=RiskParams)
    backtest: BacktestParams = field(default_factory=BacktestParams)


settings = Settings()
