# Quant OpenClaw - A股多板块量化策略系统

基于 [OpenClaw](https://github.com/openclaw) AI Agent 框架的 A 股量化交易策略系统，聚焦 **CPO/光模块、半导体、存储芯片、航空航天、电网设备** 五大新兴科技板块。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenClaw AI Agent Layer                    │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │Research   │→│Strategy Bot  │→│Risk Analysis Bot     │   │
│  │Bot        │  │(信号生成)     │  │(风控预警)             │   │
│  └──────────┘  └──────────────┘  └──────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                       Strategy Engine                        │
│  ┌────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐     │
│  │动量策略 │ │均值回归   │ │板块轮动   │ │多因子选股     │     │
│  └────────┘ └──────────┘ └──────────┘ └──────────────┘     │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │回测引擎   │  │风控管理       │  │可视化分析            │   │
│  └──────────┘  └──────────────┘  └──────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │数据获取   │  │特征工程       │  │因子计算              │   │
│  │(akshare) │  │(20+技术指标) │  │(动量/波动/量价/反转) │   │
│  └──────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 核心功能

### 四大交易策略
| 策略 | 原理 | 适用板块 |
|------|------|---------|
| **动量策略** | 追涨强势股，波动率调整后的截面动量 | CPO、半导体 |
| **均值回归** | Z-Score 偏离后回归，布林带+量能确认 | 电网设备、航空航天 |
| **板块轮动** | 综合评分选择最强板块，月度再平衡 | 全板块 |
| **多因子选股** | 5因子合成(动量/价值/波动/质量/反转)，IC加权 | 全板块 |

### 风控体系
- **事前**: 单票仓位上限15%、板块上限35%、相关性过滤(>0.85)
- **事中**: 5%止损、8%移动止盈、VaR(历史/参数/Cornish-Fisher)
- **事后**: Alpha/Beta 分解、信息比率、因子归因

### 绩效评估
Sharpe / Sortino / Calmar / 最大回撤 / VaR / CVaR / IC/IR / 胜率 / 盈亏比

## 覆盖板块

| 板块 | 核心标的 | ETF |
|------|---------|-----|
| CPO/光模块 | 中际旭创、新易盛、光迅科技、天孚通信 | 通信ETF(159853) |
| 半导体 | 中芯国际、海光信息、寒武纪、北方华创 | 半导体ETF(159813)、芯片ETF(512760) |
| 存储芯片 | 兆易创新、北京君正、华峰测控 | 芯片ETF(512760) |
| 航空航天 | 中国卫星、航发动力、中航沈飞 | 航空航天ETF(159227)、国防ETF(512670) |
| 电网设备 | 国电南瑞、思源电气、四方股份 | 电力ETF(159870) |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Token

# 3. 运行单策略回测
python main.py backtest --strategy momentum --sectors cpo,semiconductor

# 4. 对比所有策略
python main.py compare

# 5. 获取最新交易信号
python main.py signals --strategy multi_factor

# 6. 启动API服务
python main.py api --port 8000

# 7. OpenClaw Agent 交互
python main.py agent --query "分析CPO板块近期走势并给出交易建议"
```

## 运行测试

```bash
pytest tests/ -v --tb=short
```

## 技术栈

- **AI Agent**: OpenClaw SDK 2.1 (Pipeline/Structured Output/Cost Tracking)
- **数据源**: akshare (A股行情/ETF/指数)
- **策略框架**: 自研事件驱动回测引擎
- **因子计算**: pandas + numpy + scipy
- **风控模型**: VaR(3种方法) + Kelly Criterion + 相关性监控
- **可视化**: matplotlib + seaborn
- **API服务**: FastAPI + uvicorn
- **测试**: pytest + pytest-asyncio

## 项目结构

```
quant_openclaw/
├── config/settings.py        # 板块定义、股票池、策略/风控/回测参数
├── data/
│   ├── fetcher.py            # A股/ETF/指数数据获取，支持缓存
│   └── processor.py          # 20+技术指标、5类量化因子
├── strategy/
│   ├── base.py               # 策略基类(信号枚举/权重计算接口)
│   ├── momentum.py           # 波动率调整截面动量策略
│   ├── mean_reversion.py     # Z-Score均值回归(布林带+量能)
│   ├── sector_rotation.py    # 多维度板块轮动评分
│   └── multi_factor.py       # 5因子合成+IC动态加权
├── backtest/
│   ├── engine.py             # 事件驱动回测(手续费/滑点/印花税)
│   └── metrics.py            # 15+绩效指标(含Alpha/Beta分解)
├── risk/
│   └── manager.py            # 三层风控(事前/事中/事后)
├── agent/
│   └── openclaw_agent.py     # OpenClaw Agent集成(Pipeline/API)
├── visualization/
│   └── plotter.py            # 6类图表(净值/回撤/因子IC/热力图)
├── tests/                    # 单元测试
├── main.py                   # CLI入口
└── requirements.txt
```

## License

MIT
