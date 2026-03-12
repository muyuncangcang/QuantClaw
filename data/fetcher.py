"""
数据获取模块：从 akshare / tushare 获取A股行情、基金净值、财务数据
支持本地缓存以减少重复请求
"""
from __future__ import annotations

import hashlib
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import akshare as ak
import numpy as np
import pandas as pd
from loguru import logger

from config.settings import SECTOR_UNIVERSE, Sector, SectorConfig, settings


class DataFetcher:
    """统一数据源接口，封装 akshare 获取 A 股日线、财务和 ETF 数据"""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or settings.data_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, prefix: str, code: str, start: str, end: str) -> Path:
        h = hashlib.md5(f"{prefix}_{code}_{start}_{end}".encode()).hexdigest()[:12]
        return self.cache_dir / f"{prefix}_{code}_{h}.pkl"

    def _load_cache(self, path: Path, max_age_hours: int = 12) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        age = datetime.now().timestamp() - path.stat().st_mtime
        if age > max_age_hours * 3600:
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _save_cache(self, path: Path, df: pd.DataFrame) -> None:
        with open(path, "wb") as f:
            pickle.dump(df, f)

    def fetch_stock_daily(
        self,
        code: str,
        start_date: str = "20230101",
        end_date: Optional[str] = None,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """获取个股日线行情（前复权）"""
        end_date = end_date or datetime.now().strftime("%Y%m%d")
        cache_path = self._cache_key("stock_daily", code, start_date, end_date)
        cached = self._load_cache(cache_path)
        if cached is not None:
            logger.debug(f"Cache hit: {code}")
            return cached

        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
                "成交额": "amount", "换手率": "turnover",
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df["code"] = code
            self._save_cache(cache_path, df)
            logger.info(f"Fetched {code}: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {code}: {e}")
            return pd.DataFrame()

    def fetch_etf_daily(
        self,
        code: str,
        start_date: str = "20230101",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取 ETF 日线行情"""
        end_date = end_date or datetime.now().strftime("%Y%m%d")
        cache_path = self._cache_key("etf_daily", code, start_date, end_date)
        cached = self._load_cache(cache_path)
        if cached is not None:
            return cached

        try:
            df = ak.fund_etf_hist_em(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
                "成交额": "amount", "换手率": "turnover",
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df["code"] = code
            self._save_cache(cache_path, df)
            logger.info(f"Fetched ETF {code}: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch ETF {code}: {e}")
            return pd.DataFrame()

    def fetch_fund_daily(
        self,
        code: str,
        start_date: str = "20230101",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取开放式基金净值数据（单位净值 + 累计净值）"""
        end_date = end_date or datetime.now().strftime("%Y%m%d")
        cache_path = self._cache_key("fund_daily", code, start_date, end_date)
        cached = self._load_cache(cache_path)
        if cached is not None:
            logger.debug(f"Cache hit: fund {code}")
            return cached

        try:
            df_nav = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
            df_cum = ak.fund_open_fund_info_em(symbol=code, indicator="累计净值走势")

            df = pd.DataFrame()
            df["date"] = pd.to_datetime(df_nav.iloc[:, 0])
            df["close"] = df_nav.iloc[:, 1].astype(float)
            df["daily_return"] = df_nav.iloc[:, 2].astype(float)
            df["cum_nav"] = df_cum.iloc[:, 1].astype(float)

            df["open"] = df["close"]
            df["high"] = df["close"]
            df["low"] = df["close"]
            df["volume"] = 0.0
            df["amount"] = 0.0

            df = df.set_index("date").sort_index()

            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df.loc[start_dt:end_dt]

            df["code"] = code
            self._save_cache(cache_path, df)
            logger.info(f"Fetched fund {code}: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch fund {code}: {e}")
            return pd.DataFrame()

    def fetch_sector_data(
        self,
        sector: Sector,
        start_date: str = "20230101",
        end_date: Optional[str] = None,
        include_etf: bool = True,
        include_fund: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """获取某板块的全部标的日线数据（股票 + ETF + 基金）"""
        cfg = SECTOR_UNIVERSE[sector]
        result: Dict[str, pd.DataFrame] = {}

        for code, name in cfg.stocks.items():
            df = self.fetch_stock_daily(code, start_date, end_date)
            if not df.empty:
                result[code] = df

        if include_etf:
            for code, name in cfg.etfs.items():
                df = self.fetch_etf_daily(code, start_date, end_date)
                if not df.empty:
                    result[f"ETF_{code}"] = df

        if include_fund:
            for code, name in cfg.funds.items():
                df = self.fetch_fund_daily(code, start_date, end_date)
                if not df.empty:
                    result[f"FUND_{code}"] = df

        return result

    def fetch_all_sectors(
        self,
        start_date: str = "20230101",
        end_date: Optional[str] = None,
    ) -> Dict[Sector, Dict[str, pd.DataFrame]]:
        """获取所有板块数据"""
        all_data = {}
        for sector in Sector:
            logger.info(f"Fetching sector: {sector.value}")
            all_data[sector] = self.fetch_sector_data(sector, start_date, end_date)
        return all_data

    def fetch_index_daily(
        self,
        code: str = "000300",
        start_date: str = "20230101",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取指数日线（基准）"""
        end_date = end_date or datetime.now().strftime("%Y%m%d")
        cache_path = self._cache_key("index", code, start_date, end_date)
        cached = self._load_cache(cache_path)
        if cached is not None:
            return cached

        try:
            df = ak.stock_zh_index_daily(symbol=f"sh{code}")
            df = df.rename(columns={"date": "date"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df = df.loc[start_date:end_date]
            self._save_cache(cache_path, df)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch index {code}: {e}")
            return pd.DataFrame()

    def build_close_panel(
        self, sector_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """将多只标的的收盘价合并为一张 panel 表"""
        frames = {}
        for code, df in sector_data.items():
            if "close" in df.columns:
                frames[code] = df["close"]
        if not frames:
            return pd.DataFrame()
        panel = pd.DataFrame(frames)
        panel = panel.dropna(how="all").ffill()
        return panel
