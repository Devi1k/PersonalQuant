#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据接口模块
提供统一的数据访问接口，供策略和回测模块调用
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

from src.data.akshare_data import AKShareData
from src.data.data_processor import DataProcessor

# 设置日志
logger = logging.getLogger(__name__)

class DataInterface:
    """数据接口类，提供统一的数据访问方法"""
    
    def __init__(self, raw_data_dir="data/raw", processed_data_dir="data/processed"):
        """
        初始化数据接口类
        
        Parameters
        ----------
        raw_data_dir : str
            原始数据存储目录
        processed_data_dir : str
            处理后数据存储目录
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        
        # 确保目录存在
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建数据获取和处理实例
        self.data_fetcher = AKShareData(raw_data_dir, processed_data_dir)
        self.data_processor = DataProcessor(raw_data_dir, processed_data_dir)
        
        # 缓存
        self._etf_list_cache = None
        self._industry_etfs_cache = None
        self._etf_history_cache = {}
        self._index_history_cache = {}
        
        logger.info(f"数据接口模块初始化完成，原始数据目录: {raw_data_dir}, 处理后数据目录: {processed_data_dir}")
    
    def get_etf_list(self, use_cache=True, force_update=False):
        """
        获取ETF基金列表
        
        Parameters
        ----------
        use_cache : bool, default True
            是否使用缓存
        force_update : bool, default False
            是否强制更新数据
            
        Returns
        -------
        pandas.DataFrame
            ETF基金列表
        """
        # 如果使用缓存且缓存存在且不强制更新
        if use_cache and self._etf_list_cache is not None and not force_update:
            logger.info("使用缓存的ETF基金列表")
            return self._etf_list_cache
        
        # 获取最新数据
        etf_list = self.data_fetcher.get_etf_list()
        
        # 更新缓存
        self._etf_list_cache = etf_list
        
        return etf_list
    
    def get_industry_etfs(self, use_cache=True, force_update=False):
        """
        获取行业ETF列表
        
        Parameters
        ----------
        use_cache : bool, default True
            是否使用缓存
        force_update : bool, default False
            是否强制更新数据
            
        Returns
        -------
        pandas.DataFrame
            行业ETF列表
        """
        # 如果使用缓存且缓存存在且不强制更新
        if use_cache and self._industry_etfs_cache is not None and not force_update:
            logger.info("使用缓存的行业ETF列表")
            return self._industry_etfs_cache
        
        # 获取最新数据
        industry_etfs = self.data_fetcher.get_industry_etfs()
        
        # 更新缓存
        self._industry_etfs_cache = industry_etfs
        
        return industry_etfs
    
    def get_etf_history(self, code, start_date=None, end_date=None, 
                       fields=None, adjust="qfq", use_cache=True, 
                       force_update=False, with_indicators=True):
        """
        获取ETF历史行情数据
        
        Parameters
        ----------
        code : str
            ETF代码，如 "159915"
        start_date : str, default None
            开始日期，格式 "YYYY-MM-DD"，默认为一年前
        end_date : str, default None
            结束日期，格式 "YYYY-MM-DD"，默认为今天
        fields : list, default None
            需要的字段列表，默认为全部字段
        adjust : str, default "qfq"
            复权方式，"qfq"为前复权，"hfq"为后复权，None为不复权
        use_cache : bool, default True
            是否使用缓存
        force_update : bool, default False
            是否强制更新数据
        with_indicators : bool, default True
            是否计算技术指标
            
        Returns
        -------
        pandas.DataFrame
            ETF历史行情数据
        """
        # 设置默认日期
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # 缓存键
        cache_key = f"{code}_{start_date}_{end_date}_{adjust}_{with_indicators}"
        
        # 如果使用缓存且缓存存在且不强制更新
        if use_cache and cache_key in self._etf_history_cache and not force_update:
            logger.info(f"使用缓存的ETF {code} 历史数据")
            df = self._etf_history_cache[cache_key]
            
            # 如果指定了字段，筛选字段
            if fields is not None:
                df = df[["date"] + [f for f in fields if f in df.columns]]
            
            return df
        
        # 获取最新数据
        df = self.data_fetcher.get_etf_history(code, start_date, end_date, fields, adjust)
        
        # 处理数据
        df = self.data_processor.process_etf_history(df)
        
        # 计算技术指标
        if with_indicators:
            df = self.data_processor.calculate_technical_indicators(df)
        
        # 更新缓存
        self._etf_history_cache[cache_key] = df
        
        # 如果指定了字段，筛选字段
        if fields is not None:
            df = df[["date"] + [f for f in fields if f in df.columns]]
        
        return df
    
    def get_index_history(self, code, start_date=None, end_date=None, 
                         fields=None, use_cache=True, force_update=False,
                         with_indicators=True):
        """
        获取指数历史行情数据
        
        Parameters
        ----------
        code : str
            指数代码，如 "000300" (沪深300), "000001" (上证指数)
        start_date : str, default None
            开始日期，格式 "YYYY-MM-DD"，默认为一年前
        end_date : str, default None
            结束日期，格式 "YYYY-MM-DD"，默认为今天
        fields : list, default None
            需要的字段列表，默认为全部字段
        use_cache : bool, default True
            是否使用缓存
        force_update : bool, default False
            是否强制更新数据
        with_indicators : bool, default True
            是否计算技术指标
            
        Returns
        -------
        pandas.DataFrame
            指数历史行情数据
        """
        # 设置默认日期
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # 缓存键
        cache_key = f"{code}_{start_date}_{end_date}_{with_indicators}"
        
        # 如果使用缓存且缓存存在且不强制更新
        if use_cache and cache_key in self._index_history_cache and not force_update:
            logger.info(f"使用缓存的指数 {code} 历史数据")
            df = self._index_history_cache[cache_key]
            
            # 如果指定了字段，筛选字段
            if fields is not None:
                df = df[["date"] + [f for f in fields if f in df.columns]]
            
            return df
        
        # 获取最新数据
        df = self.data_fetcher.get_index_history(code, start_date, end_date)
        
        # 处理数据
        df = self.data_processor.process_etf_history(df)  # 可以复用ETF处理逻辑
        
        # 计算技术指标
        if with_indicators:
            df = self.data_processor.calculate_technical_indicators(df)
        
        # 更新缓存
        self._index_history_cache[cache_key] = df
        
        # 如果指定了字段，筛选字段
        if fields is not None:
            df = df[["date"] + [f for f in fields if f in df.columns]]
        
        return df
    
    def get_industry_fund_flow(self, date=None, force_update=False):
        """
        获取行业资金流向数据
        
        Parameters
        ----------
        date : str, default None
            日期，格式 "YYYY-MM-DD"，默认为今天
        force_update : bool, default False
            是否强制更新数据
            
        Returns
        -------
        pandas.DataFrame
            行业资金流向数据
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # 构建文件名
        file_name = f"industry_fund_flow_{date.replace('-', '')}.csv"
        file_path = self.processed_data_dir / file_name
        
        # 如果文件存在且不强制更新，直接读取
        if file_path.exists() and not force_update:
            logger.info(f"从文件 {file_path} 读取行业资金流向数据")
            return pd.read_csv(file_path)
        
        # 获取最新数据
        df = self.data_fetcher.get_fund_flow(date)
        
        # 处理数据
        df = self.data_processor.process_industry_fund_flow(df)
        
        # 保存处理后的数据
        self.data_processor.save_processed_data(df, "industry_fund_flow", date)
        
        return df
    
    def get_market_sentiment(self, date=None, force_update=False):
        """
        获取市场情绪指标数据
        
        Parameters
        ----------
        date : str, default None
            日期，格式 "YYYY-MM-DD"，默认为今天
        force_update : bool, default False
            是否强制更新数据
            
        Returns
        -------
        dict
            市场情绪指标数据字典
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # 构建文件名
        file_name = f"market_sentiment_{date.replace('-', '')}.csv"
        file_path = self.processed_data_dir / file_name
        
        # 如果文件存在且不强制更新，直接读取
        if file_path.exists() and not force_update:
            logger.info(f"从文件 {file_path} 读取市场情绪指标数据")
            df = pd.read_csv(file_path)
            # 转换为字典
            return df.iloc[0].to_dict() if not df.empty else {}
        
        # 获取最新数据
        sentiment_dict = self.data_fetcher.get_market_sentiment(date)
        
        # 处理数据
        sentiment_dict = self.data_processor.process_market_sentiment(sentiment_dict)
        
        # 保存处理后的数据
        pd.DataFrame([sentiment_dict]).to_csv(file_path, index=False)
        
        return sentiment_dict
    
    def get_technical_indicators(self, code, indicator_name, params=None, 
                                start_date=None, end_date=None):
        """
        获取技术指标
        
        Parameters
        ----------
        code : str
            ETF或指数代码
        indicator_name : str
            指标名称，如 "ma", "macd", "rsi", "kdj", "boll", "cci", "atr", "obv"
        params : dict, default None
            指标参数，如 {"period": 14} 表示14日RSI
        start_date : str, default None
            开始日期，格式 "YYYY-MM-DD"，默认为一年前
        end_date : str, default None
            结束日期，格式 "YYYY-MM-DD"，默认为今天
            
        Returns
        -------
        pandas.DataFrame
            技术指标数据
        """
        # 设置默认日期
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # 设置默认参数
        if params is None:
            params = {}
        
        # 判断是ETF还是指数
        if code.startswith("0") or code.startswith("3"):
            # 指数代码通常以0或3开头
            df = self.get_index_history(code, start_date, end_date, with_indicators=True)
        else:
            # ETF代码
            df = self.get_etf_history(code, start_date, end_date, with_indicators=True)
        
        # 根据指标名称筛选列
        indicator_map = {
            "ma": ["ma_5", "ma_10", "ma_20", "ma_60"],
            "macd": ["macd", "macd_signal", "macd_hist"],
            "rsi": ["rsi_14"],
            "kdj": ["kdj_k", "kdj_d", "kdj_j"],
            "boll": ["bb_middle", "bb_upper", "bb_lower"],
            "cci": ["cci_20"],
            "atr": ["atr_14"],
            "obv": ["obv"]
        }
        
        # 获取指标列
        indicator_cols = indicator_map.get(indicator_name.lower(), [])
        
        if not indicator_cols:
            logger.warning(f"未找到指标 {indicator_name}")
            return pd.DataFrame()
        
        # 筛选列
        result = df[["date"] + indicator_cols]
        
        return result
    
    def update_all_data(self, start_date=None, end_date=None, etf_codes=None):
        """
        更新所有数据
        
        Parameters
        ----------
        start_date : str, default None
            开始日期，格式 "YYYY-MM-DD"，默认为一年前
        end_date : str, default None
            结束日期，格式 "YYYY-MM-DD"，默认为今天
        etf_codes : list, default None
            ETF代码列表，默认为None，将获取所有行业ETF
            
        Returns
        -------
        dict
            更新结果
        """
        # 设置默认日期
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        logger.info(f"开始更新从 {start_date} 到 {end_date} 的所有数据...")
        
        result = {}
        
        # 1. 更新ETF列表
        industry_etfs = self.get_industry_etfs(use_cache=False, force_update=True)
        result["industry_etfs"] = len(industry_etfs) if not industry_etfs.empty else 0
        
        # 如果未指定ETF代码，则使用获取的行业ETF
        if etf_codes is None and not industry_etfs.empty:
            etf_codes = industry_etfs["code"].tolist()
        
        # 2. 更新ETF历史数据
        etf_count = 0
        if etf_codes:
            for code in etf_codes:
                df = self.get_etf_history(code, start_date, end_date, 
                                        use_cache=False, force_update=True)
                if not df.empty:
                    etf_count += 1
        result["etf_history"] = etf_count
        
        # 3. 更新指数历史数据
        index_codes = ["000001", "000300", "000905", "399001", "399006"]  # 上证、沪深300、中证500、深证成指、创业板指
        index_count = 0
        for code in index_codes:
            df = self.get_index_history(code, start_date, end_date, 
                                      use_cache=False, force_update=True)
            if not df.empty:
                index_count += 1
        result["index_history"] = index_count
        
        # 4. 更新最新的行业资金流向
        fund_flow = self.get_industry_fund_flow(force_update=True)
        result["fund_flow"] = len(fund_flow) if not fund_flow.empty else 0
        
        # 5. 更新最新的市场情绪指标
        sentiment = self.get_market_sentiment(force_update=True)
        result["sentiment"] = len(sentiment) if sentiment else 0
        
        logger.info("所有数据更新完成")
        
        return result


# 测试代码
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建DataInterface实例
    data_interface = DataInterface()
    
    # 测试获取ETF列表
    etf_list = data_interface.get_etf_list()
    print(f"获取到 {len(etf_list)} 个ETF")
    
    # 测试获取行业ETF列表
    industry_etfs = data_interface.get_industry_etfs()
    print(f"获取到 {len(industry_etfs)} 个行业ETF")
    
    # 测试获取ETF历史数据 (以创业板ETF为例)
    etf_history = data_interface.get_etf_history("159915", start_date="2023-01-01", end_date="2023-12-31")
    print(f"获取到创业板ETF历史数据 {len(etf_history)} 条记录")
    
    # 测试获取技术指标
    macd = data_interface.get_technical_indicators("159915", "macd", start_date="2023-01-01", end_date="2023-12-31")
    print(f"获取到MACD指标 {len(macd)} 条记录")