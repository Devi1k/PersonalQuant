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
# --- 新增导入 ---
from ..utils.db_utils import get_db_engine, upsert_df_to_sql, load_config
# --- ---

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
        self._minute_kline_cache = {}
        self._minute_indicators_cache = {}

        # --- 初始化数据库引擎 ---
        self.config = load_config()
        self.engine = get_db_engine(self.config)
        # --- ---

        logger.info(f"数据接口模块初始化完成，原始数据目录: {raw_data_dir}, 处理后数据目录: {processed_data_dir}")
        if not self.engine:
             logger.warning("数据库引擎初始化失败，部分数据将无法保存到数据库。")
    
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
                       fields=None, adjust="", use_cache=True, 
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
        # df = self.data_processor.process_industry_fund_flow(df)
        
        # 保存处理后的数据
        # self.data_processor.save_processed_data(df, "industry_fund_flow", date)
        
        return df
    
    def get_market_sentiment(self, date=None, force_update=False, save_to_db=True):
        """
        获取市场情绪指标数据并写入数据库

        Parameters
        ----------
        date : str, default None
            日期，格式 "YYYY-MM-DD"，默认为今天
        force_update : bool, default False
            是否强制更新数据 (当前主要影响数据获取，不影响DB写入)
        save_to_db : bool, default True
            是否将获取并处理后的数据写入数据库

        Returns
        -------
        dict
            市场情绪指标数据字典
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # --- 移除从 CSV 读取的逻辑 ---
        # # 构建文件名
        # file_name = f"market_sentiment_{date.replace('-', '')}.csv"
        # file_path = self.processed_data_dir / file_name
        # # 如果文件存在且不强制更新，直接读取
        # if file_path.exists() and not force_update:
        #     logger.info(f"从文件 {file_path} 读取市场情绪指标数据")
        #     df = pd.read_csv(file_path)
        #     # 转换为字典
        #     return df.iloc[0].to_dict() if not df.empty else {}
        # --- ---
        
        # 获取最新数据
        sentiment_dict = self.data_fetcher.get_market_sentiment(date)
        
        # 处理数据
        sentiment_dict = self.data_processor.process_market_sentiment(sentiment_dict)
        
        # --- 保存到数据库 ---
        if save_to_db and self.engine and sentiment_dict:
            try:
                sentiment_df = pd.DataFrame([sentiment_dict])
                # 确保 trade_date 列存在且为 date 类型
                if 'trade_date' in sentiment_df.columns:
                    sentiment_df['trade_date'] = pd.to_datetime(sentiment_df['trade_date']).dt.date

                    # 选择数据库列
                    db_columns = ['trade_date', 'up_limit_count', 'down_limit_count', 'up_down_ratio'] # 根据 market_sentiment 表结构
                    db_columns_present = [col for col in db_columns if col in sentiment_df.columns]
                    sentiment_df_to_save = sentiment_df[db_columns_present].copy()

                    # 转换数据类型 (upsert_df_to_sql 内部会处理部分类型，但这里可以预处理)
                    if 'up_limit_count' in sentiment_df_to_save.columns:
                        sentiment_df_to_save['up_limit_count'] = sentiment_df_to_save['up_limit_count'].astype('Int64')
                    if 'down_limit_count' in sentiment_df_to_save.columns:
                         sentiment_df_to_save['down_limit_count'] = sentiment_df_to_save['down_limit_count'].astype('Int64')
                    if 'up_down_ratio' in sentiment_df_to_save.columns:
                         sentiment_df_to_save['up_down_ratio'] = pd.to_numeric(sentiment_df_to_save['up_down_ratio'], errors='coerce')

                    # 移除 NaN 主键行
                    sentiment_df_to_save.dropna(subset=['trade_date'], inplace=True)

                    if not sentiment_df_to_save.empty:
                        success = upsert_df_to_sql(sentiment_df_to_save, 'market_sentiment', self.engine, unique_columns=['trade_date'])
                        if success:
                            logger.info(f"市场情绪数据 ({date}) 已成功写入数据库 market_sentiment")
                        else:
                            logger.error(f"市场情绪数据 ({date}) 写入数据库失败")
                    else:
                        logger.warning("没有有效的市场情绪数据可写入数据库。")
                else:
                    logger.error("处理后的市场情绪数据缺少 'trade_date' 列，无法写入数据库。")

            except Exception as e:
                logger.error(f"保存市场情绪数据到数据库时出错: {e}", exc_info=True)
        elif save_to_db and not self.engine:
            logger.error("无法连接数据库，市场情绪数据未保存到数据库。")
        # --- ---

        # --- 移除 CSV 保存 ---
        # pd.DataFrame([sentiment_dict]).to_csv(file_path, index=False)
        # logger.info(f"市场情绪数据已保存至 {file_path}") # 保留日志或移除
        # --- ---
        
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
    
    def get_etf_minute_kline(self, code, period=5, start_date=None, end_date=None, 
                             use_cache=True, force_update=False, with_indicators=True):
        """
        获取ETF分钟级别K线数据并计算技术指标
        
        Parameters
        ----------
        code : str
            ETF代码，例如 "sh510050"
        period : int, default 5
            K线周期，支持 5（5分钟）、15（15分钟）、60（60分钟）
        start_date : str, default None
            开始日期，格式为 "YYYY-MM-DD"，默认为当前日期前7天
        end_date : str, default None
            结束日期，格式为 "YYYY-MM-DD"，默认为当前日期
        use_cache : bool, default True
            是否使用缓存
        force_update : bool, default False
            是否强制更新数据
        with_indicators : bool, default True
            是否计算技术指标
            
        Returns
        -------
        tuple
            (分钟K线数据DataFrame, 技术指标DataFrame)
        """
        # 设置默认日期
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # 缓存键
        cache_key = f"{code}_{period}_{start_date}_{end_date}"
        
        # 创建缓存字典（如果不存在）
        if not hasattr(self, '_minute_kline_cache'):
            self._minute_kline_cache = {}
        if not hasattr(self, '_minute_indicators_cache'):
            self._minute_indicators_cache = {}
            
        # 如果使用缓存且缓存存在且不强制更新
        if use_cache and cache_key in self._minute_kline_cache and not force_update:
            logger.info(f"使用缓存的ETF {code} 的 {period} 分钟K线数据")
            kline_df = self._minute_kline_cache[cache_key]
            
            if with_indicators and cache_key in self._minute_indicators_cache:
                indicators_df = self._minute_indicators_cache[cache_key]
                return kline_df, indicators_df
            elif not with_indicators:
                return kline_df, None
        
        # 获取最新数据
        logger.info(f"从AKShare获取ETF {code} 的 {period} 分钟K线数据...")
        kline_df = self.data_fetcher.get_etf_minute_kline(
            code=code,
            period=period,
            start_date=start_date,
            end_date=end_date
        )
        
        if kline_df.empty:
            logger.warning(f"未能获取到ETF {code} 的 {period} 分钟K线数据")
            return pd.DataFrame(), pd.DataFrame()
        
        # 处理数据
        logger.info(f"处理ETF {code} 的 {period} 分钟K线数据...")
        processed_kline_df = self.data_processor.process_minute_kline(kline_df)
        
        # 更新K线数据缓存
        self._minute_kline_cache[cache_key] = processed_kline_df
        
        # 计算技术指标
        indicators_df = None
        if with_indicators:
            logger.info(f"计算ETF {code} 的 {period} 分钟K线技术指标...")
            indicators_df = self.data_processor.calculate_minute_indicators(processed_kline_df)
            
            # 更新指标数据缓存
            self._minute_indicators_cache[cache_key] = indicators_df
        
        return processed_kline_df, indicators_df
    
    def update_minute_data(self, etf_codes, periods=None, start_date=None, end_date=None):
        """
        更新分钟级别K线数据和技术指标
        
        Parameters
        ----------
        etf_codes : list
            ETF代码列表，例如 ["sh510050", "sh510300"]
        periods : list, default None
            K线周期列表，例如 [5, 15, 60]，默认为 [5, 15, 60]
        start_date : str, default None
            开始日期，格式为 "YYYY-MM-DD"，默认为当前日期前7天
        end_date : str, default None
            结束日期，格式为 "YYYY-MM-DD"，默认为当前日期
            
        Returns
        -------
        dict
            更新结果
        """
        # 设置默认周期
        if periods is None:
            periods = [5, 15, 60]
            
        # 设置默认日期
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        logger.info(f"开始更新ETF分钟级别K线数据，从 {start_date} 到 {end_date}...")
        
        result = {
            "kline_data": 0,
            "indicators_data": 0
        }
        
        for code in etf_codes:
            for period in periods:
                logger.info(f"更新ETF {code} 的 {period} 分钟K线数据...")
                kline_df, indicators_df = self.get_etf_minute_kline(
                    code=code,
                    period=period,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=False,
                    force_update=True
                )
                
                if not kline_df.empty:
                    result["kline_data"] += len(kline_df)
                    
                if indicators_df is not None and not indicators_df.empty:
                    result["indicators_data"] += len(indicators_df)
        
        logger.info(f"ETF分钟级别K线数据更新完成，共更新K线数据 {result['kline_data']} 条，技术指标数据 {result['indicators_data']} 条")
        
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
        
        # 2. 更新ETF历史数据和分钟线数据
        etf_count = 0
        minute_kline_count = 0
        minute_indicators_count = 0
        
        # 设置分钟线数据的起始日期与传入参数一致
        minute_start_date = start_date
        minute_end_date = end_date
        
        if etf_codes:
            for code in etf_codes:
                # 获取日线数据
                df = self.get_etf_history(code, start_date, end_date, 
                                        use_cache=False, force_update=True)
                if not df.empty:
                    etf_count += 1
                
                # 获取分钟线数据
                # 对于分钟线数据，需要添加sh或sz前缀
                if not code.startswith("sh") and not code.startswith("sz"):
                    if code.startswith("6"):  # 上海市场
                        minute_code = f"sh{code}"
                    else:  # 深圳市场
                        minute_code = f"sz{code}"
                else:
                    minute_code = code
                
                # 获取5分钟、15分钟和60分钟K线数据
                for period in [5, 15, 60]:
                    kline_df, indicators_df = self.get_etf_minute_kline(
                        code=minute_code,
                        period=period,
                        start_date=minute_start_date,
                        end_date=minute_end_date,
                        use_cache=False,
                        force_update=True
                    )
                    
                    if not kline_df.empty:
                        minute_kline_count += len(kline_df)
                    
                    if indicators_df is not None and not indicators_df.empty:
                        minute_indicators_count += len(indicators_df)
        
        result["etf_history"] = etf_count
        result["minute_kline"] = minute_kline_count
        result["minute_indicators"] = minute_indicators_count
        
        # 3. 更新指数历史数据
        index_codes = ["sh000001", "sh000300", "sh000905", "sz399001", "sz399006"]  # 上证、沪深300、中证500、深证成指、创业板指
        index_count = 0
        for code in index_codes:
            df = self.get_index_history(code, start_date, end_date, 
                                      use_cache=False, force_update=True)
            if not df.empty:
                index_count += 1
                
            # 获取指数的分钟线数据
            for period in [5, 15, 60]:
                kline_df, indicators_df = self.get_etf_minute_kline(
                    code=code,
                    period=period,
                    start_date=minute_start_date,
                    end_date=minute_end_date,
                    use_cache=False,
                    force_update=True
                )
                
                if not kline_df.empty:
                    minute_kline_count += len(kline_df)
                
                if indicators_df is not None and not indicators_df.empty:
                    minute_indicators_count += len(indicators_df)
                    
        result["index_history"] = index_count
        # 更新分钟线数据计数
        result["minute_kline"] = minute_kline_count
        result["minute_indicators"] = minute_indicators_count
        
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
    etf_history = data_interface.get_etf_history("sz159915", start_date="2023-01-01", end_date="2023-12-31")
    print(f"获取到创业板ETF历史数据 {len(etf_history)} 条记录")
    
    # 测试获取技术指标
    macd = data_interface.get_technical_indicators("159915", "macd", start_date="2023-01-01", end_date="2023-12-31")
    print(f"获取到MACD指标 {len(macd)} 条记录")