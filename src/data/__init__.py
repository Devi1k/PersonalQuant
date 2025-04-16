#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据模块
提供获取、处理和管理量化交易系统所需的各类市场数据的功能
"""

import logging
from pathlib import Path
from src.data.data_interface import DataInterface

# 设置日志
logger = logging.getLogger(__name__)

# 创建全局数据接口实例
_data_interface = None

def get_data_interface(raw_data_dir="data/raw", processed_data_dir="data/processed"):
    """
    获取数据接口实例（单例模式）
    
    Parameters
    ----------
    raw_data_dir : str, default "data/raw"
        原始数据存储目录
    processed_data_dir : str, default "data/processed"
        处理后数据存储目录
        
    Returns
    -------
    DataInterface
        数据接口实例
    """
    global _data_interface
    
    if _data_interface is None:
        logger.info("初始化数据接口...")
        _data_interface = DataInterface(raw_data_dir, processed_data_dir)
    
    return _data_interface

# 导出核心函数，方便直接调用
def get_etf_history(code, start_date=None, end_date=None, fields=None, adjust="qfq", force_update=False):
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
        
    Returns
    -------
    pandas.DataFrame
        ETF历史行情数据
    """
    return get_data_interface().get_etf_history(code, start_date, end_date, fields, adjust)

def get_industry_etfs():
    """
    获取行业ETF列表
    
    Returns
    -------
    pandas.DataFrame
        行业ETF列表
    """
    return get_data_interface().get_industry_etfs()

def get_technical_indicators(code, indicator_name, params=None, start_date=None, end_date=None):
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
    return get_data_interface().get_technical_indicators(code, indicator_name, params, start_date, end_date)

def get_industry_fund_flow(date=None):
    """
    获取行业资金流向数据
    
    Parameters
    ----------
    date : str, default None
        日期，格式 "YYYY-MM-DD"，默认为今天
        
    Returns
    -------
    pandas.DataFrame
        行业资金流向数据
    """
    return get_data_interface().get_industry_fund_flow(date)

def get_market_sentiment(date=None):
    """
    获取市场情绪指标数据
    
    Parameters
    ----------
    date : str, default None
        日期，格式 "YYYY-MM-DD"，默认为今天
        
    Returns
    -------
    dict
        市场情绪指标数据字典
    """
    return get_data_interface().get_market_sentiment(date)

def update_all_data(start_date=None, end_date=None, etf_codes=None):
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
    return get_data_interface().update_all_data(start_date, end_date, etf_codes)

def get_etf_minute_kline(code, period=5, start_date=None, end_date=None, force_update=False):
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
    force_update : bool, default False
        是否强制更新数据
        
    Returns
    -------
    tuple
        (分钟K线数据DataFrame, 技术指标DataFrame)
    """
    return get_data_interface().get_etf_minute_kline(
        code=code,
        period=period,
        start_date=start_date,
        end_date=end_date,
        use_cache=not force_update,
        force_update=force_update
    )

def update_minute_data(etf_codes, periods=None, start_date=None, end_date=None):
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
    return get_data_interface().update_minute_data(
        etf_codes=etf_codes,
        periods=periods,
        start_date=start_date,
        end_date=end_date
    )