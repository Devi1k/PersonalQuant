#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据获取脚本
用于获取和更新量化交易系统所需的各类市场数据
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data import (
    get_etf_history,
    get_industry_etfs,
    get_technical_indicators,
    get_industry_fund_flow,
    get_market_sentiment,
    update_all_data,
    get_etf_minute_kline,
    update_minute_data
)

# 设置日志
# 确保使用绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"data_fetch_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(description='数据获取脚本')
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['all', 'etf_list', 'etf_history', 'index', 'fund_flow', 'sentiment', 'minute_kline', 'minute_update'],
                        help='获取模式: all(所有数据), etf_list(ETF列表), etf_history(ETF历史数据), index(指数数据), fund_flow(资金流向), sentiment(市场情绪), minute_kline(分钟K线数据), minute_update(更新分钟数据)')
    parser.add_argument('--code', type=str, default='',
                        help='ETF或指数代码，如 "159915"')
    parser.add_argument('--start', type=str, default='',
                        help='开始日期，格式YYYY-MM-DD')
    parser.add_argument('--end', type=str, default='',
                        help='结束日期，格式YYYY-MM-DD')
    parser.add_argument('--force', action='store_true',
                        help='强制更新数据，忽略缓存')
    parser.add_argument('--period', type=int, default=5, choices=[5, 15, 60],
                        help='K线周期（分钟），支持5、15、60，默认为5')
    parser.add_argument('--codes', type=str, nargs='+', default=[],
                        help='ETF代码列表，用于批量更新分钟K线数据')
    parser.add_argument('--periods', type=int, nargs='+', default=[],
                        help='K线周期列表，用于批量更新分钟K线数据')
    
    return parser.parse_args()

def create_directories():
    """创建必要的目录结构"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    
    dirs = [
        os.path.join(project_root, 'data/raw'),
        os.path.join(project_root, 'data/processed'),
        os.path.join(project_root, 'logs')
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.debug(f"确保目录存在: {d}")

def fetch_etf_list():
    """获取ETF列表"""
    logger.info("开始获取ETF列表...")
    
    # 获取所有ETF
    etf_list = get_industry_etfs()
    
    if etf_list.empty:
        logger.warning("未获取到ETF列表")
        return
    
    logger.info(f"成功获取ETF列表，共 {len(etf_list)} 个ETF")
    
    # 打印前10个ETF
    if len(etf_list) > 0:
        logger.info("ETF列表前10个:")
        for i, (_, row) in enumerate(etf_list.head(10).iterrows()):
            logger.info(f"{i+1}. {row['code']} - {row['name']}")

def fetch_etf_history(code, start_date=None, end_date=None, fields=None, adjust="qfq", force_update=False):
    """
    获取ETF历史数据
    
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
    force_update : bool, default False
        是否强制更新数据
    """
    if not code:
        logger.error("未指定ETF代码")
        return
    
    # 设置默认日期
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"开始获取ETF {code} 从 {start_date} 到 {end_date} 的历史数据...")
    
    # 获取ETF历史数据
    df = get_etf_history(code, start_date, end_date, fields, adjust, force_update=force_update)
    
    if df.empty:
        logger.warning(f"未获取到ETF {code} 的历史数据")
        return
    
    logger.info(f"成功获取ETF {code} 历史数据，共 {len(df)} 条记录")
    
    # 打印最近5条记录
    if len(df) > 0:
        logger.info("最近5条记录:")
        recent_data = df.sort_values("trade_date", ascending=False).head(5)
        for i, (_, row) in enumerate(recent_data.iterrows()):
            logger.info(f"{i+1}. {row['trade_date'].strftime('%Y-%m-%d')} - 开盘: {row['open']:.2f}, 最高: {row['high']:.2f}, 最低: {row['low']:.2f}, 收盘: {row['close']:.2f}, 成交量: {row['volume']}")
    
    # 打印MACD指标信息
    if 'macd' in df.columns:
        logger.info("最近3条MACD记录:")
        recent_macd = df.sort_values("trade_date", ascending=False).head(3)
        for i, (_, row) in enumerate(recent_macd.iterrows()):
            logger.info(f"{i+1}. {row['trade_date'].strftime('%Y-%m-%d')} - MACD: {row['macd']:.4f}, Signal: {row['macd_signal']:.4f}, Hist: {row['macd_hist']:.4f}")

def fetch_fund_flow(date=None, force_update=False):
    """获取行业资金流向数据"""
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"开始获取 {date} 的行业资金流向数据...")
    
    # 获取行业资金流向数据
    df = get_industry_fund_flow(date)
    
    if df.empty:
        logger.warning(f"未获取到 {date} 的行业资金流向数据")
        return
    
    logger.info(f"成功获取行业资金流向数据，共 {len(df)} 条记录")
    
    # 打印前5条记录
    if len(df) > 0:
        logger.info("行业资金流向前5条记录:")
        for i, (_, row) in enumerate(df.head(5).iterrows()):
            # 根据实际列名调整
            if '行业名称' in df.columns and '净流入额' in df.columns:
                logger.info(f"{i+1}. {row['行业名称']} - 净流入额: {row['净流入额']}")
            else:
                # 打印前两列作为替代
                cols = df.columns.tolist()
                if len(cols) >= 2:
                    logger.info(f"{i+1}. {row[cols[0]]} - {cols[1]}: {row[cols[1]]}")

def fetch_market_sentiment(date=None, force_update=False):
    """获取市场情绪指标数据"""
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"开始获取 {date} 的市场情绪指标数据...")
    
    # 获取市场情绪指标数据
    sentiment = get_market_sentiment(date)
    
    if not sentiment:
        logger.warning(f"未获取到 {date} 的市场情绪指标数据")
        return
    
    logger.info(f"成功获取市场情绪指标数据，共 {len(sentiment)} 个指标")
    
    # 打印所有指标
    logger.info("市场情绪指标:")
    for key, value in sentiment.items():
        if key != 'date':
            logger.info(f"{key}: {value}")

def fetch_minute_kline(code, period=5, start_date=None, end_date=None, force_update=False):
    """
    获取ETF分钟级别K线数据
    
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
    """
    if not code:
        logger.error("未指定ETF代码")
        return
    
    # 设置默认日期
    if not start_date:
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"开始获取ETF {code} 从 {start_date} 到 {end_date} 的 {period} 分钟K线数据...")
    
    # 获取ETF分钟级别K线数据
    kline_df, indicators_df = get_etf_minute_kline(
        code=code,
        period=period,
        start_date=start_date,
        end_date=end_date,
        force_update=force_update
    )
    
    if kline_df.empty:
        logger.warning(f"未获取到ETF {code} 的 {period} 分钟K线数据")
        return
    
    logger.info(f"成功获取ETF {code} 的 {period} 分钟K线数据，共 {len(kline_df)} 条记录")
    
    # 打印最近5条K线记录
    if len(kline_df) > 0:
        logger.info(f"最近5条 {period} 分钟K线记录:")
        # 按日期和时间排序
        recent_data = kline_df.sort_values(["trade_date", "trade_time"], ascending=[False, False]).head(5)
        for i, (_, row) in enumerate(recent_data.iterrows()):
            date_str = row['trade_date'].strftime('%Y-%m-%d') if hasattr(row['trade_date'], 'strftime') else row['trade_date']
            time_str = row['trade_time'].strftime('%H:%M:%S') if hasattr(row['trade_time'], 'strftime') else row['trade_time']
            logger.info(f"{i+1}. {date_str} {time_str} - 开盘: {row['open']:.2f}, 最高: {row['high']:.2f}, 最低: {row['low']:.2f}, 收盘: {row['close']:.2f}, 成交量: {row['volume']}")
    
    # 打印技术指标信息
    if indicators_df is not None and not indicators_df.empty:
        logger.info(f"成功计算ETF {code} 的 {period} 分钟K线技术指标，共 {len(indicators_df)} 条记录")
        
        # 打印最近3条MACD指标记录
        if 'macd' in indicators_df.columns:
            logger.info(f"最近3条 {period} 分钟MACD记录:")
            recent_macd = indicators_df.sort_values(["trade_date", "trade_time"], ascending=[False, False]).head(3)
            for i, (_, row) in enumerate(recent_macd.iterrows()):
                date_str = row['trade_date'].strftime('%Y-%m-%d') if hasattr(row['trade_date'], 'strftime') else row['trade_date']
                time_str = row['trade_time'].strftime('%H:%M:%S') if hasattr(row['trade_time'], 'strftime') else row['trade_time']
                logger.info(f"{i+1}. {date_str} {time_str} - MACD: {row['macd']:.4f}, Signal: {row['macd_signal']:.4f}, Hist: {row['macd_hist']:.4f}")

def update_minute_kline_data(etf_codes, periods=None, start_date=None, end_date=None):
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
    """
    if not etf_codes:
        logger.error("未指定ETF代码列表")
        return
    
    # 设置默认日期
    if not start_date:
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # 设置默认周期
    if not periods:
        periods = [5, 15, 60]
    
    logger.info(f"开始更新ETF分钟级别K线数据，从 {start_date} 到 {end_date}...")
    logger.info(f"ETF代码列表: {etf_codes}")
    logger.info(f"K线周期列表: {periods}")
    
    # 更新分钟级别K线数据
    result = update_minute_data(
        etf_codes=etf_codes,
        periods=periods,
        start_date=start_date,
        end_date=end_date
    )
    
    logger.info("分钟级别K线数据更新结果:")
    for key, value in result.items():
        logger.info(f"{key}: {value}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建目录结构
    create_directories()
    args.start = "2018-01-01"
    args.end = "2025-04-15"
    args.mode = "all"
    logger.info(f"启动数据获取脚本，获取模式: {args.mode}")
    try:
        if args.mode == 'all':
            # 更新所有数据
            start_date = args.start if args.start else (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            end_date = args.end if args.end else datetime.now().strftime("%Y-%m-%d")
            
            logger.info(f"开始更新从 {start_date} 到 {end_date} 的所有数据...")
            
            result = update_all_data(start_date, end_date)
            
            logger.info("数据更新结果:")
            for key, value in result.items():
                logger.info(f"{key}: {value}")
                
        elif args.mode == 'etf_list':
            # 获取ETF列表
            fetch_etf_list()
            
        elif args.mode == 'etf_history':
            # 获取ETF历史数据
            if not args.code:
                logger.error("获取ETF历史数据需要指定ETF代码")
                return 1
            for code in args.code:
                fetch_etf_history(code=code, start_date=args.start, end_date=args.end, force_update=args.force)
            
        elif args.mode == 'fund_flow':
            # 获取行业资金流向数据
            fetch_fund_flow(args.end, args.force)
            
        elif args.mode == 'sentiment':
            # 获取市场情绪指标数据
            fetch_market_sentiment(args.end, args.force)
            
        elif args.mode == 'minute_kline':
            # 获取分钟级别K线数据
            if not args.code:
                logger.error("获取分钟级别K线数据需要指定ETF代码")
                return 1
            
            fetch_minute_kline(
                code=args.code,
                period=args.period,
                start_date=args.start,
                end_date=args.end,
                force_update=args.force
            )
            
        elif args.mode == 'minute_update':
            # 更新分钟级别K线数据
            if not args.code:
                logger.error("更新分钟级别K线数据需要指定ETF代码列表")
                return 1
            
            update_minute_kline_data(
                etf_codes=args.code,
                periods=args.periods if args.periods else None,
                start_date=args.start,
                end_date=args.end
            )
            
        else:
            logger.error(f"不支持的获取模式: {args.mode}")
            return 1
            
    except Exception as e:
        logger.exception(f"数据获取过程中出现错误: {e}")
        return 1
        
    logger.info("数据获取脚本运行完成")
    return 0

if __name__ == "__main__":
    sys.exit(main())