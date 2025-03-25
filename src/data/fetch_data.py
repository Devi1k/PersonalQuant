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
    update_all_data
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/data_fetch_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(description='数据获取脚本')
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['all', 'etf_list', 'etf_history', 'index', 'fund_flow', 'sentiment'],
                        help='获取模式: all(所有数据), etf_list(ETF列表), etf_history(ETF历史数据), index(指数数据), fund_flow(资金流向), sentiment(市场情绪)')
    parser.add_argument('--code', type=str, default='',
                        help='ETF或指数代码，如 "159915"')
    parser.add_argument('--start', type=str, default='',
                        help='开始日期，格式YYYY-MM-DD')
    parser.add_argument('--end', type=str, default='',
                        help='结束日期，格式YYYY-MM-DD')
    parser.add_argument('--force', action='store_true',
                        help='强制更新数据，忽略缓存')
    
    return parser.parse_args()

def create_directories():
    """创建必要的目录结构"""
    dirs = [
        'data/raw',
        'data/processed',
        'logs'
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

def fetch_etf_history(code, start_date=None, end_date=None, force_update=False):
    """获取ETF历史数据"""
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
    df = get_etf_history(code, start_date, end_date, force_update=force_update)
    
    if df.empty:
        logger.warning(f"未获取到ETF {code} 的历史数据")
        return
    
    logger.info(f"成功获取ETF {code} 历史数据，共 {len(df)} 条记录")
    
    # 打印最近5条记录
    if len(df) > 0:
        logger.info("最近5条记录:")
        recent_data = df.sort_values("date", ascending=False).head(5)
        for i, (_, row) in enumerate(recent_data.iterrows()):
            logger.info(f"{i+1}. {row['date'].strftime('%Y-%m-%d')} - 开盘: {row['open']:.2f}, 最高: {row['high']:.2f}, 最低: {row['low']:.2f}, 收盘: {row['close']:.2f}, 成交量: {row['volume']}")
    
    # 获取技术指标
    macd = get_technical_indicators(code, "macd", start_date=start_date, end_date=end_date)
    if not macd.empty:
        logger.info(f"成功获取ETF {code} MACD指标，共 {len(macd)} 条记录")
        
        # 打印最近3条记录
        if len(macd) > 0:
            logger.info("最近3条MACD记录:")
            recent_macd = macd.sort_values("date", ascending=False).head(3)
            for i, (_, row) in enumerate(recent_macd.iterrows()):
                logger.info(f"{i+1}. {row['date'].strftime('%Y-%m-%d')} - MACD: {row['macd']:.4f}, Signal: {row['macd_signal']:.4f}, Hist: {row['macd_hist']:.4f}")

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

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建目录结构
    create_directories()
    
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
                
            fetch_etf_history(args.code, args.start, args.end, args.force)
            
        elif args.mode == 'fund_flow':
            # 获取行业资金流向数据
            fetch_fund_flow(args.end, args.force)
            
        elif args.mode == 'sentiment':
            # 获取市场情绪指标数据
            fetch_market_sentiment(args.end, args.force)
            
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