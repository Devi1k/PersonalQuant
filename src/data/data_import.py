#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据导入模块 - 将CSV数据导入MySQL数据库
"""

import os
import sys
import logging
import argparse
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_import.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("data_import")

# 数据库连接配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'password',  # 请修改为实际密码
    'database': 'quant_db',
    'charset': 'utf8mb4'
}

def get_db_engine():
    """创建数据库连接引擎"""
    conn_str = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?charset={DB_CONFIG['charset']}"
    return create_engine(conn_str)

def import_etf_list(file_path, update_date=None):
    """
    导入ETF列表数据
    
    Args:
        file_path (str): CSV文件路径
        update_date (str, optional): 更新日期，格式YYYYMMDD。默认从文件名提取。
    
    Returns:
        int: 导入的记录数
    """
    try:
        # 如果未提供更新日期，尝试从文件名提取
        if update_date is None:
            filename = os.path.basename(file_path)
            # 假设文件名格式为etf_list_YYYYMMDD.csv
            date_part = filename.split('_')[-1].split('.')[0]
            update_date = datetime.strptime(date_part, '%Y%m%d').strftime('%Y-%m-%d')
        else:
            update_date = datetime.strptime(update_date, '%Y%m%d').strftime('%Y-%m-%d')
        
        logger.info(f"开始导入ETF列表数据: {file_path}, 更新日期: {update_date}")
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 重命名列以匹配数据库表结构
        column_mapping = {
            '代码': 'code',
            '名称': 'name',
            '最新价': 'latest_price',
            'change': 'price_change',
            'pct_change': 'pct_change',
            '买入': 'buy_price',
            '卖出': 'sell_price',
            'pre_close': 'pre_close',
            '今开': 'open_price',
            '最高': 'high_price',
            '最低': 'low_price',
            'volume': 'volume',
            'amount': 'amount'
        }
        df = df.rename(columns=column_mapping)
        
        # 添加更新日期
        df['update_date'] = update_date
        
        # 连接数据库
        engine = get_db_engine()
        
        # 导入数据 - 使用REPLACE模式处理重复记录
        df.to_sql('etf_list', engine, if_exists='replace', index=False)
        
        record_count = len(df)
        logger.info(f"成功导入ETF列表数据: {record_count}条记录")
        return record_count
        
    except Exception as e:
        logger.error(f"导入ETF列表数据失败: {str(e)}")
        raise

def import_etf_indicators(file_path, etf_code=None):
    """
    导入ETF指标数据
    
    Args:
        file_path (str): CSV文件路径
        etf_code (str, optional): ETF代码。默认从文件名提取。
    
    Returns:
        int: 导入的记录数
    """
    try:
        # 如果未提供ETF代码，尝试从文件名提取
        if etf_code is None:
            filename = os.path.basename(file_path)
            # 假设文件名格式为etf_XXXXXX_with_indicators_YYYYMMDD.csv
            etf_code = filename.split('_')[1]
        
        logger.info(f"开始导入ETF指标数据: {file_path}, ETF代码: {etf_code}")
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 添加ETF代码
        df['etf_code'] = etf_code
        
        # 确保日期格式正确
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df = df.rename(columns={'date': 'trade_date'})
        
        # 连接数据库
        engine = get_db_engine()
        
        # 导入数据 - 使用REPLACE模式处理重复记录
        df.to_sql('etf_indicators', engine, if_exists='append', index=False, 
                  chunksize=1000, method='multi')
        
        record_count = len(df)
        logger.info(f"成功导入ETF指标数据: {record_count}条记录")
        return record_count
        
    except Exception as e:
        logger.error(f"导入ETF指标数据失败: {str(e)}")
        raise

def import_market_sentiment(file_path):
    """
    导入市场情绪数据
    
    Args:
        file_path (str): CSV文件路径
    
    Returns:
        int: 导入的记录数
    """
    try:
        logger.info(f"开始导入市场情绪数据: {file_path}")
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 确保日期格式正确
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df = df.rename(columns={'date': 'trade_date'})
        
        # 连接数据库
        engine = get_db_engine()
        
        # 导入数据 - 使用REPLACE模式处理重复记录
        df.to_sql('market_sentiment', engine, if_exists='append', index=False)
        
        record_count = len(df)
        logger.info(f"成功导入市场情绪数据: {record_count}条记录")
        return record_count
        
    except Exception as e:
        logger.error(f"导入市场情绪数据失败: {str(e)}")
        raise

def import_industry_fund_flow(file_path):
    """
    导入行业资金流向数据
    
    Args:
        file_path (str): CSV文件路径
    
    Returns:
        int: 导入的记录数
    """
    try:
        logger.info(f"开始导入行业资金流向数据: {file_path}")
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 重命名列以匹配数据库表结构
        column_mapping = {
            '序号': 'id',
            '名称': 'industry_name',
            '今日涨跌幅': 'price_change_pct',
            '今日主力净流入-净额': 'main_net_inflow',
            '今日主力净流入-净占比': 'main_net_inflow_pct',
            '今日超大单净流入-净额': 'super_large_net_inflow',
            '今日超大单净流入-净占比': 'super_large_net_inflow_pct',
            '今日大单净流入-净额': 'large_net_inflow',
            '今日大单净流入-净占比': 'large_net_inflow_pct',
            '今日中单净流入-净额': 'medium_net_inflow',
            '今日中单净流入-净占比': 'medium_net_inflow_pct',
            '今日小单净流入-净额': 'small_net_inflow',
            '今日小单净流入-净占比': 'small_net_inflow_pct',
            '今日主力净流入最大股': 'top_stock',
            'date': 'trade_date'
        }
        df = df.rename(columns=column_mapping)
        
        # 确保日期格式正确
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
        
        # 连接数据库
        engine = get_db_engine()
        
        # 导入数据 - 使用REPLACE模式处理重复记录
        df.to_sql('industry_fund_flow', engine, if_exists='append', index=False, 
                  chunksize=1000, method='multi')
        
        record_count = len(df)
        logger.info(f"成功导入行业资金流向数据: {record_count}条记录")
        return record_count
        
    except Exception as e:
        logger.error(f"导入行业资金流向数据失败: {str(e)}")
        raise

def init_database():
    """初始化数据库结构"""
    try:
        logger.info("开始初始化数据库结构")
        
        # 读取SQL脚本
        with open('schema/quant_db.sql', 'r', encoding='utf-8') as f:
            sql_script = f.read()
        
        # 连接数据库
        engine = create_engine(f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}?charset={DB_CONFIG['charset']}")
        
        # 执行SQL脚本
        with engine.connect() as conn:
            # 分割SQL语句并执行
            for statement in sql_script.split(';'):
                if statement.strip():
                    conn.execute(text(statement))
        
        logger.info("成功初始化数据库结构")
        return True
        
    except Exception as e:
        logger.error(f"初始化数据库结构失败: {str(e)}")
        raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='量化数据导入工具')
    parser.add_argument('--init', action='store_true', help='初始化数据库')
    parser.add_argument('--etf-list', type=str, help='ETF列表CSV文件路径')
    parser.add_argument('--etf-indicators', type=str, help='ETF指标CSV文件路径')
    parser.add_argument('--etf-code', type=str, help='ETF代码(用于指标导入)')
    parser.add_argument('--market-sentiment', type=str, help='市场情绪CSV文件路径')
    parser.add_argument('--industry-flow', type=str, help='行业资金流向CSV文件路径')
    parser.add_argument('--date', type=str, help='数据日期(YYYYMMDD格式)')
    
    args = parser.parse_args()
    
    try:
        # 创建日志目录
        os.makedirs('logs', exist_ok=True)
        
        # 初始化数据库
        if args.init:
            init_database()
        
        # 导入ETF列表
        if args.etf_list:
            import_etf_list(args.etf_list, args.date)
        
        # 导入ETF指标
        if args.etf_indicators:
            import_etf_indicators(args.etf_indicators, args.etf_code)
        
        # 导入市场情绪
        if args.market_sentiment:
            import_market_sentiment(args.market_sentiment)
        
        # 导入行业资金流向
        if args.industry_flow:
            import_industry_fund_flow(args.industry_flow)
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()