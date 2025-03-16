#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
量化交易系统主程序
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/quant_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(description='量化交易系统')
    parser.add_argument('--mode', type=str, default='backtest', 
                        choices=['backtest', 'paper', 'live'],
                        help='运行模式: backtest(回测), paper(模拟), live(实盘)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--start', type=str, default='',
                        help='回测开始日期，格式YYYY-MM-DD')
    parser.add_argument('--end', type=str, default='',
                        help='回测结束日期，格式YYYY-MM-DD')
    
    return parser.parse_args()

def create_directories():
    """创建必要的目录结构"""
    dirs = [
        'data/raw',
        'data/processed',
        'logs',
        'results',
        'results/backtest',
        'results/signals'
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.debug(f"确保目录存在: {d}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建目录结构
    create_directories()
    
    logger.info(f"启动量化交易系统，运行模式: {args.mode}")
    
    try:
        if args.mode == 'backtest':
            from src.backtesting.backtest_engine import run_backtest
            run_backtest(args.config, args.start, args.end)
        elif args.mode == 'paper':
            from src.trading.paper_trading import run_paper_trading
            run_paper_trading(args.config)
        elif args.mode == 'live':
            from src.trading.live_trading import run_live_trading
            run_live_trading(args.config)
        else:
            logger.error(f"不支持的运行模式: {args.mode}")
            return 1
            
    except Exception as e:
        logger.exception(f"运行过程中出现错误: {e}")
        return 1
        
    logger.info("量化交易系统运行完成")
    return 0

if __name__ == "__main__":
    sys.exit(main())