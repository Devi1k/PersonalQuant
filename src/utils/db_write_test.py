#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据库写入测试模块
用于验证数据库写入功能是否正常
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
from sqlalchemy import text

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.akshare_data import AKShareData
from src.utils.db_utils import get_db_engine, load_config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DBWriteTest:
    """数据库写入测试类"""
    
    def __init__(self):
        """初始化数据库写入测试类"""
        self.config = load_config()
        self.db_engine = get_db_engine(self.config)
        self.akshare_data = AKShareData()
        
        if self.db_engine is None:
            logger.error("数据库引擎初始化失败，请检查配置")
            sys.exit(1)
        
        logger.info("数据库写入测试模块初始化完成")
    
    def test_connection(self):
        """测试数据库连接"""
        try:
            with self.db_engine.connect() as conn:
                logger.info("数据库连接测试成功")
                return True
        except Exception as e:
            logger.error(f"数据库连接测试失败: {e}")
            return False
    
    def test_etf_list_write(self):
        """测试ETF列表写入"""
        try:
            logger.info("开始测试ETF列表写入...")
            etf_list = self.akshare_data.get_etf_list(save=True)
            
            if etf_list.empty:
                logger.error("获取ETF列表失败")
                return False
            
            logger.info(f"成功获取并写入 {len(etf_list)} 个ETF基金信息")
            
            # 验证数据是否写入数据库
            with self.db_engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM etf_list")).fetchone()
                count = result[0]
                logger.info(f"数据库中共有 {count} 条ETF基金记录")
                
                if count > 0:
                    logger.info("ETF列表写入测试成功")
                    return True
                else:
                    logger.error("ETF列表写入测试失败，数据库中无记录")
                    return False
        except Exception as e:
            logger.error(f"ETF列表写入测试失败: {e}")
            return False
    
    def test_etf_indicators_write(self):
        """测试ETF指标数据写入"""
        try:
            logger.info("开始测试ETF指标数据写入...")
            
            # 以创业板ETF为例
            etf_code = "159915"
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            end_date = datetime.now().strftime("%Y-%m-%d")
            
            etf_history = self.akshare_data.get_etf_history(
                etf_code, 
                start_date=start_date, 
                end_date=end_date,
                save=True
            )
            
            if etf_history.empty:
                logger.error(f"获取ETF {etf_code} 历史数据失败")
                return False
            
            logger.info(f"成功获取并写入 {len(etf_history)} 条ETF {etf_code} 历史数据")
            
            # 验证数据是否写入数据库
            with self.db_engine.connect() as conn:
                query = text(f"SELECT COUNT(*) FROM etf_indicators WHERE etf_code = '{etf_code}'")
                result = conn.execute(query).fetchone()
                count = result[0]
                logger.info(f"数据库中共有 {count} 条ETF {etf_code} 历史数据记录")
                
                if count > 0:
                    logger.info("ETF指标数据写入测试成功")
                    return True
                else:
                    logger.error("ETF指标数据写入测试失败，数据库中无记录")
                    return False
        except Exception as e:
            logger.error(f"ETF指标数据写入测试失败: {e}")
            return False
    
    def test_market_sentiment_write(self):
        """测试市场情绪数据写入"""
        try:
            logger.info("开始测试市场情绪数据写入...")
            
            # 获取当前日期
            date = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d")
            
            # 获取市场情绪数据
            market_sentiment = self.akshare_data.get_market_sentiment(date=date, save=True)
            
            if not market_sentiment:
                logger.error("获取市场情绪数据失败")
                return False
            
            logger.info(f"成功获取并写入市场情绪数据")
            
            # 验证数据是否写入数据库
            with self.db_engine.connect() as conn:
                query = text("SELECT COUNT(*) FROM market_sentiment")
                result = conn.execute(query).fetchone()
                count = result[0]
                logger.info(f"数据库中共有 {count} 条市场情绪数据记录")
                
                if count > 0:
                    logger.info("市场情绪数据写入测试成功")
                    return True
                else:
                    logger.error("市场情绪数据写入测试失败，数据库中无记录")
                    return False
        except Exception as e:
            logger.error(f"市场情绪数据写入测试失败: {e}")
            return False
    
    def test_industry_fund_flow_write(self):
        """测试行业资金流向数据写入"""
        try:
            logger.info("开始测试行业资金流向数据写入...")
            
            # 获取当前日期
            date = datetime.now().strftime("%Y-%m-%d")
            
            # 获取行业资金流向数据
            fund_flow = self.akshare_data.get_fund_flow(date=date, save=True)
            
            if fund_flow.empty:
                logger.error("获取行业资金流向数据失败")
                return False
            
            logger.info(f"成功获取并写入 {len(fund_flow)} 条行业资金流向数据")
            
            # 验证数据是否写入数据库
            with self.db_engine.connect() as conn:
                query = text("SELECT COUNT(*) FROM industry_fund_flow")
                result = conn.execute(query).fetchone()
                count = result[0]
                logger.info(f"数据库中共有 {count} 条行业资金流向数据记录")
                
                if count > 0:
                    logger.info("行业资金流向数据写入测试成功")
                    return True
                else:
                    logger.error("行业资金流向数据写入测试失败，数据库中无记录")
                    return False
        except Exception as e:
            logger.error(f"行业资金流向数据写入测试失败: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("开始运行所有数据库写入测试...")
        
        # 测试数据库连接
        if not self.test_connection():
            logger.error("数据库连接测试失败，终止后续测试")
            return False
        
        # 测试ETF列表写入
        # etf_list_result = self.test_etf_list_write()
        
        # 测试ETF指标数据写入
        # etf_indicators_result = self.test_etf_indicators_write()
        
        # 测试市场情绪数据写入
        market_sentiment_result = self.test_market_sentiment_write()
        
        # 测试行业资金流向数据写入
        fund_flow_result = self.test_industry_fund_flow_write()
        
        # 汇总测试结果
        results = {
            "ETF列表写入": etf_list_result,
            "ETF指标数据写入": etf_indicators_result,
            "市场情绪数据写入": market_sentiment_result,
            "行业资金流向数据写入": fund_flow_result
        }
        
        # 输出测试结果
        logger.info("数据库写入测试结果汇总:")
        for test_name, result in results.items():
            status = "成功" if result else "失败"
            logger.info(f"{test_name}: {status}")
        
        # 判断整体测试结果
        overall_result = all(results.values())
        if overall_result:
            logger.info("所有数据库写入测试均已通过")
        else:
            logger.error("部分数据库写入测试未通过，请检查日志")
        
        return overall_result


if __name__ == "__main__":
    # 运行测试
    test = DBWriteTest()
    test.run_all_tests()
