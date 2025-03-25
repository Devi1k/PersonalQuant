#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AKShare数据获取模块
用于从AKShare获取ETF、指数和股票数据
"""

import os
import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta
import logging
from pathlib import Path

# 设置日志
logger = logging.getLogger(__name__)

class AKShareData:
    """AKShare数据获取类"""
    
    def __init__(self, raw_data_dir="data/raw", processed_data_dir="data/processed"):
        """
        初始化AKShare数据获取类
        
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
        
        logger.info(f"AKShare数据获取模块初始化完成，原始数据目录: {raw_data_dir}, 处理后数据目录: {processed_data_dir}")
    
    def get_etf_list(self, save=True):
        """
        获取所有ETF基金列表
        
        Parameters
        ----------
        save : bool, default True
            是否保存到文件
            
        Returns
        -------
        pandas.DataFrame
            ETF基金列表
        """
        logger.info("开始获取ETF基金列表...")
        try:
            # 使用AKShare获取ETF基金列表
            etf_df = ak.fund_etf_category_sina(symbol="ETF基金")
            
            # 重命名列，使其更符合标准
            etf_df = etf_df.rename(columns={
                "基金代码": "code",
                "基金简称": "name",
                "当前价格": "price",
                "涨跌额": "change",
                "涨跌幅": "pct_change",
                "成交量": "volume",
                "成交额": "amount",
                "开盘价": "open",
                "最高价": "high",
                "最低价": "low",
                "昨收": "pre_close"
            })
            
            # 处理百分比字符串
            if "pct_change" in etf_df.columns:
                etf_df["pct_change"] = etf_df["pct_change"].astype(str).str.replace("%", "").astype(float) / 100
            
            # 添加获取日期
            etf_df["date"] = datetime.now().strftime("%Y-%m-%d")
            
            if save:
                # 保存到CSV文件
                file_path = self.raw_data_dir / f"etf_list_{datetime.now().strftime('%Y%m%d')}.csv"
                etf_df.to_csv(file_path, index=False, encoding="utf-8-sig")
                logger.info(f"ETF基金列表已保存至 {file_path}，共 {len(etf_df)} 条记录")
            
            return etf_df
            
        except Exception as e:
            logger.error(f"获取ETF基金列表失败: {e}")
            return pd.DataFrame()
    
    def get_industry_etfs(self, save=True):
        """
        获取行业ETF列表
        通过关键词筛选出行业相关ETF
        
        Parameters
        ----------
        save : bool, default True
            是否保存到文件
            
        Returns
        -------
        pandas.DataFrame
            行业ETF列表
        """
        logger.info("开始获取行业ETF列表...")
        
        try:
            # 获取所有ETF
            all_etfs = self.get_etf_list(save=False)
            
            # 行业ETF关键词
            industry_keywords = [
                "行业", "消费", "医药", "金融", "地产", "科技", "材料", 
                "能源", "通信", "公用", "汽车", "家电", "军工", "传媒",
                "银行", "保险", "证券", "计算机", "半导体", "有色", "煤炭",
                "石油", "钢铁", "化工", "基建", "环保", "农业", "酒"
            ]
            
            # 筛选行业ETF
            industry_etfs = all_etfs[all_etfs["name"].str.contains("|".join(industry_keywords))]
            
            if save:
                # 保存到CSV文件
                file_path = self.raw_data_dir / f"industry_etfs_{datetime.now().strftime('%Y%m%d')}.csv"
                industry_etfs.to_csv(file_path, index=False, encoding="utf-8-sig")
                logger.info(f"行业ETF列表已保存至 {file_path}，共 {len(industry_etfs)} 条记录")
            
            return industry_etfs
            
        except Exception as e:
            logger.error(f"获取行业ETF列表失败: {e}")
            return pd.DataFrame()
    
    def get_etf_history(self, code, start_date=None, end_date=None, 
                       fields=None, adjust="qfq", save=True):
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
        save : bool, default True
            是否保存到文件
            
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
            
        logger.info(f"开始获取ETF {code} 从 {start_date} 到 {end_date} 的历史数据...")
        
        try:
            # 使用AKShare获取ETF历史数据
            if adjust == "qfq":
                df = ak.fund_etf_hist_sina(symbol=code, adjust="qfq")
            elif adjust == "hfq":
                df = ak.fund_etf_hist_sina(symbol=code, adjust="hfq")
            else:
                df = ak.fund_etf_hist_sina(symbol=code)
            
            # 重命名列，使其更符合标准
            df = df.rename(columns={
                "date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume"
            })
            
            # 转换日期格式
            df["date"] = pd.to_datetime(df["date"])
            
            # 筛选日期范围
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
            
            # 按日期排序
            df = df.sort_values("date")
            
            # 筛选字段
            if fields is not None:
                df = df[["date"] + [f for f in fields if f in df.columns]]
            
            if save:
                # 保存到CSV文件
                file_path = self.raw_data_dir / f"etf_{code}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
                df.to_csv(file_path, index=False)
                logger.info(f"ETF {code} 历史数据已保存至 {file_path}，共 {len(df)} 条记录")
            
            return df
            
        except Exception as e:
            logger.error(f"获取ETF {code} 历史数据失败: {e}")
            return pd.DataFrame()
    
    def get_index_history(self, code, start_date=None, end_date=None, 
                         fields=None, save=True):
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
        save : bool, default True
            是否保存到文件
            
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
            
        logger.info(f"开始获取指数 {code} 从 {start_date} 到 {end_date} 的历史数据...")
        
        try:
            # 使用AKShare获取指数历史数据
            df = ak.stock_zh_index_daily(symbol=code)
            
            # 转换日期格式
            df["date"] = pd.to_datetime(df.index)
            df = df.reset_index(drop=True)
            
            # 筛选日期范围
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
            
            # 按日期排序
            df = df.sort_values("date")
            
            # 筛选字段
            if fields is not None:
                df = df[["date"] + [f for f in fields if f in df.columns]]
            
            if save:
                # 保存到CSV文件
                file_path = self.raw_data_dir / f"index_{code}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
                df.to_csv(file_path, index=False)
                logger.info(f"指数 {code} 历史数据已保存至 {file_path}，共 {len(df)} 条记录")
            
            return df
            
        except Exception as e:
            logger.error(f"获取指数 {code} 历史数据失败: {e}")
            return pd.DataFrame()
    
    def get_fund_flow(self, date=None, save=True):
        """
        获取行业资金流向数据
        
        Parameters
        ----------
        date : str, default None
            日期，格式 "YYYY-MM-DD"，默认为今天
        save : bool, default True
            是否保存到文件
            
        Returns
        -------
        pandas.DataFrame
            行业资金流向数据
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
            
        logger.info(f"开始获取 {date} 的行业资金流向数据...")
        
        try:
            # 使用AKShare获取行业资金流向数据
            df = ak.stock_sector_fund_flow_rank(indicator="今日")
            
            # 添加日期列
            df["date"] = date
            
            if save:
                # 保存到CSV文件
                file_path = self.raw_data_dir / f"industry_fund_flow_{date.replace('-', '')}.csv"
                df.to_csv(file_path, index=False, encoding="utf-8-sig")
                logger.info(f"行业资金流向数据已保存至 {file_path}，共 {len(df)} 条记录")
            
            return df
            
        except Exception as e:
            logger.error(f"获取行业资金流向数据失败: {e}")
            return pd.DataFrame()
    
    def get_market_sentiment(self, date=None, save=True):
        """
        获取市场情绪指标数据
        包括北向资金、融资融券、涨跌停数量等
        
        Parameters
        ----------
        date : str, default None
            日期，格式 "YYYY-MM-DD"，默认为今天
        save : bool, default True
            是否保存到文件
            
        Returns
        -------
        dict
            市场情绪指标数据字典
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
            
        logger.info(f"开始获取 {date} 的市场情绪指标数据...")
        
        sentiment_data = {}
        
        try:
            # 1. 获取北向资金数据
            try:
                north_df = ak.stock_em_hsgt_north_net_flow_in(indicator="沪股通")
                north_df["date"] = pd.to_datetime(north_df["date"])
                north_df = north_df[north_df["date"] == date]
                if not north_df.empty:
                    sentiment_data["north_fund"] = north_df.iloc[0]["value"]
            except Exception as e:
                logger.error(f"获取北向资金数据失败: {e}")
            
            # 2. 获取融资融券数据
            try:
                margin_df = ak.stock_margin_underlying_info_szse(date=date.replace("-", ""))
                if not margin_df.empty:
                    sentiment_data["margin_balance"] = margin_df["融资余额"].sum()
                    sentiment_data["margin_buy"] = margin_df["融资买入额"].sum()
            except Exception as e:
                logger.error(f"获取融资融券数据失败: {e}")
            
            # 3. 获取涨跌停数据
            try:
                limit_df = ak.stock_em_zt_pool(date=date)
                sentiment_data["up_limit_count"] = len(limit_df)
                
                limit_down_df = ak.stock_em_dt_pool(date=date)
                sentiment_data["down_limit_count"] = len(limit_down_df)
            except Exception as e:
                logger.error(f"获取涨跌停数据失败: {e}")
            
            # 添加日期
            sentiment_data["date"] = date
            
            if save:
                # 保存到CSV文件
                file_path = self.raw_data_dir / f"market_sentiment_{date.replace('-', '')}.csv"
                pd.DataFrame([sentiment_data]).to_csv(file_path, index=False)
                logger.info(f"市场情绪指标数据已保存至 {file_path}")
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"获取市场情绪指标数据失败: {e}")
            return {}
    
    def get_all_data(self, start_date=None, end_date=None, etf_codes=None):
        """
        获取所有需要的数据
        
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
            包含所有获取数据的字典
        """
        # 设置默认日期
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        logger.info(f"开始获取从 {start_date} 到 {end_date} 的所有数据...")
        
        result = {}
        
        # 1. 获取ETF列表
        industry_etfs = self.get_industry_etfs()
        result["industry_etfs"] = industry_etfs
        
        # 如果未指定ETF代码，则使用获取的行业ETF
        if etf_codes is None and not industry_etfs.empty:
            etf_codes = industry_etfs["code"].tolist()
        
        # 2. 获取ETF历史数据
        etf_history = {}
        if etf_codes:
            for code in etf_codes:
                etf_history[code] = self.get_etf_history(code, start_date, end_date)
        result["etf_history"] = etf_history
        
        # 3. 获取指数历史数据
        index_codes = ["000001", "000300", "000905", "399001", "399006"]  # 上证、沪深300、中证500、深证成指、创业板指
        index_history = {}
        for code in index_codes:
            index_history[code] = self.get_index_history(code, start_date, end_date)
        result["index_history"] = index_history
        
        # 4. 获取最新的行业资金流向
        fund_flow = self.get_fund_flow()
        result["fund_flow"] = fund_flow
        
        # 5. 获取最新的市场情绪指标
        sentiment = self.get_market_sentiment()
        result["sentiment"] = sentiment
        
        logger.info("所有数据获取完成")
        
        return result


# 测试代码
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建AKShareData实例
    data = AKShareData()
    
    # 测试获取ETF列表
    etf_list = data.get_etf_list()
    print(f"获取到 {len(etf_list)} 个ETF")
    
    # 测试获取行业ETF列表
    industry_etfs = data.get_industry_etfs()
    print(f"获取到 {len(industry_etfs)} 个行业ETF")
    
    # 测试获取ETF历史数据 (以创业板ETF为例)
    etf_history = data.get_etf_history("159915", start_date="2023-01-01", end_date="2023-12-31")
    print(f"获取到创业板ETF历史数据 {len(etf_history)} 条记录")
    
    # 测试获取指数历史数据 (以沪深300为例)
    index_history = data.get_index_history("000300", start_date="2023-01-01", end_date="2023-12-31")
    print(f"获取到沪深300指数历史数据 {len(index_history)} 条记录")