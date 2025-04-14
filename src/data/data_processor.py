#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据处理模块
用于清洗、转换和处理从AKShare获取的原始数据
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

# 设置日志
logger = logging.getLogger(__name__)

class DataProcessor:
    """数据处理类"""
    
    def __init__(self, raw_data_dir="data/raw", processed_data_dir="data/processed"):
        """
        初始化数据处理类
        
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

        # --- 初始化数据库引擎 ---
        self.config = load_config()
        self.engine = get_db_engine(self.config)
        # --- ---

        logger.info(f"数据处理模块初始化完成，原始数据目录: {raw_data_dir}, 处理后数据目录: {processed_data_dir}")
        if not self.engine:
             logger.warning("数据库引擎初始化失败，处理后的数据将无法保存到数据库。")
    
    def process_etf_history(self, df, fill_missing=True, detect_outliers=True):
        """
        处理ETF历史数据
        
        Parameters
        ----------
        df : pandas.DataFrame
            ETF历史数据
        fill_missing : bool, default True
            是否填充缺失值
        detect_outliers : bool, default True
            是否检测异常值
            
        Returns
        -------
        pandas.DataFrame
            处理后的ETF历史数据
        """
        if df.empty:
            logger.warning("输入的ETF历史数据为空")
            return df
        
        logger.info(f"开始处理ETF历史数据，共 {len(df)} 条记录")
        
        # 复制数据，避免修改原始数据
        processed_df = df.copy()
        
        # 确保日期列为datetime类型
        if "trade_date" in processed_df.columns:
            processed_df["trade_date"] = pd.to_datetime(processed_df["trade_date"])
        
        # 填充缺失值
        if fill_missing:
            # 对于价格数据，使用前值填充
            price_cols = ["open", "high", "low", "close"]
            for col in price_cols:
                if col in processed_df.columns and processed_df[col].isnull().any():
                    missing_count = processed_df[col].isnull().sum()
                    processed_df[col] = processed_df[col].fillna(method="ffill")
                    logger.info(f"填充 {col} 列的 {missing_count} 个缺失值")
            
            # 对于成交量和成交额，使用0填充
            volume_cols = ["volume", "amount"]
            for col in volume_cols:
                if col in processed_df.columns and processed_df[col].isnull().any():
                    missing_count = processed_df[col].isnull().sum()
                    processed_df[col] = processed_df[col].fillna(0)
                    logger.info(f"填充 {col} 列的 {missing_count} 个缺失值")
        
        # 检测异常值
        if detect_outliers:
            # 对于价格数据，使用滚动窗口的3倍标准差检测异常值，避免前视偏差
            price_cols = ["open", "high", "low", "close"]
            window_size = 20  # 使用20日滚动窗口
            min_obs = 5      # 最少需要5个观测值才能计算
            
            for col in price_cols:
                if col in processed_df.columns:
                    # 计算滚动均值和标准差
                    rolling_mean = processed_df[col].rolling(window=window_size, min_periods=min_obs).mean()
                    rolling_std = processed_df[col].rolling(window=window_size, min_periods=min_obs).std()
                    
                    # 计算上下边界
                    lower_bound = rolling_mean - 3 * rolling_std
                    upper_bound = rolling_mean + 3 * rolling_std
                    
                    # 检测异常值
                    outliers_mask = (processed_df[col] < lower_bound) | (processed_df[col] > upper_bound)
                    outliers = processed_df[outliers_mask]
                    
                    if not outliers.empty:
                        logger.warning(f"检测到 {col} 列中有 {len(outliers)} 个异常值")
                        
                        # 使用前值替换异常值
                        for idx in outliers.index:
                            if idx > 0:  # 确保有前值可用
                                processed_df.loc[idx, col] = processed_df.loc[idx-1, col]
                                logger.info(f"替换 {col} 列索引 {idx} 处的异常值")
        
        # 计算日收益率
        if "close" in processed_df.columns:
            processed_df["daily_return"] = processed_df["close"].pct_change()
            # 将第一个NaN值填充为0
            processed_df["daily_return"] = processed_df["daily_return"].fillna(0)
            
        # 计算波动率 (20日滚动标准差)
        if "daily_return" in processed_df.columns:
            # 计算20日滚动标准差
            processed_df["volatility_20d"] = processed_df["daily_return"].rolling(window=20).std()
            
            # 避免前视偏差：不填充初始NaN值，让下游策略处理
            # 如果确实需要填充，可以使用expanding窗口计算
            logger.info("计算20日波动率，初始NaN值保留不填充，避免前视偏差")
        
        # 确保数据按日期排序
        if "trade_date" in processed_df.columns:
            processed_df = processed_df.sort_values("trade_date")
        
        logger.info(f"ETF历史数据处理完成，处理后共 {len(processed_df)} 条记录")
        
        return processed_df
    
    def calculate_technical_indicators(self, df, config=None):
        """
        计算技术指标
        
        Parameters
        ----------
        df : pandas.DataFrame
            ETF历史数据，必须包含OHLCV数据
        config : dict, default None
            策略配置参数
            
        Returns
        -------
        pandas.DataFrame
            添加了技术指标的数据
        """
        if df.empty:
            logger.warning("输入的ETF历史数据为空")
            return df
        
        # 确保必要的列存在
        required_cols = ["trade_date", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"计算技术指标所需的列缺失: {missing_cols}")
            return df
        
        logger.info(f"开始计算技术指标，基于 {len(df)} 条历史数据")
        
        # 复制数据，避免修改原始数据
        result_df = df.copy()
        
        # 确保数据按日期排序
        result_df = result_df.sort_values("trade_date")
        
        # 1. 计算移动平均线 - 改进填充策略
        ma_periods = [5, 10, 20, 60]
        for period in ma_periods:
            # 计算移动平均线
            result_df[f"ma_{period}"] = result_df["close"].rolling(window=period).mean()
            # 避免前视偏差：只使用前向填充，不使用后向填充
            result_df[f"ma_{period}"] = result_df[f"ma_{period}"].fillna(method="ffill")
            logger.info(f"计算 {period} 日移动平均线")
            
        # 1.1 计算指数移动平均线(EMA)
        # 从配置中获取EMA通道参数，如果没有则使用默认值
        ema_channel_period = 144
        ema_channel_width = 0.05
        # TODO: 处理空值
        if config and 'strategy' in config and 'trend' in config['strategy']:
            trend_config = config['strategy']['trend']
            ema_channel_period = trend_config.get('ema_channel_period', 144)
            ema_channel_width = trend_config.get('ema_channel_width', 0.05)
        
        # 添加MACD所需的EMA周期
        ema_periods = [12, 26, ema_channel_period]
        # 添加配置中的MACD参数
        if config and 'strategy' in config and 'trend' in config['strategy']:
            trend_config = config['strategy']['trend']
            if 'macd_short_period' in trend_config:
                ema_periods.append(trend_config['macd_short_period'])
            if 'macd_long_period' in trend_config:
                ema_periods.append(trend_config['macd_long_period'])
        
        # 去重
        ema_periods = list(set(ema_periods))
        
        for period in ema_periods:
            # 计算EMA
            result_df[f"ema_{period}"] = result_df["close"].ewm(span=period, adjust=False).mean()
            # 避免前视偏差：只使用前向填充，不使用后向填充
            result_df[f"ema_{period}"] = result_df[f"ema_{period}"].fillna(method="ffill")
            logger.info(f"计算 {period} 日指数移动平均线(EMA)")
            
        # 1.2 计算EMA均线通道
        ema_col = f"ema_{ema_channel_period}"
        if ema_col in result_df.columns:
            # 计算标准差作为通道宽度
            ema_std = result_df["close"].rolling(window=ema_channel_period).std(ddof=1)
            # 上通道 = EMA + 通道宽度
            result_df[f"{ema_col}_upper"] = result_df[ema_col] * (1 + ema_channel_width)
            # 下通道 = EMA - 通道宽度
            result_df[f"{ema_col}_lower"] = result_df[ema_col] * (1 - ema_channel_width)
            # 填充NaN值
            channel_cols = [f"{ema_col}_upper", f"{ema_col}_lower"]
            result_df[channel_cols] = result_df[channel_cols].fillna(method="ffill")
            logger.info(f"计算{ema_channel_period}日EMA均线通道，宽度={ema_channel_width}")
        
        # 2. 计算MACD
        # 从配置中获取MACD参数，如果没有则使用默认值
        macd_short_period = 12
        macd_signal_period = 9
        macd_long_period = 26
        macd_short_ema = 21  # 1个月期EMA
        macd_long_ema = 200  # 200日EMA
        
        if config and 'strategy' in config and 'trend' in config['strategy']:
            trend_config = config['strategy']['trend']
            macd_short_ema = trend_config.get('macd_short_period', 21)
            macd_long_ema = trend_config.get('macd_long_period', 200)
        
        # MACD = 12日EMA - 26日EMA
        # 信号线 = MACD的9日EMA
        # MACD柱 = MACD - 信号线
        # 计算MACD
        # adjust=False确保与TA-Lib等标准库计算结果一致
        ema12 = result_df["close"].ewm(span=macd_short_period, adjust=False).mean()
        ema26 = result_df["close"].ewm(span=macd_long_period, adjust=False).mean()
        result_df["macd"] = ema12 - ema26
        result_df["macd_signal"] = result_df["macd"].ewm(span=macd_signal_period, adjust=False).mean()
        result_df["macd_hist"] = result_df["macd"] - result_df["macd_signal"]
        logger.info(f"计算MACD指标，短期={macd_short_period}，长期={macd_long_period}，信号线={macd_signal_period}")
        
        # 计算1个月期与200日EMA的收敛发散
        ema21 = result_df["close"].ewm(span=macd_short_ema, adjust=False).mean()
        ema200 = result_df["close"].ewm(span=macd_long_ema, adjust=False).mean()
        result_df["ema_21"] = ema21
        result_df["ema_200"] = ema200
        result_df["ema_21_200_diff"] = ema21 - ema200
        logger.info(f"计算{macd_short_ema}日与{macd_long_ema}日EMA收敛发散")
        
        # 填充NaN值
        macd_cols = ["macd", "macd_signal", "macd_hist"]
        result_df[macd_cols] = result_df[macd_cols].fillna(0)
        logger.info("计算MACD指标")
        
        # 3. 计算RSI (相对强弱指数)
        # RSI = 100 - (100 / (1 + RS))
        # RS = 平均上涨幅度 / 平均下跌幅度
        delta = result_df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        # 使用Wilder's Smoothing计算RSI
        avg_gain_14 = gain.ewm(com=13, adjust=False).mean()
        avg_loss_14 = loss.ewm(com=13, adjust=False).mean()
        
        # 处理除零情况
        rs_14 = avg_gain_14 / avg_loss_14.replace(0, np.finfo(float).eps)
        result_df["rsi_14"] = 100 - (100 / (1 + rs_14))
        # 处理极值情况
        result_df["rsi_14"] = result_df["rsi_14"].clip(0, 100).fillna(50)
        logger.info("计算RSI指标")
        
        # 4. 计算布林带
        # 从配置中获取布林带参数，如果没有则使用默认值
        bollinger_period = 20
        bollinger_std_dev = 2.0
        
        if config and 'strategy' in config and 'trend' in config['strategy']:
            trend_config = config['strategy']['trend']
            bollinger_period = trend_config.get('bollinger_period', 20)
            bollinger_std_dev = trend_config.get('bollinger_std_dev', 2.0)
        
        # 检查数据量是否足够计算布林带
        if len(result_df) >= bollinger_period:
            # 中轨 = N日移动平均线
            result_df["bb_middle"] = result_df["close"].rolling(window=bollinger_period).mean()
            # 使用样本标准差(ddof=1)而非总体标准差(ddof=0)
            result_df["bb_std"] = result_df["close"].rolling(window=bollinger_period).std(ddof=1)
            result_df["bb_upper"] = result_df["bb_middle"] + bollinger_std_dev * result_df["bb_std"]
            result_df["bb_lower"] = result_df["bb_middle"] - bollinger_std_dev * result_df["bb_std"]
            
            # 改进填充策略
            # 对于中轨，使用与移动平均线相同的填充策略
            result_df["bb_middle"] = result_df["bb_middle"].fillna(method="ffill")
            
            # 对于标准差，使用前向填充，避免前视偏差
            result_df["bb_std"] = result_df["bb_std"].fillna(method="ffill")
            
            # 重新计算上下轨，确保它们与中轨和标准差一致
            result_df["bb_upper"] = result_df["bb_middle"] + bollinger_std_dev * result_df["bb_std"]
            result_df["bb_lower"] = result_df["bb_middle"] - bollinger_std_dev * result_df["bb_std"]
            
            logger.info(f"计算布林带指标，周期={bollinger_period}，标准差倍数={bollinger_std_dev}")
        else:
            # 数据不足时，使用合理的默认值
            logger.warning(f"数据量不足以计算布林带指标(需要至少{bollinger_period}个数据点，当前只有{len(result_df)}个)")
            
            # 设置默认值
            mean_price = result_df["close"].mean()
            std_price = result_df["close"].std() if len(result_df) > 1 else mean_price * 0.02
            
            result_df["bb_middle"] = mean_price
            result_df["bb_std"] = std_price
            result_df["bb_upper"] = mean_price + bollinger_std_dev * std_price
            result_df["bb_lower"] = mean_price - bollinger_std_dev * std_price
        
        # 5. 计算KDJ
        # 未成熟随机值（RSV）= (收盘价 - 最低价) / (最高价 - 最低价) * 100
        # K = 2/3 * 前一日K值 + 1/3 * 当日RSV
        # D = 2/3 * 前一日D值 + 1/3 * 当日K值
        # J = 3 * K - 2 * D
        # 计算KDJ指标
        low_9 = result_df["low"].rolling(window=9).min()
        high_9 = result_df["high"].rolling(window=9).max()
        
        # 处理除零问题
        high_low_range = high_9 - low_9
        rsv = np.where(high_low_range != 0,
                      (result_df["close"] - low_9) / high_low_range * 100,
                      50)  # 当最高价=最低价时，使用中性值50
        
        # 使用向量化计算K/D值，使用EMA实现平滑
        k = pd.Series(rsv).ewm(alpha=1/3, adjust=False).mean().fillna(50)
        d = k.ewm(alpha=1/3, adjust=False).mean().fillna(50)
        j = 3 * k - 2 * d
        
        result_df["kdj_k"] = k
        result_df["kdj_d"] = d
        result_df["kdj_j"] = j
        
        # 处理极值
        kdj_cols = ["kdj_k", "kdj_d", "kdj_j"]
        result_df[kdj_cols] = result_df[kdj_cols].clip(0, 100)
        logger.info("计算KDJ指标")
        
        # 6. 计算CCI (顺势指标)
        # CCI = (TP - SMA(TP, n)) / (0.015 * MD)
        # TP = (最高价 + 最低价 + 收盘价) / 3
        # SMA(TP, n) = TP的n日简单移动平均
        # MD = TP与SMA(TP, n)的平均绝对偏差
        tp = (result_df["high"] + result_df["low"] + result_df["close"]) / 3
        tp_sma_20 = tp.rolling(window=20).mean()
        md_20 = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        # 处理除零情况
        md_20_safe = md_20.replace(0, np.finfo(float).eps)
        result_df["cci_20"] = (tp - tp_sma_20) / (0.015 * md_20_safe)
        # 填充NaN值并处理极值
        result_df["cci_20"] = result_df["cci_20"].fillna(0).clip(-100, 100)
        logger.info("计算CCI指标")
        
        # 7. 计算ATR (平均真实波幅)
        # TR = max(当日最高价 - 当日最低价, |当日最高价 - 前一日收盘价|, |当日最低价 - 前一日收盘价|)
        # ATR = TR的n日简单移动平均
        tr1 = result_df["high"] - result_df["low"]
        tr2 = (result_df["high"] - result_df["close"].shift(1)).abs()
        tr3 = (result_df["low"] - result_df["close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # 使用Wilder's Smoothing计算ATR
        result_df["atr_14"] = tr.ewm(alpha=1/14, adjust=False).mean()
        logger.info("计算ATR指标")
        
        # 8. 计算OBV (能量潮)
        # 如果当日收盘价 > 前一日收盘价，OBV = 前一日OBV + 当日成交量
        # 如果当日收盘价 < 前一日收盘价，OBV = 前一日OBV - 当日成交量
        # 如果当日收盘价 = 前一日收盘价，OBV = 前一日OBV
        # 使用向量化计算OBV，提高效率
        price_change = result_df["close"].diff()
        volume_direction = np.sign(price_change)  # 1 for up, -1 for down, 0 for no change
        obv = (volume_direction * result_df["volume"]).cumsum()
        result_df["obv"] = obv.fillna(0)
        logger.info("计算OBV指标")
        
        # 9. 计算成交量的5日均线（用于量价齐升确认）
        result_df["volume_ma_5"] = result_df["volume"].rolling(window=5).mean().fillna(method="ffill")
        logger.info("计算成交量5日均线")
        
        # 10. 计算VWAP（成交量加权平均价格）
        # 标准VWAP通常是日内指标，这里实现一个滚动的VWAP用于日线数据
        # 避免前视偏差，只使用历史数据
        if "volume" in result_df.columns:
            # 计算典型价格 (TP)
            result_df["typical_price"] = (result_df["high"] + result_df["low"] + result_df["close"]) / 3
            # 计算价格乘以成交量
            result_df["tp_volume"] = result_df["typical_price"] * result_df["volume"]
            
            # 计算20日滚动VWAP
            window_size = 20
            result_df["vwap_20"] = (
                result_df["tp_volume"].rolling(window=window_size).sum() / 
                result_df["volume"].rolling(window=window_size).sum()
            )
            
            # 清理临时列
            result_df = result_df.drop(columns=["typical_price", "tp_volume"])
            logger.info(f"计算{window_size}日滚动VWAP")
        
        logger.info(f"技术指标计算完成，共添加 {len(result_df.columns) - len(df.columns)} 个技术指标")
        
        # --- 添加保存数据的逻辑 ---
        if hasattr(df, 'etf_code') and 'etf_code' in df.columns:
            etf_code = df['etf_code'].iloc[0] if not df.empty else None
        else:
            etf_code = None
            
        if hasattr(df, 'trade_date') and 'trade_date' in df.columns:
            # 确保结果 DataFrame 也有 trade_date 列
            if 'trade_date' not in result_df.columns and 'date' in result_df.columns:
                result_df['trade_date'] = result_df['date']
                
            # 保存数据到数据库
            if self.engine and etf_code is not None:
                success = upsert_df_to_sql(
                    result_df,
                    "etf_indicators",
                    self.engine,
                    unique_columns=["etf_code", "trade_date"],
                )
                if success:
                    logger.info(
                        f"ETF {etf_code} 技术指标数据已成功写入数据库 etf_indicators，共 {len(result_df)} 条记录"
                    )
                else:
                    logger.error(f"ETF {etf_code} 技术指标数据写入数据库失败。")
        # --- 保存逻辑结束 ---
        
        return result_df
    
    def process_industry_fund_flow(self, df):
        """
        处理行业资金流向数据
        
        Parameters
        ----------
        df : pandas.DataFrame
            行业资金流向数据
            
        Returns
        -------
        pandas.DataFrame
            处理后的行业资金流向数据
        """
        if df.empty:
            logger.warning("输入的行业资金流向数据为空")
            return df
        
        logger.info(f"开始处理行业资金流向数据，共 {len(df)} 条记录")
        
        # 复制数据，避免修改原始数据
        processed_df = df.copy()
        
        # 确保日期列为datetime类型
        if "trade_date" in processed_df.columns:
            processed_df["trade_date"] = pd.to_datetime(processed_df["trade_date"])
        
        # 处理金额列，将字符串转换为数值
        money_cols = [col for col in processed_df.columns if "金额" in col or "净额" in col or "成交额" in col]
        for col in money_cols:
            if col in processed_df.columns:
                # 处理带有"亿"、"万"等单位的金额
                processed_df[col] = processed_df[col].astype(str)
                processed_df[col] = processed_df[col].str.replace("亿", "e8")
                processed_df[col] = processed_df[col].str.replace("万", "e4")
                processed_df[col] = processed_df[col].apply(lambda x: eval(x) if isinstance(x, str) and any(unit in x for unit in ["e8", "e4"]) else x)
                processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")
        
        # 处理百分比列，将字符串转换为数值
        pct_cols = [col for col in processed_df.columns if "涨跌幅" in col or "占比" in col or "%" in col]
        for col in pct_cols:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].astype(str)
                processed_df[col] = processed_df[col].str.replace("%", "")
                processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce") / 100
        
        logger.info(f"行业资金流向数据处理完成，处理后共 {len(processed_df)} 条记录")
        
        return processed_df
    
    def process_market_sentiment(self, sentiment_dict):
        """
        处理市场情绪指标数据
        
        Parameters
        ----------
        sentiment_dict : dict
            市场情绪指标数据字典
            
        Returns
        -------
        dict
            处理后的市场情绪指标数据字典
        """
        if not sentiment_dict:
            logger.warning("输入的市场情绪指标数据为空")
            return sentiment_dict
        
        logger.info("开始处理市场情绪指标数据")
        
        # 复制数据，避免修改原始数据
        processed_dict = sentiment_dict.copy()
        
        # 确保日期为datetime类型
        if "date" in processed_dict:
            processed_dict["date"] = pd.to_datetime(processed_dict["date"])
        
        # 处理金额数据，确保为数值类型
        money_keys = ["margin_balance", "margin_buy", "north_fund"]
        for key in money_keys:
            if key in processed_dict:
                # 如果是字符串，尝试转换为数值
                if isinstance(processed_dict[key], str):
                    # 处理带有"亿"、"万"等单位的金额
                    value = processed_dict[key]
                    if "亿" in value:
                        value = float(value.replace("亿", "")) * 1e8
                    elif "万" in value:
                        value = float(value.replace("万", "")) * 1e4
                    processed_dict[key] = float(value)
        
        # 计算涨跌停比例
        if "up_limit_count" in processed_dict and "down_limit_count" in processed_dict:
            total = processed_dict["up_limit_count"] + processed_dict["down_limit_count"]
            if total > 0:
                processed_dict["up_down_ratio"] = processed_dict["up_limit_count"] / total
            else:
                processed_dict["up_down_ratio"] = 0.5  # 默认为中性
        
        logger.info("市场情绪指标数据处理完成")
        
        return processed_dict
    
    def save_processed_data(self, df, name, date=None, etf_code=None, save_to_db=True):
        """
        保存处理后的数据到数据库 (如果适用)

        Parameters
        ----------
        df : pandas.DataFrame
            处理后的数据
        name : str
            数据名称 (用于判断是否为指标数据)
        date : str, default None
            日期 (当前未使用)
        etf_code : str, default None
            相关的 ETF 代码 (如果保存的是 ETF 指标)
        save_to_db : bool, default True
            是否尝试保存到数据库

        Returns
        -------
        bool
            数据库保存是否成功 (如果尝试了保存)
        """
        if df.empty:
            logger.warning(f"要保存的{name}数据为空")
            return ""
        
        # if date is None:
        #     date = datetime.now().strftime("%Y-%m-%d") # date 参数当前未使用

        db_saved = False
        if save_to_db and self.engine:
            # 尝试根据 name 判断是否为 ETF 指标数据
            # 注意：这种判断方式比较脆弱，更好的方式是调用者明确指定目标表
            if "etf" in name.lower() or "indicator" in name.lower():
                if etf_code is None and 'etf_code' not in df.columns:
                    logger.error(f"无法将 {name} 数据写入数据库：缺少 etf_code。")
                elif 'trade_date' not in df.columns:
                     logger.error(f"无法将 {name} 数据写入数据库：缺少 trade_date 列。")
                else:
                    logger.info(f"尝试将处理后的 {name} 数据 (ETF: {etf_code or '从df获取'}) 写入数据库 etf_indicators...")
                    df_to_save = df.copy()
                    if etf_code and 'etf_code' not in df_to_save.columns:
                        df_to_save['etf_code'] = etf_code

                    # 确保日期是 date 类型
                    df_to_save['trade_date'] = pd.to_datetime(df_to_save['trade_date']).dt.date

                    # 选择与 etf_indicators 表匹配的列 (包括主键和指标列)
                    # 需要动态获取 etf_indicators 的列名或硬编码已知列
                    # 这里假设 df 包含了所有需要的指标列
                    indicator_cols = list(df_to_save.columns) # 获取所有列
                    unique_cols = ['etf_code', 'trade_date']
                    # 确保主键列存在
                    if not all(col in indicator_cols for col in unique_cols):
                         logger.error(f"DataFrame 中缺少主键列 {unique_cols}，无法写入 etf_indicators。")
                    else:
                        try:
                            # 移除主键为空的行
                            df_to_save.dropna(subset=unique_cols, inplace=True)
                            if not df_to_save.empty:
                                # 使用 upsert 更新指标
                                success = upsert_df_to_sql(df_to_save[indicator_cols], 'etf_indicators', self.engine, unique_columns=unique_cols)
                                if success:
                                    logger.info(f"成功将 {len(df_to_save)} 条处理后的 {name} 数据更新到数据库 etf_indicators。")
                                    db_saved = True
                                else:
                                    logger.error(f"处理后的 {name} 数据写入数据库 etf_indicators 失败。")
                            else:
                                logger.warning("处理后的数据在移除空主键后变为空，未写入数据库。")
                        except Exception as e:
                            logger.error(f"保存处理后的 {name} 数据到 etf_indicators 时出错: {e}", exc_info=True)
            else:
                # 对于其他类型的数据，暂不写入数据库
                logger.warning(f"数据库保存逻辑未针对数据类型 '{name}' 实现，数据未保存到数据库。")
        elif save_to_db and not self.engine:
             logger.error("无法连接数据库，处理后的数据未保存。")

        # --- 移除 CSV 保存 ---
        # # 构建文件名
        # file_name = f"{name}_{date.replace('-', '')}.csv"
        # file_path = self.processed_data_dir / file_name
        # # 保存数据
        # df.to_csv(file_path, index=False, encoding="utf-8-sig")
        # logger.info(f"{name}数据已保存至 {file_path}，共 {len(df)} 条记录")
        # return str(file_path) # 返回文件路径

        return db_saved # 返回数据库保存是否成功


# 测试代码
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建DataProcessor实例
    processor = DataProcessor()
    
    # 测试处理ETF历史数据
    # 假设我们有一个ETF历史数据文件
    test_file = Path("C:\\Zepu\\Code\\PersonalQuant\\data\\raw\\etf_sz159998_20240401_20250401.csv")
    if test_file.exists():
        df = pd.read_csv(test_file)
        processed_df = processor.process_etf_history(df)
        print(f"处理后的ETF历史数据共 {len(processed_df)} 条记录")
        
        # 计算技术指标
        with_indicators = processor.calculate_technical_indicators(processed_df)
        print(f"添加技术指标后的数据共 {len(with_indicators.columns)} 列")
        
        # 保存处理后的数据
        processor.save_processed_data(with_indicators, "etf_159915_with_indicators", "2023-12-31")
    else:
        print(f"测试文件 {test_file} 不存在")