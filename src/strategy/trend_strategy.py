#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
趋势策略模块
实现各种趋势跟踪和反转交易策略
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path

# 设置日志
logger = logging.getLogger(__name__)

class TrendStrategy:
    """趋势策略类"""
    
    def __init__(self, config=None):
        """
        初始化趋势策略
        
        Parameters
        ----------
        config : dict, default None
            策略配置参数
        """
        self.config = config or {}
        
        # 从配置中获取策略参数，如果没有则使用默认值
        trend_config = self.config.get('strategy', {}).get('trend', {})
        
        # 移动平均线参数
        self.fast_ma = trend_config.get('fast_ma', 20)
        self.slow_ma = trend_config.get('slow_ma', 60)
        
        # 布林带参数
        self.bollinger_period = trend_config.get('bollinger_period', 20)
        self.bollinger_std_dev = trend_config.get('bollinger_std_dev', 2.0)
        
        # EMA参数
        self.ema_short_period = trend_config.get('ema_short_period', 21)
        self.ema_long_period = trend_config.get('ema_long_period', 200)
        
        # 多维周期参数
        self.timeframes = trend_config.get('timeframes', [5, 15, 60])
        
        # 对冲型反转策略参数
        self.ema_channel_period = trend_config.get('ema_channel_period', 144)
        self.ema_channel_width = trend_config.get('ema_channel_width', 0.05)
        
        # 成交量阈值
        self.volume_threshold = trend_config.get('volume_threshold', 1.2)
        
        # 策略类型仓位权重
        self.trend_weight = trend_config.get('trend_weight', 0.7)  # 趋势跟踪 (60-80%)
        self.multi_tf_weight = trend_config.get('multi_tf_weight', 0.25)  # 多周期策略 (20-30%)
        self.reversal_weight = trend_config.get('reversal_weight', 0.05)  # 反转策略 (5-10%)
        
        # 信号组合权重
        default_weights = {
            'bb_signal': 2.0,            # 布林带信号权重提高（趋势跟踪核心指标）
            'ema_signal': 2.5,          # EMA信号权重提高（趋势跟踪核心指标）
            'multi_timeframe_signal': 3.0, # 多周期信号权重最高（优化入场点）
            'ema_reversal_signal': 1.5,   # 反转信号权重适中（风险对冲）
            'volume_price_signal': 1.0    # 成交量确认信号
        }
        self.signal_weights = trend_config.get('signal_weights', default_weights)
        
        logger.info(f"趋势策略初始化完成，参数：快速MA={self.fast_ma}, 慢速MA={self.slow_ma}, "
                   f"布林带周期={self.bollinger_period}, 布林带标准差={self.bollinger_std_dev}, "
                   f"EMA短期={self.ema_short_period}, EMA长期={self.ema_long_period}, "
                   f"EMA通道周期={self.ema_channel_period}, 成交量阈值={self.volume_threshold}")
    
    def bollinger_bands_breakout(self, df):
        """
        布林带（Bollinger Bands）突破信号识别
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含价格和技术指标的数据框
            
        Returns
        -------
        pandas.DataFrame
            添加了布林带突破信号的数据框
        """
        if df.empty:
            logger.warning("输入的数据为空")
            return df
        
        # 确保必要的列存在
        required_cols = ["date", "close", "bb_upper", "bb_lower", "bb_middle", "ema_21", "ema_200"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"布林带突破策略所需的列缺失: {missing_cols}")
            return df
        
        logger.info("开始计算布林带突破信号")
        
        # 复制数据，避免修改原始数据
        result_df = df.copy()
        
        # 计算布林带突破信号
        # 1. 上轨突破: 收盘价高于上轨
        # 2. 下轨突破: 收盘价低于下轨
        
        # 上轨突破信号
        result_df['bb_upper_breakout'] = (result_df['close'] > result_df['bb_upper'])
        
        # 下轨突破信号
        result_df['bb_lower_breakout'] = (result_df['close'] < result_df['bb_lower'])
        
        # 中轨突破信号（从下向上或从上向下）- 保留用于参考
        result_df['bb_middle_up_cross'] = (
            (result_df['close'] > result_df['bb_middle']) & 
            (result_df['close'].shift(1) <= result_df['bb_middle'].shift(1))
        )
        
        result_df['bb_middle_down_cross'] = (
            (result_df['close'] < result_df['bb_middle']) & 
            (result_df['close'].shift(1) >= result_df['bb_middle'].shift(1))
        )
        
        # 判断趋势方向（使用EMA指标）
        result_df['uptrend'] = (result_df['ema_21'] > result_df['ema_200'])
        
        # 综合信号
        # 1 = 买入信号（收盘价 > 上轨 且处于上升趋势）
        # -1 = 卖出信号（收盘价 < 下轨 且处于下降趋势）
        # 0 = 无信号
        
        # 初始化信号列
        result_df['bb_signal'] = 0
        
        # 买入信号：收盘价高于上轨且处于上升趋势
        result_df.loc[result_df['bb_upper_breakout'] & result_df['uptrend'], 'bb_signal'] = 1
        
        # 卖出信号：收盘价低于下轨且处于下降趋势
        result_df.loc[result_df['bb_lower_breakout'] & (~result_df['uptrend']), 'bb_signal'] = -1
        
        # 删除临时列
        result_df.drop(['uptrend'], axis=1, inplace=True)
        
        # 统计信号数量
        buy_signals = (result_df['bb_signal'] == 1).sum()
        sell_signals = (result_df['bb_signal'] == -1).sum()
        logger.info(f"布林带突破信号计算完成，买入信号: {buy_signals}个, 卖出信号: {sell_signals}个")
        
        return result_df
    
    def ema_crossover_signal(self, df):
        """
        EMA长短周期移动平均线交叉信号（21周期与200周期EMA）
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含价格和技术指标的数据框
            
        Returns
        -------
        pandas.DataFrame
            添加了EMA交叉信号的数据框
        """
        if df.empty:
            logger.warning("输入的数据为空")
            return df
        
        # 确保必要的列存在
        required_cols = ["date", "close", "ema_21", "ema_200", "ema_21_200_diff"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"EMA交叉策略所需的列缺失: {missing_cols}")
            return df
        
        logger.info("开始计算EMA长短周期交叉信号")
        
        # 复制数据，避免修改原始数据
        result_df = df.copy()
        
        # 计算差值的变化率
        result_df['ema_diff_change'] = result_df['ema_21_200_diff'].diff()
        
        # 计算交叉信号
        # 1. 金叉：短期EMA从下方穿过长期EMA
        # 2. 死叉：短期EMA从上方穿过长期EMA
        
        # 金叉信号
        result_df['ema_golden_cross'] = (
            (result_df['ema_21'] > result_df['ema_200']) & 
            (result_df['ema_21'].shift(1) <= result_df['ema_200'].shift(1))
        )
        
        # 死叉信号
        result_df['ema_death_cross'] = (
            (result_df['ema_21'] < result_df['ema_200']) & 
            (result_df['ema_21'].shift(1) >= result_df['ema_200'].shift(1))
        )
        
        # 收敛信号：差值向0靠近
        result_df['ema_converging'] = (
            (result_df['ema_21'] > result_df['ema_200']) & 
            (result_df['ema_diff_change'] < 0) |
            (result_df['ema_21'] < result_df['ema_200']) & 
            (result_df['ema_diff_change'] > 0)
        )
        
        # 发散信号：差值远离0
        result_df['ema_diverging'] = (
            (result_df['ema_21'] > result_df['ema_200']) & 
            (result_df['ema_diff_change'] > 0) |
            (result_df['ema_21'] < result_df['ema_200']) & 
            (result_df['ema_diff_change'] < 0)
        )
        
        # 综合信号
        # 1 = 买入信号（金叉或多头发散）
        # -1 = 卖出信号（死叉或空头发散）
        # 0 = 无信号
        
        # 初始化信号列
        result_df['ema_signal'] = 0
        
        # 买入信号：金叉或多头发散（短期EMA在长期EMA上方且差距扩大）
        result_df.loc[result_df['ema_golden_cross'], 'ema_signal'] = 1
        result_df.loc[(result_df['ema_21'] > result_df['ema_200']) & 
                      (result_df['ema_diverging']), 'ema_signal'] = 1
        
        # 卖出信号：死叉或空头发散（短期EMA在长期EMA下方且差距扩大）
        result_df.loc[result_df['ema_death_cross'], 'ema_signal'] = -1
        result_df.loc[(result_df['ema_21'] < result_df['ema_200']) & 
                      (result_df['ema_diverging']), 'ema_signal'] = -1
        
        # 统计信号数量
        buy_signals = (result_df['ema_signal'] == 1).sum()
        sell_signals = (result_df['ema_signal'] == -1).sum()
        logger.info(f"EMA交叉信号计算完成，买入信号: {buy_signals}个, 卖出信号: {sell_signals}个")
        
        return result_df
    
    def multi_timeframe_strategy(self, df_5min, df_15min, df_60min):
        """
        多维周期组合策略（5分钟/15分钟/60分钟）
        
        Parameters
        ----------
        df_5min : pandas.DataFrame
            5分钟K线数据
        df_15min : pandas.DataFrame
            15分钟K线数据
        df_60min : pandas.DataFrame
            60分钟K线数据
            
        Returns
        -------
        pandas.DataFrame
            添加了多维周期组合信号的数据框（基于5分钟K线）
        """
        if df_5min.empty or df_15min.empty or df_60min.empty:
            logger.warning("输入的数据为空")
            return df_5min
        
        # 确保必要的列存在
        required_cols = ["date", "close", "ma_5", "ma_20", "volume", "volume_ma_5"]
        for df, timeframe in [(df_5min, "5分钟"), (df_15min, "15分钟"), (df_60min, "60分钟")]:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"{timeframe}K线数据缺失必要的列: {missing_cols}")
                return df_5min
        
        logger.info("开始计算多维周期组合策略信号")
        
        # 复制数据，避免修改原始数据
        result_df = df_5min.copy()
        
        # 计算各个周期的趋势信号
        # 使用5日和20日移动平均线判断趋势
        
        # 5分钟周期趋势
        df_5min['trend'] = 0
        df_5min.loc[df_5min['ma_5'] > df_5min['ma_20'], 'trend'] = 1  # 上升趋势
        df_5min.loc[df_5min['ma_5'] < df_5min['ma_20'], 'trend'] = -1  # 下降趋势
        
        # 15分钟周期趋势
        df_15min['trend'] = 0
        df_15min.loc[df_15min['ma_5'] > df_15min['ma_20'], 'trend'] = 1  # 上升趋势
        df_15min.loc[df_15min['ma_5'] < df_15min['ma_20'], 'trend'] = -1  # 下降趋势
        
        # 60分钟周期趋势
        df_60min['trend'] = 0
        df_60min.loc[df_60min['ma_5'] > df_60min['ma_20'], 'trend'] = 1  # 上升趋势
        df_60min.loc[df_60min['ma_5'] < df_60min['ma_20'], 'trend'] = -1  # 下降趋势
        
        # 将15分钟和60分钟的趋势信号合并到5分钟数据中
        # 需要根据时间戳进行匹配
        
        # 确保日期列为datetime类型
        for df in [result_df, df_15min, df_60min]:
            if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
        
        # 创建用于合并的辅助列
        result_df['date_15min'] = result_df['date'].apply(
            lambda x: x.replace(minute=(x.minute // 15) * 15, second=0, microsecond=0)
        )
        
        result_df['date_60min'] = result_df['date'].apply(
            lambda x: x.replace(minute=0, second=0, microsecond=0)
        )
        
        # 准备15分钟和60分钟的趋势数据用于合并
        df_15min_trend = df_15min[['date', 'trend']].rename(
            columns={'date': 'date_15min', 'trend': 'trend_15min'}
        )
        
        df_60min_trend = df_60min[['date', 'trend']].rename(
            columns={'date': 'date_60min', 'trend': 'trend_60min'}
        )
        
        # 合并趋势数据
        result_df = pd.merge(result_df, df_15min_trend, on='date_15min', how='left')
        result_df = pd.merge(result_df, df_60min_trend, on='date_60min', how='left')
        
        # 填充可能的缺失值
        result_df['trend_15min'] = result_df['trend_15min'].fillna(0)
        result_df['trend_60min'] = result_df['trend_60min'].fillna(0)
        
        # 重命名5分钟趋势列
        result_df.rename(columns={'trend': 'trend_5min'}, inplace=True)
        
        # 计算综合趋势得分 (-3 到 3)
        result_df['trend_score'] = result_df['trend_5min'] + result_df['trend_15min'] + result_df['trend_60min']
        
        # 计算多维周期组合信号
        # 1 = 买入信号（趋势得分 >= 2，即至少两个周期为上升趋势）
        # -1 = 卖出信号（趋势得分 <= -2，即至少两个周期为下降趋势）
        # 0 = 无信号
        
        result_df['multi_timeframe_signal'] = 0
        result_df.loc[result_df['trend_score'] >= 2, 'multi_timeframe_signal'] = 1
        result_df.loc[result_df['trend_score'] <= -2, 'multi_timeframe_signal'] = -1
        
        # 统计信号数量
        buy_signals = (result_df['multi_timeframe_signal'] == 1).sum()
        sell_signals = (result_df['multi_timeframe_signal'] == -1).sum()
        logger.info(f"多维周期组合策略信号计算完成，买入信号: {buy_signals}个, 卖出信号: {sell_signals}个")
        
        # 删除辅助列
        result_df.drop(['date_15min', 'date_60min'], axis=1, inplace=True)
        
        return result_df
    
    def ema_channel_reversal(self, df):
        """
        对冲型反转策略（基于144日EMA均线通道的突破交易）
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含价格和技术指标的数据框
            
        Returns
        -------
        pandas.DataFrame
            添加了EMA通道反转信号的数据框
        """
        if df.empty:
            logger.warning("输入的数据为空")
            return df
        
        # 确保必要的列存在
        required_cols = ["date", "close", "ema_144", "ema_144_upper", "ema_144_lower"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"EMA通道反转策略所需的列缺失: {missing_cols}")
            return df
        
        logger.info("开始计算EMA通道反转信号")
        
        # 复制数据，避免修改原始数据
        result_df = df.copy()
        
        # 计算价格与EMA的距离百分比
        result_df['price_to_ema_pct'] = (result_df['close'] - result_df['ema_144']) / result_df['ema_144'] * 100
        
        # 计算通道宽度百分比
        result_df['channel_width_pct'] = (result_df['ema_144_upper'] - result_df['ema_144_lower']) / result_df['ema_144'] * 100
        
        # 计算价格在通道中的相对位置 (0-1)
        result_df['channel_position'] = (result_df['close'] - result_df['ema_144_lower']) / (result_df['ema_144_upper'] - result_df['ema_144_lower'])
        
        # 计算通道突破信号
        # 上轨突破
        result_df['upper_breakout'] = (
            (result_df['close'] > result_df['ema_144_upper']) &
            (result_df['close'].shift(1) <= result_df['ema_144_upper'].shift(1))
        )
        
        # 下轨突破
        result_df['lower_breakout'] = (
            (result_df['close'] < result_df['ema_144_lower']) &
            (result_df['close'].shift(1) >= result_df['ema_144_lower'].shift(1))
        )
        
        # 计算反转信号
        # 1. 超买反转：价格突破上轨后回落
        # 2. 超卖反转：价格突破下轨后回升
        
        # 初始化信号列
        result_df['ema_reversal_signal'] = 0
        
        # 超买反转信号（做空）：价格突破上轨后回落至通道内
        for i in range(1, len(result_df)):
            # 查找过去5个周期内是否有上轨突破
            lookback = min(5, i)
            if (result_df['upper_breakout'].iloc[i-lookback:i].any() and
                result_df['close'].iloc[i] < result_df['ema_144_upper'].iloc[i] and
                result_df['close'].iloc[i-1] >= result_df['ema_144_upper'].iloc[i-1]):
                result_df.loc[result_df.index[i], 'ema_reversal_signal'] = -1
        
        # 超卖反转信号（做多）：价格突破下轨后回升至通道内
        for i in range(1, len(result_df)):
            # 查找过去5个周期内是否有下轨突破
            lookback = min(5, i)
            if (result_df['lower_breakout'].iloc[i-lookback:i].any() and
                result_df['close'].iloc[i] > result_df['ema_144_lower'].iloc[i] and
                result_df['close'].iloc[i-1] <= result_df['ema_144_lower'].iloc[i-1]):
                result_df.loc[result_df.index[i], 'ema_reversal_signal'] = 1
        
        # 统计信号数量
        buy_signals = (result_df['ema_reversal_signal'] == 1).sum()
        sell_signals = (result_df['ema_reversal_signal'] == -1).sum()
        logger.info(f"EMA通道反转信号计算完成，买入信号: {buy_signals}个, 卖出信号: {sell_signals}个")
        
        return result_df
    
    def volume_price_confirmation(self, df):
        """
        量价齐升确认（成交量>5日均量*1.2）
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含价格和成交量数据的数据框
            
        Returns
        -------
        pandas.DataFrame
            添加了量价齐升确认信号的数据框
        """
        if df.empty:
            logger.warning("输入的数据为空")
            return df
        
        # 确保必要的列存在
        required_cols = ["date", "close", "volume", "volume_ma_5"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"量价齐升确认策略所需的列缺失: {missing_cols}")
            return df
        
        logger.info("开始计算量价齐升确认信号")
        
        # 复制数据，避免修改原始数据
        result_df = df.copy()
        
        # 计算价格变化
        result_df['price_change'] = result_df['close'].pct_change()
        
        # 计算成交量是否超过5日均量的阈值
        result_df['volume_surge'] = result_df['volume'] > (result_df['volume_ma_5'] * self.volume_threshold)
        
        # 计算量价齐升确认信号
        # 1 = 买入信号（价格上涨且成交量放大）
        # 0 = 无信号
        
        result_df['volume_price_signal'] = 0
        result_df.loc[(result_df['price_change'] > 0) &
                      (result_df['volume_surge']), 'volume_price_signal'] = 1
        
        # 统计信号数量
        buy_signals = (result_df['volume_price_signal'] == 1).sum()
        logger.info(f"量价齐升确认信号计算完成，买入信号: {buy_signals}个")
        
        return result_df
    
    def close_price_execution(self, df, signal_col):
        """
        K线收盘价交易执行（减少滑点冲击）
        
        支持小数值信号处理：
        - 1.0: 完全买入信号（标准仓位）
        - 0.5: 试探性买入信号（小仓位）
        - -0.5: 减仓信号（部分卖出）
        - -1.0: 完全卖出信号（清仓）
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含价格和信号数据的数据框
        signal_col : str
            信号列的名称
            
        Returns
        -------
        pandas.DataFrame
            添加了收盘价交易执行信号的数据框
        """
        if df.empty:
            logger.warning("输入的数据为空")
            return df
        
        # 确保必要的列存在
        required_cols = ["date", "close", signal_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"收盘价交易执行策略所需的列缺失: {missing_cols}")
            return df
        
        logger.info(f"开始计算收盘价交易执行信号，基于信号列: {signal_col}")
        
        # 复制数据，避免修改原始数据
        result_df = df.copy()
        
        # 创建执行价格列
        result_df['execution_price'] = np.nan
        
        # 对于有信号的K线，使用收盘价作为执行价格
        result_df.loc[result_df[signal_col] != 0, 'execution_price'] = result_df.loc[result_df[signal_col] != 0, 'close']
        
        # 创建执行信号列（与原信号相同）
        execution_col = f"{signal_col}_execution"
        result_df[execution_col] = result_df[signal_col]
        
        # 创建仓位大小列（根据信号强度）
        position_col = f"{signal_col}_position"
        result_df[position_col] = 0.0
        
        # 设置不同信号对应的仓位大小
        result_df.loc[result_df[signal_col] == 1, position_col] = 1.0      # 完全买入（标准仓位）
        result_df.loc[result_df[signal_col] == 0.5, position_col] = 0.3    # 试探性买入（小仓位，30%）
        result_df.loc[result_df[signal_col] == -0.5, position_col] = -0.5  # 减仓（减少50%仓位）
        result_df.loc[result_df[signal_col] == -1, position_col] = -1.0    # 完全卖出（清仓）
        
        # 统计信号数量和类型
        full_buy = (result_df[execution_col] == 1).sum()
        small_buy = (result_df[execution_col] == 0.5).sum()
        reduce = (result_df[execution_col] == -0.5).sum()
        full_sell = (result_df[execution_col] == -1).sum()
        
        logger.info(f"收盘价交易执行信号计算完成，完全买入: {full_buy}个, 试探性买入: {small_buy}个, "
                   f"减仓: {reduce}个, 完全卖出: {full_sell}个")
        
        return result_df
    
    def combine_signals(self, df):
        """
        组合多个策略信号，生成最终交易信号
        
        根据策略类型分配仓位：
        - 趋势跟踪为主（60-80%仓位）：布林带和EMA信号
        - 多周期策略优化入场点（20-30%）：多维周期组合策略
        - 反转策略作为风险对冲（5-10%）：EMA通道反转策略
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含多个策略信号的数据框
            
        Returns
        -------
        pandas.DataFrame
            添加了组合信号的数据框
        """
        if df.empty:
            logger.warning("输入的数据为空")
            return df
        
        # 首先调用各个信号函数生成必要的信号列
        logger.info("开始生成各个策略信号")
        
        # 生成布林带突破信号
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns and 'bb_middle' in df.columns:
            df = self.bollinger_bands_breakout(df)
        
        # 生成EMA交叉信号
        if 'ema_21' in df.columns and 'ema_200' in df.columns:
            df = self.ema_crossover_signal(df)
        
        # 生成EMA通道反转信号
        if 'ema_144' in df.columns and 'ema_144_upper' in df.columns and 'ema_144_lower' in df.columns:
            df = self.ema_channel_reversal(df)
        
        # 生成量价齐升确认信号
        if 'volume' in df.columns and 'volume_ma_5' in df.columns:
            df = self.volume_price_confirmation(df)
        
        # 确保必要的信号列存在
        signal_cols = [
            'bb_signal', 'ema_signal', 'multi_timeframe_signal',
            'ema_reversal_signal', 'volume_price_signal'
        ]
        
        available_signals = [col for col in signal_cols if col in df.columns]
        
        if not available_signals:
            logger.error("没有可用的策略信号列")
            return df
        
        logger.info(f"开始组合策略信号，可用信号: {available_signals}")
        
        # 复制数据，避免修改原始数据
        result_df = df.copy()
        
        # 按照策略类型分组
        trend_signals = ['bb_signal', 'ema_signal']
        multi_tf_signals = ['multi_timeframe_signal']
        reversal_signals = ['ema_reversal_signal']
        confirmation_signals = ['volume_price_signal']
        
        # 初始化各类型策略的得分
        result_df['trend_score'] = 0
        result_df['multi_tf_score'] = 0
        result_df['reversal_score'] = 0
        result_df['confirmation_score'] = 0
        
        # 使用配置文件中的信号权重
        weights = self.signal_weights
        
        # 计算各类型策略的加权得分
        for col in available_signals:
            if col in weights:
                if col in trend_signals:
                    result_df['trend_score'] += result_df[col] * weights[col]
                elif col in multi_tf_signals:
                    result_df['multi_tf_score'] += result_df[col] * weights[col]
                elif col in reversal_signals:
                    result_df['reversal_score'] += result_df[col] * weights[col]
                elif col in confirmation_signals:
                    result_df['confirmation_score'] += result_df[col] * weights[col]
        
        # 使用从配置中读取的策略类型仓位权重
        trend_weight = self.trend_weight      # 趋势跟踪 (60-80%)
        multi_tf_weight = self.multi_tf_weight  # 多周期策略 (20-30%)
        reversal_weight = self.reversal_weight  # 反转策略 (5-10%)
        
        # 计算综合信号得分
        result_df['signal_score'] = (
            result_df['trend_score'] * trend_weight +
            result_df['multi_tf_score'] * multi_tf_weight +
            result_df['reversal_score'] * reversal_weight
        )
        
        # 成交量确认可以作为额外的过滤条件
        result_df['volume_confirmed'] = result_df['confirmation_score'] > 0
        
        # 生成最终交易信号
        # 1 = 买入信号（得分 > 0 且至少一个主要策略有信号）
        # -1 = 卖出信号（得分 < 0 且至少一个主要策略有信号）
        # 0 = 无信号
        
        result_df['final_signal'] = 0
        
        # 买入条件：综合得分为正，且至少有一个主要策略（趋势或多周期）发出买入信号
        buy_condition = (
            (result_df['signal_score'] > 0) &
            ((result_df['trend_score'] > 0) | (result_df['multi_tf_score'] > 0))
        )
        
        # 卖出条件：综合得分为负，且至少有一个主要策略（趋势或多周期）发出卖出信号
        sell_condition = (
            (result_df['signal_score'] < 0) &
            ((result_df['trend_score'] < 0) | (result_df['multi_tf_score'] < 0))
        )
        
        # 反转策略的特殊处理：当主要趋势为强烈上升但反转信号出现时，减仓但不完全卖出
        reversal_hedge_condition = (
            (result_df['trend_score'] > 1.5) &
            (result_df['reversal_score'] < -1.0)
        )
        
        # 反转策略的特殊处理：当主要趋势为强烈下跌但反转信号出现时，小仓位试探性买入
        reversal_buy_condition = (
            (result_df['trend_score'] < -1.5) &
            (result_df['reversal_score'] > 1.0)
        )
        
        # 设置最终信号
        result_df.loc[buy_condition, 'final_signal'] = 1
        result_df.loc[sell_condition, 'final_signal'] = -1
        
        # 反转策略对冲信号（减仓信号，用-0.5表示）
        result_df.loc[reversal_hedge_condition, 'final_signal'] = -0.5
        
        # 反转策略试探性买入信号（小仓位，用0.5表示）
        result_df.loc[reversal_buy_condition, 'final_signal'] = 0.5
        
        # 成交量确认：如果没有成交量确认，可以降低信号强度
        result_df.loc[(result_df['final_signal'] != 0) & (~result_df['volume_confirmed']), 'final_signal'] *= 0.8
        
        # 应用收盘价交易执行
        result_df = self.close_price_execution(result_df, 'final_signal')
        
        # 统计信号数量
        buy_signals = (result_df['final_signal'] > 0).sum()
        sell_signals = (result_df['final_signal'] < 0).sum()
        logger.info(f"组合信号计算完成，买入信号: {buy_signals}个, 卖出信号: {sell_signals}个")
        logger.info(f"其中完全买入信号: {(result_df['final_signal'] == 1).sum()}个, "
                   f"试探性买入信号: {(result_df['final_signal'] == 0.5).sum()}个, "
                   f"完全卖出信号: {(result_df['final_signal'] == -1).sum()}个, "
                   f"减仓信号: {(result_df['final_signal'] == -0.5).sum()}个")
        
        return result_df


# 测试代码
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建策略实例
    strategy = TrendStrategy()
    
    # 测试策略
    print("趋势策略模块测试完成")
