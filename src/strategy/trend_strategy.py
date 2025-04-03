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
        
        # MACD参数
        self.macd_short_period = trend_config.get('macd_short_period', 21)
        self.macd_long_period = trend_config.get('macd_long_period', 200)
        
        # 多维周期参数
        self.timeframes = trend_config.get('timeframes', [5, 15, 60])
        
        # 对冲型反转策略参数
        self.ema_channel_period = trend_config.get('ema_channel_period', 144)
        self.ema_channel_width = trend_config.get('ema_channel_width', 0.05)
        
        # 成交量阈值
        self.volume_threshold = trend_config.get('volume_threshold', 1.2)
        
        # 信号组合权重
        default_weights = {
            'bb_signal': 1.0,
            'macd_signal': 1.5,
            'multi_timeframe_signal': 2.0,
            'ema_reversal_signal': 1.2,
            'volume_price_signal': 0.8
        }
        self.signal_weights = trend_config.get('signal_weights', default_weights)
        
        logger.info(f"趋势策略初始化完成，参数：快速MA={self.fast_ma}, 慢速MA={self.slow_ma}, "
                   f"布林带周期={self.bollinger_period}, 布林带标准差={self.bollinger_std_dev}, "
                   f"MACD短期={self.macd_short_period}, MACD长期={self.macd_long_period}, "
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
        required_cols = ["date", "close", "bb_upper", "bb_lower", "bb_middle"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"布林带突破策略所需的列缺失: {missing_cols}")
            return df
        
        logger.info("开始计算布林带突破信号")
        
        # 复制数据，避免修改原始数据
        result_df = df.copy()
        
        # 计算布林带突破信号
        # 1. 上轨突破: 收盘价从下方突破上轨
        # 2. 下轨突破: 收盘价从上方突破下轨
        
        # 上轨突破信号
        result_df['bb_upper_breakout'] = (
            (result_df['close'] > result_df['bb_upper']) & 
            (result_df['close'].shift(1) <= result_df['bb_upper'].shift(1))
        )
        
        # 下轨突破信号
        result_df['bb_lower_breakout'] = (
            (result_df['close'] < result_df['bb_lower']) & 
            (result_df['close'].shift(1) >= result_df['bb_lower'].shift(1))
        )
        
        # 中轨突破信号（从下向上或从上向下）
        result_df['bb_middle_up_cross'] = (
            (result_df['close'] > result_df['bb_middle']) & 
            (result_df['close'].shift(1) <= result_df['bb_middle'].shift(1))
        )
        
        result_df['bb_middle_down_cross'] = (
            (result_df['close'] < result_df['bb_middle']) & 
            (result_df['close'].shift(1) >= result_df['bb_middle'].shift(1))
        )
        
        # 综合信号
        # 1 = 买入信号（下轨突破后回升穿过中轨）
        # -1 = 卖出信号（上轨突破后回落穿过中轨）
        # 0 = 无信号
        
        # 初始化信号列
        result_df['bb_signal'] = 0
        
        # 买入信号：之前有下轨突破，现在向上穿过中轨
        for i in range(1, len(result_df)):
            # 查找过去10个周期内是否有下轨突破
            lookback = min(10, i)
            if (result_df['bb_lower_breakout'].iloc[i-lookback:i].any() and 
                result_df['bb_middle_up_cross'].iloc[i]):
                result_df.loc[result_df.index[i], 'bb_signal'] = 1
        
        # 卖出信号：之前有上轨突破，现在向下穿过中轨
        for i in range(1, len(result_df)):
            # 查找过去10个周期内是否有上轨突破
            lookback = min(10, i)
            if (result_df['bb_upper_breakout'].iloc[i-lookback:i].any() and 
                result_df['bb_middle_down_cross'].iloc[i]):
                result_df.loc[result_df.index[i], 'bb_signal'] = -1
        
        # 统计信号数量
        buy_signals = (result_df['bb_signal'] == 1).sum()
        sell_signals = (result_df['bb_signal'] == -1).sum()
        logger.info(f"布林带突破信号计算完成，买入信号: {buy_signals}个, 卖出信号: {sell_signals}个")
        
        return result_df
    
    def macd_convergence_divergence(self, df):
        """
        MACD长短周期移动平均线收敛与发散（1个月期与200日EMA）
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含价格和技术指标的数据框
            
        Returns
        -------
        pandas.DataFrame
            添加了MACD收敛发散信号的数据框
        """
        if df.empty:
            logger.warning("输入的数据为空")
            return df
        
        # 确保必要的列存在
        required_cols = ["date", "close", "ema_21", "ema_200", "ema_21_200_diff"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"MACD收敛发散策略所需的列缺失: {missing_cols}")
            return df
        
        logger.info("开始计算MACD长短周期收敛发散信号")
        
        # 复制数据，避免修改原始数据
        result_df = df.copy()
        
        # 计算差值的变化率
        result_df['ema_diff_change'] = result_df['ema_21_200_diff'].diff()
        
        # 计算收敛发散信号
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
        result_df['macd_signal'] = 0
        
        # 买入信号：金叉或多头发散（短期EMA在长期EMA上方且差距扩大）
        result_df.loc[result_df['ema_golden_cross'], 'macd_signal'] = 1
        result_df.loc[(result_df['ema_21'] > result_df['ema_200']) & 
                      (result_df['ema_diverging']), 'macd_signal'] = 1
        
        # 卖出信号：死叉或空头发散（短期EMA在长期EMA下方且差距扩大）
        result_df.loc[result_df['ema_death_cross'], 'macd_signal'] = -1
        result_df.loc[(result_df['ema_21'] < result_df['ema_200']) & 
                      (result_df['ema_diverging']), 'macd_signal'] = -1
        
        # 统计信号数量
        buy_signals = (result_df['macd_signal'] == 1).sum()
        sell_signals = (result_df['macd_signal'] == -1).sum()
        logger.info(f"MACD收敛发散信号计算完成，买入信号: {buy_signals}个, 卖出信号: {sell_signals}个")
        
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
        
        # 统计信号数量
        buy_signals = (result_df[execution_col] > 0).sum()
        sell_signals = (result_df[execution_col] < 0).sum()
        logger.info(f"收盘价交易执行信号计算完成，买入信号: {buy_signals}个, 卖出信号: {sell_signals}个")
        
        return result_df
    
    def combine_signals(self, df):
        """
        组合多个策略信号，生成最终交易信号
        
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
        
        # 确保必要的信号列存在
        signal_cols = [
            'bb_signal', 'macd_signal', 'multi_timeframe_signal',
            'ema_reversal_signal', 'volume_price_signal'
        ]
        
        available_signals = [col for col in signal_cols if col in df.columns]
        
        if not available_signals:
            logger.error("没有可用的策略信号列")
            return df
        
        logger.info(f"开始组合策略信号，可用信号: {available_signals}")
        
        # 复制数据，避免修改原始数据
        result_df = df.copy()
        
        # 计算信号得分（各个信号的加权和）
        result_df['signal_score'] = 0
        
        # 使用配置文件中的信号权重
        weights = self.signal_weights
        
        # 计算加权得分
        for col in available_signals:
            if col in weights:
                result_df['signal_score'] += result_df[col] * weights[col]
        
        # 生成最终交易信号
        # 1 = 买入信号（得分 >= 2）
        # -1 = 卖出信号（得分 <= -2）
        # 0 = 无信号
        
        result_df['final_signal'] = 0
        result_df.loc[result_df['signal_score'] >= 2, 'final_signal'] = 1
        result_df.loc[result_df['signal_score'] <= -2, 'final_signal'] = -1
        
        # 应用收盘价交易执行
        result_df = self.close_price_execution(result_df, 'final_signal')
        
        # 统计信号数量
        buy_signals = (result_df['final_signal'] == 1).sum()
        sell_signals = (result_df['final_signal'] == -1).sum()
        logger.info(f"组合信号计算完成，买入信号: {buy_signals}个, 卖出信号: {sell_signals}个")
        
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
