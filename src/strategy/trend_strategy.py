#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
趋势策略模块
实现各种趋势跟踪和反转交易策略
"""

import pandas as pd
import numpy as np
import logging
import talib
from datetime import datetime, timedelta
from pathlib import Path
import os
import sys
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

# 设置日志
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)
# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"trend_strategy_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
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
        trend_config = self.config
        logger.info(f"趋势策略配置: {trend_config}")
        # 移动平均线参数
        self.fast_ma = trend_config.get('fast_ma', 20)
        self.slow_ma = trend_config.get('slow_ma', 60)
        
        # 布林带参数
        self.bollinger_period = trend_config.get('bollinger_period', 20)
        self.bollinger_std_dev = trend_config.get('bollinger_std_dev', 2.0)
        
        # EMA参数
        self.ema_short_period = trend_config.get('ema_short_period', 21)
        self.ema_long_period = trend_config.get('ema_long_period', 200)
        
        # MACD参数
        self.macd_fast_period = trend_config.get('macd_fast_period', 12)
        self.macd_slow_period = trend_config.get('macd_slow_period', 26)
        self.macd_signal_period = trend_config.get('macd_signal_period', 9)
        
        # 多维周期参数
        self.timeframes = trend_config.get('timeframes', [5, 15, 60])
        
        # 对冲型反转策略参数
        self.ema_channel_period = trend_config.get('ema_channel_period', 144)
        self.ema_channel_width = trend_config.get('ema_channel_width', 0.05)
        
        # 成交量阈值
        self.volume_threshold = trend_config.get('volume_threshold', 1.2)
        
        # 价格形态识别参数
        self.channel_period = trend_config.get('channel_period', 30)  # 上升通道周期
        self.channel_slope_threshold = trend_config.get('channel_slope_threshold', 0.001)  # 通道斜率阈值
        self.channel_width_threshold = trend_config.get('channel_width_threshold', 0.02)  # 通道宽度阈值
        self.breakout_resistance_period = trend_config.get('breakout_resistance_period', 60)  # 突破阻力位周期
        self.breakout_volume_multiplier = trend_config.get('breakout_volume_multiplier', 1.5)  # 突破量能倍数
        self.breakout_confirmation_days = trend_config.get('breakout_confirmation_days', 2)  # 突破确认天数
        self.head_risk_peak_period = trend_config.get('head_risk_peak_period', 60)  # 顶部风险高点周期
        self.rsi_period = trend_config.get('rsi_period', 14)  # RSI周期
        self.volume_sma_period = trend_config.get('volume_sma_period', 20)  # 成交量均线周期
        
        # 移动平均线分析参数
        self.ma_periods = trend_config.get('ma_periods', [5, 10, 20, 60])  # 移动平均线周期
        self.ma_support_threshold = trend_config.get('ma_support_threshold', 0.02)  # 支撑阈值敏感度
        self.trend_strength_sensitivity = trend_config.get('trend_strength_sensitivity', 1.0)  # 趋势强度敏感度
        
        # 策略类型仓位权重
        self.trend_weight = trend_config.get('trend_weight', 0.7)  # 趋势跟踪 (60-80%)
        self.multi_tf_weight = trend_config.get('multi_tf_weight', 0.25)  # 多周期策略 (20-30%)
        self.reversal_weight = trend_config.get('reversal_weight', 0.05)  # 反转策略 (5-10%)
        
        # 信号组合权重
        default_weights = {
            'bb_signal': 2.0,            # 布林带信号权重提高（趋势跟踪核心指标）
            'ema_signal': 2.5,          # EMA信号权重提高（趋势跟踪核心指标）
            'macd_signal': 2.0,          # MACD信号权重（趋势跟踪核心指标）
            'multi_timeframe_signal': 3.0, # 多周期信号权重最高（优化入场点）
            'ema_reversal_signal': 1.5,   # 反转信号权重适中（风险对冲）
            'volume_price_signal': 1.0    # 成交量确认信号
        }
        self.signal_weights = trend_config.get('signal_weights', default_weights)
        
        logger.info(f"趋势策略初始化完成，参数：快速MA={self.fast_ma}, 慢速MA={self.slow_ma}, "
                   f"布林带周期={self.bollinger_period}, 布林带标准差={self.bollinger_std_dev}, "
                   f"EMA短期={self.ema_short_period}, EMA长期={self.ema_long_period}, "
                   f"MACD参数=({self.macd_fast_period}, {self.macd_slow_period}, {self.macd_signal_period}), "
                   f"MA分析周期={self.ma_periods}, MA支撑阈值={self.ma_support_threshold:.2%}, "
                   f"趋势强度敏感度={self.trend_strength_sensitivity}, "
                   f"EMA通道周期={self.ema_channel_period}, 成交量阈值={self.volume_threshold}, "
                   f"通道周期={self.channel_period}, 突破阻力位周期={self.breakout_resistance_period}, "
                   f"顶部风险高点周期={self.head_risk_peak_period}")
    
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
    
    def macd_signal(self, df):
        """
        MACD指标分析功能
        
        实现 DIF、DEA、MACD 柱线的计算和金叉死叉判断
        - 计算DIF, DEA, MACD柱线
        - 判断DIF线与DEA线是否均在0轴上方运行
        - 判断近期是否发生金叉，或金叉后红柱是否持续
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含价格数据的数据框
            
        Returns
        -------
        pandas.DataFrame
            添加了MACD信号的数据框
        """
        if df.empty:
            logger.warning("输入的数据为空")
            return df
        
        # 确保必要的列存在
        required_cols = ["date", "close"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"MACD策略所需的列缺失: {missing_cols}")
            return df
        
        logger.info("开始计算MACD信号")
        
        # 复制数据，避免修改原始数据
        result_df = df.copy()
        
        # 使用talib计算MACD指标
        macd_dif, macd_dea, macd_histogram = talib.MACD(
            result_df['close'].values,
            fastperiod=self.macd_fast_period,
            slowperiod=self.macd_slow_period,
            signalperiod=self.macd_signal_period
        )
        
        # 添加MACD指标到数据框
        result_df['macd_dif'] = macd_dif
        result_df['macd_dea'] = macd_dea
        result_df['macd_histogram'] = macd_histogram
        
        # 计算MACD信号
        # 1. 判断DIF和DEA是否均在0轴上方
        result_df['macd_above_zero'] = (result_df['macd_dif'] > 0) & (result_df['macd_dea'] > 0)
        
        # 2. 判断是否发生金叉（DIF从下方穿过DEA）
        result_df['macd_golden_cross'] = (
            (result_df['macd_dif'] > result_df['macd_dea']) &
            (result_df['macd_dif'].shift(1) <= result_df['macd_dea'].shift(1))
        )
        
        # 3. 判断是否发生死叉（DIF从上方穿过DEA）
        result_df['macd_death_cross'] = (
            (result_df['macd_dif'] < result_df['macd_dea']) &
            (result_df['macd_dif'].shift(1) >= result_df['macd_dea'].shift(1))
        )
        
        # 4. 判断MACD柱线是否为红柱（正值）且持续
        result_df['macd_red_bar'] = result_df['macd_histogram'] > 0
        result_df['macd_green_bar'] = result_df['macd_histogram'] < 0
        
        # 5. 判断红柱是否持续（连续2个周期以上）
        result_df['macd_red_continuing'] = (
            (result_df['macd_red_bar']) &
            (result_df['macd_red_bar'].shift(1))
        )
        
        # 6. 判断绿柱是否持续（连续2个周期以上）
        result_df['macd_green_continuing'] = (
            (result_df['macd_green_bar']) &
            (result_df['macd_green_bar'].shift(1))
        )
        
        # 初始化MACD信号列
        result_df['macd_signal'] = 0
        
        # 买入信号条件：
        # 1. DIF和DEA均在0轴上方 且 发生金叉
        # 2. 或者 DIF和DEA均在0轴上方 且 红柱持续
        buy_condition_1 = result_df['macd_above_zero'] & result_df['macd_golden_cross']
        buy_condition_2 = result_df['macd_above_zero'] & result_df['macd_red_continuing']
        
        result_df.loc[buy_condition_1 | buy_condition_2, 'macd_signal'] = 1
        
        # 卖出信号条件：
        # 1. DIF和DEA均在0轴下方 且 发生死叉
        # 2. 或者 DIF和DEA均在0轴下方 且 绿柱持续
        sell_condition_1 = (~result_df['macd_above_zero']) & result_df['macd_death_cross']
        sell_condition_2 = (~result_df['macd_above_zero']) & result_df['macd_green_continuing']
        
        result_df.loc[sell_condition_1 | sell_condition_2, 'macd_signal'] = -1
        
        # 统计信号数量
        buy_signals = (result_df['macd_signal'] == 1).sum()
        sell_signals = (result_df['macd_signal'] == -1).sum()
        logger.info(f"MACD信号计算完成，买入信号: {buy_signals}个, 卖出信号: {sell_signals}个")
        
        return result_df
    
    def moving_average_analysis(self, df, ma_periods=None, support_threshold=0.02, trend_strength_sensitivity=1.0):
        """
        综合移动平均线分析函数
        
        实现多头排列检测和关键移动平均线支撑分析，包括趋势强度评分
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含价格数据的数据框，必须包含 'close' 列
        ma_periods : list, default [5, 10, 20, 60]
            移动平均线周期列表
        support_threshold : float, default 0.02
            支撑阈值敏感度，用于判断价格是否接近移动平均线支撑位（2%）
        trend_strength_sensitivity : float, default 1.0
            趋势强度敏感度调节因子，值越大对趋势强度要求越高
            
        Returns
        -------
        pandas.DataFrame
            添加了移动平均线分析结果的数据框，包含以下列：
            - ma_5, ma_10, ma_20, ma_60: 各周期移动平均线
            - bullish_alignment: 多头排列状态 (True/False)
            - alignment_strength: 排列强度评分 (0-1)
            - ma20_support: MA20支撑状态 (True/False)
            - ma60_support: MA60支撑状态 (True/False)
            - key_ma_support: 关键移动平均线支撑综合评分 (0-1)
            - trend_strength_score: 趋势强度评分 (0-1)
            - ma_signal: 移动平均线综合信号 (-1, 0, 1)
        """
        if df.empty:
            logger.warning("移动平均线分析：输入数据为空")
            return df
            
        # 验证必要的列
        required_cols = ['close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"移动平均线分析所需的列缺失: {missing_cols}")
            return df
            
        # 设置默认移动平均线周期
        if ma_periods is None:
            ma_periods = [5, 10, 20, 60]
            
        # 验证数据长度是否足够
        max_period = max(ma_periods)
        if len(df) < max_period:
            logger.warning(f"数据长度不足，需要至少{max_period}个周期进行移动平均线分析")
            return df
            
        logger.info(f"开始移动平均线分析，周期: {ma_periods}, 支撑阈值: {support_threshold:.2%}")
        
        # 复制数据避免修改原始数据
        result_df = df.copy()
        
        try:
            # ===== 第一步：计算各周期移动平均线 =====
            ma_columns = {}
            for period in ma_periods:
                col_name = f'ma_{period}'
                result_df[col_name] = talib.SMA(result_df['close'].values, timeperiod=period)
                ma_columns[period] = col_name
                
            # ===== 第二步：多头排列检测 =====
            result_df = self._detect_bullish_alignment(result_df, ma_periods, ma_columns)
            
            # ===== 第三步：关键移动平均线支撑分析 =====
            result_df = self._analyze_key_ma_support(result_df, ma_columns, support_threshold)
            
            # ===== 第四步：趋势强度评分 =====
            result_df = self._calculate_trend_strength_score(result_df, ma_periods, ma_columns, trend_strength_sensitivity)
            
            # 统计分析结果
            bullish_count = result_df['bullish_alignment'].sum()
            ma20_support_count = result_df['ma20_support'].sum()
            ma60_support_count = result_df['ma60_support'].sum()
            
            avg_trend_strength = result_df['trend_strength_score'].mean()
            max_trend_strength = result_df['trend_strength_score'].max()
            
            logger.info(f"=== 移动平均线分析结果 ===")
            logger.info(f"多头排列次数: {bullish_count}/{len(result_df)} ({bullish_count/len(result_df):.1%})")
            logger.info(f"MA20支撑次数: {ma20_support_count}/{len(result_df)} ({ma20_support_count/len(result_df):.1%})")
            logger.info(f"MA60支撑次数: {ma60_support_count}/{len(result_df)} ({ma60_support_count/len(result_df):.1%})")
            logger.info(f"趋势强度 - 平均: {avg_trend_strength:.3f}, 最大: {max_trend_strength:.3f}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"移动平均线分析过程中发生错误: {e}")
            return df
    
    def _detect_bullish_alignment(self, df, ma_periods, ma_columns):
        """
        检测多头排列
        
        多头排列定义：
        1. 当前价格 > MA20 > MA60 (核心条件)
        2. 理想排列：MA5 > MA10 > MA20 > MA60 (完美多头排列)
        3. 降级排列：至少满足 MA20 > MA60 (基本多头排列)
        """
        try:
            # 初始化列
            df['bullish_alignment'] = False
            df['alignment_strength'] = 0.0
            df['alignment_type'] = '无排列'
            
            # 确保所有需要的MA列都存在
            required_mas = [5, 10, 20, 60]
            available_mas = [period for period in required_mas if period in ma_periods]
            
            if len(available_mas) < 2:
                logger.warning("移动平均线周期不足，无法进行多头排列分析")
                return df
                
            # 核心条件：价格 > MA20 且 价格 > MA60
            if 20 in available_mas and 60 in available_mas:
                core_condition = (df['close'] > df[ma_columns[20]]) & (df['close'] > df[ma_columns[60]])
                
                # 基本多头排列：MA20 > MA60
                basic_alignment = core_condition & (df[ma_columns[20]] > df[ma_columns[60]])
                
                # 完美多头排列：MA5 > MA10 > MA20 > MA60
                if all(period in available_mas for period in [5, 10, 20, 60]):
                    perfect_alignment = (basic_alignment &
                                       (df[ma_columns[5]] > df[ma_columns[10]]) &
                                       (df[ma_columns[10]] > df[ma_columns[20]]) &
                                       (df[ma_columns[20]] > df[ma_columns[60]]))
                    
                    # 设置排列状态和强度
                    df.loc[perfect_alignment, 'bullish_alignment'] = True
                    df.loc[perfect_alignment, 'alignment_strength'] = 1.0
                    df.loc[perfect_alignment, 'alignment_type'] = '完美多头排列'
                    
                    # 基本多头排列（非完美）
                    basic_only = basic_alignment & (~perfect_alignment)
                    df.loc[basic_only, 'bullish_alignment'] = True
                    df.loc[basic_only, 'alignment_strength'] = 0.6
                    df.loc[basic_only, 'alignment_type'] = '基本多头排列'
                    
                else:
                    # 只有基本排列条件
                    df.loc[basic_alignment, 'bullish_alignment'] = True
                    df.loc[basic_alignment, 'alignment_strength'] = 0.6
                    df.loc[basic_alignment, 'alignment_type'] = '基本多头排列'
                    
            return df
            
        except Exception as e:
            logger.error(f"检测多头排列时发生错误: {e}")
            return df
    
    def _analyze_key_ma_support(self, df, ma_columns, support_threshold):
        """
        分析关键移动平均线支撑
        
        重点关注MA20和MA60作为中长期支撑位
        """
        try:
            # 初始化支撑列
            df['ma20_support'] = False
            df['ma60_support'] = False
            df['key_ma_support'] = 0.0
            
            # MA20支撑分析
            if 20 in ma_columns:
                ma20_distance = abs(df['close'] - df[ma_columns[20]]) / df[ma_columns[20]]
                df['ma20_support'] = (df['close'] >= df[ma_columns[20]]) & (ma20_distance <= support_threshold)
                
            # MA60支撑分析
            if 60 in ma_columns:
                ma60_distance = abs(df['close'] - df[ma_columns[60]]) / df[ma_columns[60]]
                df['ma60_support'] = (df['close'] >= df[ma_columns[60]]) & (ma60_distance <= support_threshold)
                
            # 综合支撑评分
            support_score = 0.0
            if 20 in ma_columns:
                support_score += df['ma20_support'].astype(float) * 0.4  # MA20权重40%
            if 60 in ma_columns:
                support_score += df['ma60_support'].astype(float) * 0.6  # MA60权重60%
                
            df['key_ma_support'] = support_score
            
            return df
            
        except Exception as e:
            logger.error(f"分析关键移动平均线支撑时发生错误: {e}")
            return df
    
    def _calculate_trend_strength_score(self, df, ma_periods, ma_columns, sensitivity):
        """
        计算趋势强度评分
        
        基于移动平均线分离度和价格相对位置
        """
        try:
            df['trend_strength_score'] = 0.0
            
            if len(ma_periods) < 2:
                return df
                
            # 计算移动平均线分离度
            separation_scores = []
            
            # 短期与长期MA的分离度
            if 5 in ma_columns and 60 in ma_columns:
                ma5_ma60_sep = (df[ma_columns[5]] - df[ma_columns[60]]) / df[ma_columns[60]]
                separation_scores.append(np.clip(ma5_ma60_sep * 10, -1, 1))  # 标准化到[-1,1]
                
            # 中期MA分离度
            if 10 in ma_columns and 20 in ma_columns:
                ma10_ma20_sep = (df[ma_columns[10]] - df[ma_columns[20]]) / df[ma_columns[20]]
                separation_scores.append(np.clip(ma10_ma20_sep * 20, -1, 1))
                
            # 价格相对于各MA的位置强度
            price_position_scores = []
            for period in ma_periods:
                if period in ma_columns:
                    price_ma_ratio = (df['close'] - df[ma_columns[period]]) / df[ma_columns[period]]
                    # 根据MA周期调整权重，长期MA权重更高
                    weight = period / 60.0  # 以60日MA为基准
                    weighted_score = np.clip(price_ma_ratio * 10 * weight, -1, 1)
                    price_position_scores.append(weighted_score)
                    
            # 综合评分
            if separation_scores:
                avg_separation = np.mean(separation_scores, axis=0)
            else:
                avg_separation = 0
                
            if price_position_scores:
                avg_position = np.mean(price_position_scores, axis=0)
            else:
                avg_position = 0
                
            # 最终趋势强度评分（0-1范围）
            raw_strength = (avg_separation * 0.4 + avg_position * 0.6) * sensitivity
            df['trend_strength_score'] = np.clip((raw_strength + 1) / 2, 0, 1)  # 转换到[0,1]
            
            return df
            
        except Exception as e:
            logger.error(f"计算趋势强度评分时发生错误: {e}")
            return df
    
    
    
    # def ema_channel_reversal(self, df):
    #     """
    #     对冲型反转策略（基于144日EMA均线通道的反转交易）
        
    #     Parameters
    #     ----------
    #     df : pandas.DataFrame
    #         包含价格和技术指标的数据框
            
    #     Returns
    #     -------
    #     pandas.DataFrame
    #         添加了EMA通道反转信号的数据框
    #     """
    #     if df.empty:
    #         logger.warning("输入的数据为空")
    #         return df
        
    #     # 确保必要的列存在
    #     required_cols = ["date", "close", "ema_144", "ema_144_upper", "ema_144_lower"]
    #     missing_cols = [col for col in required_cols if col not in df.columns]
    #     if missing_cols:
    #         logger.error(f"EMA通道反转策略所需的列缺失: {missing_cols}")
    #         return df
        
    #     logger.info("开始计算EMA通道反转信号")
        
    #     # 复制数据，避免修改原始数据
    #     result_df = df.copy()
        
    #     # 计算价格与EMA的距离百分比
    #     result_df['price_to_ema_pct'] = (result_df['close'] - result_df['ema_144']) / result_df['ema_144'] * 100
        
    #     # 计算通道宽度百分比
    #     result_df['channel_width_pct'] = (result_df['ema_144_upper'] - result_df['ema_144_lower']) / result_df['ema_144'] * 100
        
    #     # 计算价格在通道中的相对位置 (0-1)
    #     result_df['channel_position'] = (result_df['close'] - result_df['ema_144_lower']) / (result_df['ema_144_upper'] - result_df['ema_144_lower'])
        
    #     # 计算通道突破信号
    #     # 上轨突破
    #     result_df['upper_breakout'] = (
    #         (result_df['close'] > result_df['ema_144_upper']) &
    #         (result_df['close'].shift(1) <= result_df['ema_144_upper'].shift(1))
    #     )
        
    #     # 下轨突破
    #     result_df['lower_breakout'] = (
    #         (result_df['close'] < result_df['ema_144_lower']) &
    #         (result_df['close'].shift(1) >= result_df['ema_144_lower'].shift(1))
    #     )
        
    #     # 初始化信号列
    #     result_df['ema_reversal_signal'] = 0
        
    #     # 向量化实现超买反转信号（做空）：价格突破上轨后回落至通道内
    #     # 创建一个滚动窗口来检测过去5个周期内是否有上轨突破
    #     result_df['upper_breakout_5d'] = result_df['upper_breakout'].rolling(window=5, min_periods=1).max()
        
    #     # 超买反转条件：过去5天内有上轨突破，且当前价格回落到通道内，前一天价格还在通道外
    #     result_df.loc[
    #         (result_df['upper_breakout_5d'] > 0) &
    #         (result_df['close'] < result_df['ema_144_upper']) &
    #         (result_df['close'].shift(1) >= result_df['ema_144_upper']),
    #         'ema_reversal_signal'
    #     ] = -1
        
    #     # 向量化实现超卖反转信号（做多）：价格突破下轨后回升至通道内
    #     # 创建一个滚动窗口来检测过去5个周期内是否有下轨突破
    #     result_df['lower_breakout_5d'] = result_df['lower_breakout'].rolling(window=5, min_periods=1).max()
        
    #     # 超卖反转条件：过去5天内有下轨突破，且当前价格回升到通道内，前一天价格还在通道外
    #     result_df.loc[
    #         (result_df['lower_breakout_5d'] > 0) &
    #         (result_df['close'] > result_df['ema_144_lower']) &
    #         (result_df['close'].shift(1) <= result_df['ema_144_lower']),
    #         'ema_reversal_signal'
    #     ] = 1
        
    #     # 删除临时列
    #     result_df.drop(['upper_breakout_5d', 'lower_breakout_5d'], axis=1, inplace=True)
        
    #     # 统计信号数量
    #     buy_signals = (result_df['ema_reversal_signal'] == 1).sum()
    #     sell_signals = (result_df['ema_reversal_signal'] == -1).sum()
    #     logger.info(f"EMA通道反转信号计算完成，买入信号: {buy_signals}个, 卖出信号: {sell_signals}个")
        
    #     return result_df
    
    def volume_price_confirmation(self, df):
        """
        基于四个阶段的量价关系细化分析系统
        
        核心思想：将成交量与价格的不同阶段（上涨、下跌、盘整、突破）结合起来分析
        
        量价关系细化规则：
        1. 上涨阶段 (Uptrend Phase):
           - 健康状态 (加分): 价涨量增 - 量价配合
           - 警示状态 (减分): 价涨量缩 - 追高意愿不足，可能是上涨末期
           - 警示状态 (减分): 价滞量增 - 主力可能在出货
        
        2. 突破阶段 (Breakout Phase):
           - 强信号 (强力加分): 价升量增 - 突破伴随成交量显著放大(>1.5倍均量)
           - 假信号 (减分/无效): 价升量缩 - 假突破概率高
        
        3. 回调/盘整阶段 (Consolidation Phase):
           - 健康状态 (加分/观望): 价跌量缩 - 洗盘而非出货，未来买入机会
           - 警示状态 (减分): 价跌量增 - 恐慌盘或主力出货，趋势可能反转
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含价格和成交量数据的数据框
            
        Returns
        -------
        pandas.DataFrame
            添加了量价确认信号的数据框
        """
        if df.empty:
            logger.warning("输入的数据为空")
            return df
        
        # 确保必要的列存在
        required_cols = ["date", "close", "high", "low", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"量价确认策略所需的列缺失: {missing_cols}")
            return df
        
        logger.info("开始计算基于四个阶段的量价关系分析")
        
        # 复制数据，避免修改原始数据
        result_df = df.copy()
        
        # ===== 第一步：基础指标计算 =====
        result_df = self._calculate_basic_volume_price_indicators(result_df)
        
        # ===== 第二步：市场阶段识别 =====
        result_df = self._identify_simplified_market_phases(result_df)
        
        # ===== 第三步：基于阶段的条件量价分析（优化版本） =====
        result_df = self._analyze_volume_price_by_stage(result_df)
        
        # ===== 第四步：综合量价信号生成 =====
        result_df = self._generate_final_volume_price_signal(result_df)
        
        # 计算量价分析质量评分
        total_periods = len(result_df)
        valid_analysis_periods = (result_df['volume_price_signal'] != 0).sum()
        analysis_coverage = valid_analysis_periods / total_periods if total_periods > 0 else 0
        
        # 各阶段识别统计
        uptrend_periods = (result_df['market_phase'] == 'uptrend').sum()
        breakout_periods = (result_df['market_phase'] == 'breakout').sum()
        consolidation_periods = (result_df['market_phase'] == 'consolidation').sum()
        retracement_periods = (result_df['market_phase'] == 'retracement').sum()
        
        # 量价分析质量评分
        avg_signal_strength = abs(result_df['volume_price_signal']).mean()
        max_signal_strength = abs(result_df['volume_price_signal']).max()
        
        logger.info(f"=== 四阶段量价分析评估 ===")
        logger.info(f"分析覆盖率: {analysis_coverage:.2%} ({valid_analysis_periods}/{total_periods})")
        logger.info(f"阶段分布 - 上涨: {uptrend_periods}, 突破: {breakout_periods}, 盘整: {consolidation_periods}, 回调: {retracement_periods}")
        logger.info(f"量价分析质量 - 平均强度: {avg_signal_strength:.3f}, 最大强度: {max_signal_strength:.3f}")
        logger.info(f"分析完成，生成量价评分列: volume_price_signal")
        
        return result_df
    
    def _calculate_basic_volume_price_indicators(self, df):
        """计算基础量价指标"""
        try:
            # 价格变化指标
            df['price_change'] = df['close'].pct_change()
            df['price_change_3d'] = df['close'].pct_change(3)
            df['price_change_5d'] = df['close'].pct_change(5)
            
            # 成交量均线和比率
            df['volume_ma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()
            df['volume_ratio_20d'] = df['volume'] / df['volume_ma_20']
            
            # 价格阻力和支撑水平（用于判断突破）
            df['resistance_level'] = df['high'].rolling(window=20, min_periods=1).max()
            df['support_level'] = df['low'].rolling(window=20, min_periods=1).min()
            df['price_position'] = (df['close'] - df['support_level']) / (df['resistance_level'] - df['support_level'])
            
            # 价格趋势方向（简化版）
            df['price_trend_3d'] = np.where(df['price_change_3d'] > 0.01, 1,
                                          np.where(df['price_change_3d'] < -0.01, -1, 0))
            
            return df
            
        except Exception as e:
            logger.error(f"计算基础量价指标时发生错误: {e}")
            return df
    
    def _identify_simplified_market_phases(self, df):
        """简化的市场阶段识别 - 专注于四个关键阶段"""
        try:
            # 初始化阶段标识
            df['market_phase'] = 'consolidation'  # 默认为盘整
            
            # 1. 突破阶段（最高优先级）
            breakout_conditions = (
                (df['close'] > df['resistance_level'].shift(1)) &  # 突破前期阻力
                (df['price_change'] > 0.01)  # 当日上涨超过1%
            )
            
            # 2. 上涨阶段
            uptrend_conditions = (
                (df['price_change_5d'] > 0.02) &  # 5日涨幅超过2%
                (df['price_trend_3d'] == 1) &    # 3日趋势向上
                (df['price_position'] > 0.5)     # 价格位置在上半部
            ) & (~breakout_conditions)  # 排除突破阶段
            
            # 3. 回调阶段
            retracement_conditions = (
                (df['price_change_3d'] < -0.01) &  # 3日跌幅超过1%
                (df['price_trend_3d'] == -1)       # 3日趋势向下
            ) & (~breakout_conditions) & (~uptrend_conditions)  # 排除其他阶段
            
            # 设置阶段标识（按优先级顺序）
            df.loc[breakout_conditions, 'market_phase'] = 'breakout'
            df.loc[uptrend_conditions, 'market_phase'] = 'uptrend'
            df.loc[retracement_conditions, 'market_phase'] = 'retracement'
            
            return df
            
        except Exception as e:
            logger.error(f"识别市场阶段时发生错误: {e}")
            return df
    
    def _analyze_uptrend_volume_price(self, df):
        """上涨阶段量价分析（优化版本）
        
        上涨阶段 (Uptrend Phase):
        - 健康状态 (加分): 价涨量增 - 量价配合
        - 警示状态 (减分): 价涨量缩 - 追高意愿不足，可能是上涨末期
        - 警示状态 (减分): 价滞量增 - 主力可能在出货
        """
        try:
            # 如果uptrend_vp_score列不存在，初始化
            if 'uptrend_vp_score' not in df.columns:
                df['uptrend_vp_score'] = 0.0
            
            # 仅在上涨阶段进行分析，提前过滤
            uptrend_mask = (df['market_phase'] == 'uptrend')
            if not uptrend_mask.any():
                return df  # 没有上涨阶段数据，直接返回
            
            # 向量化计算所有条件
            price_up = (df['price_change'] > 0)
            volume_expand = (df['volume_ratio_20d'] > 1.2)
            volume_shrink = (df['volume_ratio_20d'] < 0.8)
            high_position = (df['price_position'] > 0.8)
            price_stagnant = (abs(df['price_change']) < 0.005)
            volume_surge = (df['volume_ratio_20d'] > 1.5)
            
            # 1. 健康状态：价涨量增（量价配合）- 加分
            healthy_uptrend = uptrend_mask & price_up & volume_expand
            df.loc[healthy_uptrend, 'uptrend_vp_score'] = 0.6
            
            # 2. 警示状态：价涨量缩（追高意愿不足）- 减分
            warning_uptrend_shrink = uptrend_mask & price_up & volume_shrink
            df.loc[warning_uptrend_shrink, 'uptrend_vp_score'] = -0.4
            
            # 3. 警示状态：价滞量增（主力出货）- 减分
            warning_stagnant_volume = uptrend_mask & high_position & price_stagnant & volume_surge
            df.loc[warning_stagnant_volume, 'uptrend_vp_score'] = -0.5
            
            return df
            
        except Exception as e:
            logger.error(f"分析上涨阶段量价关系时发生错误: {e}")
            return df
    
    def _analyze_breakout_volume_price(self, df):
        """突破阶段量价分析（优化版本）
        
        突破阶段 (Breakout Phase):
        - 强信号 (强力加分): 价升量增 - 突破伴随成交量显著放大(>1.5倍均量)
        - 假信号 (减分/无效): 价升量缩 - 假突破概率高
        """
        try:
            # 如果breakout_vp_score列不存在，初始化
            if 'breakout_vp_score' not in df.columns:
                df['breakout_vp_score'] = 0.0
            
            # 仅在突破阶段进行分析，提前过滤
            breakout_mask = (df['market_phase'] == 'breakout')
            if not breakout_mask.any():
                return df  # 没有突破阶段数据，直接返回
            
            # 向量化计算所有条件
            price_rise = (df['price_change'] > 0.01)
            volume_surge = (df['volume_ratio_20d'] > 1.5)
            volume_shrink = (df['volume_ratio_20d'] < 0.9)
            
            # 1. 强信号：价升量增（最可靠的买入信号）- 强力加分
            strong_breakout = breakout_mask & price_rise & volume_surge
            df.loc[strong_breakout, 'breakout_vp_score'] = 0.8
            
            # 2. 假信号：价升量缩（假突破）- 减分/无效
            false_breakout = breakout_mask & price_rise & volume_shrink
            df.loc[false_breakout, 'breakout_vp_score'] = -0.6
            
            return df
            
        except Exception as e:
            logger.error(f"分析突破阶段量价关系时发生错误: {e}")
            return df
    
    def _analyze_consolidation_volume_price(self, df):
        """回调/盘整阶段量价分析（优化版本）
        
        回调/盘整阶段 (Consolidation Phase):
        - 健康状态 (加分/观望): 价跌量缩 - 洗盘而非出货，未来买入机会
        - 警示状态 (减分): 价跌量增 - 恐慌盘或主力出货，趋势可能反转
        """
        try:
            # 如果consolidation_vp_score列不存在，初始化
            if 'consolidation_vp_score' not in df.columns:
                df['consolidation_vp_score'] = 0.0
            
            # 回调阶段和盘整阶段都包含在内，提前过滤
            consolidation_mask = df['market_phase'].isin(['retracement', 'consolidation'])
            if not consolidation_mask.any():
                return df  # 没有盘整/回调阶段数据，直接返回
            
            # 向量化计算所有条件
            price_down = (df['price_change'] < 0)
            price_down_significant = (df['price_change'] < -0.01)
            volume_shrink = (df['volume_ratio_20d'] < 0.8)
            volume_expand = (df['volume_ratio_20d'] > 1.2)
            volume_above_avg = (df['volume_ratio_20d'] > 1.0)
            above_support = (df['price_position'] > 0.3)
            price_sideways = (abs(df['price_change']) < 0.005)
            mid_position = (df['price_position'] > 0.4) & (df['price_position'] < 0.6)
            consolidation_only = (df['market_phase'] == 'consolidation')
            
            # 1. 健康状态：价跌量缩（洗盘）- 加分/观望（未来买入机会）
            healthy_retracement = consolidation_mask & price_down & volume_shrink & above_support
            df.loc[healthy_retracement, 'consolidation_vp_score'] = 0.3
            
            # 2. 警示状态：价跌量增（恐慌盘或主力出货）- 减分
            warning_sell_off = consolidation_mask & price_down_significant & volume_expand
            df.loc[warning_sell_off, 'consolidation_vp_score'] = -0.4
            
            # 3. 盘整阶段的健康积累形态
            healthy_accumulation = consolidation_only & price_sideways & volume_above_avg & mid_position
            df.loc[healthy_accumulation, 'consolidation_vp_score'] = 0.2
            
            return df
            
        except Exception as e:
            logger.error(f"分析回调/盘整阶段量价关系时发生错误: {e}")
            return df
    
    def _analyze_volume_price_by_stage(self, df):
        """
        基于阶段的条件量价分析（优化版本）
        
        根据每行数据的市场阶段，有条件地调用对应的分析函数，
        避免为每个数据点调用所有三个分析函数，提升性能。
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含市场阶段标识的数据框
            
        Returns
        -------
        pandas.DataFrame
            添加了基于阶段的量价分析结果的数据框
        """
        try:
            # 初始化所有评分列
            df['uptrend_vp_score'] = 0.0
            df['breakout_vp_score'] = 0.0
            df['consolidation_vp_score'] = 0.0
            
            # 获取各阶段的数据掩码
            uptrend_mask = (df['market_phase'] == 'uptrend')
            breakout_mask = (df['market_phase'] == 'breakout')
            consolidation_mask = df['market_phase'].isin(['retracement', 'consolidation'])
            
            # 统计各阶段数据量
            uptrend_count = uptrend_mask.sum()
            breakout_count = breakout_mask.sum()
            consolidation_count = consolidation_mask.sum()
            
            logger.info(f"阶段化量价分析 - 上涨阶段: {uptrend_count}行, "
                       f"突破阶段: {breakout_count}行, "
                       f"盘整/回调阶段: {consolidation_count}行")
            
            # 只对有数据的阶段进行分析，避免不必要的函数调用
            if uptrend_count > 0:
                logger.debug(f"分析上涨阶段量价关系 ({uptrend_count}行)")
                df = self._analyze_uptrend_volume_price(df)
            
            if breakout_count > 0:
                logger.debug(f"分析突破阶段量价关系 ({breakout_count}行)")
                df = self._analyze_breakout_volume_price(df)
            
            if consolidation_count > 0:
                logger.debug(f"分析盘整/回调阶段量价关系 ({consolidation_count}行)")
                df = self._analyze_consolidation_volume_price(df)
            
            
            return df
            
        except Exception as e:
            logger.error(f"基于阶段的量价分析时发生错误: {e}")
            # 降级到原始方法
            logger.warning("降级到全量分析模式")
            df = self._analyze_uptrend_volume_price(df)
            df = self._analyze_breakout_volume_price(df)
            df = self._analyze_consolidation_volume_price(df)
            return df
    
    def _generate_final_volume_price_signal(self, df):
        """生成最终量价信号
        
        综合四个阶段的量价分析结果，生成简洁明确的量价确认信号
        """
        try:
            # 初始化信号
            df['volume_price_signal'] = 0.0
            
            # 确保各阶段评分列存在
            if 'uptrend_vp_score' not in df.columns:
                df['uptrend_vp_score'] = 0.0
            if 'breakout_vp_score' not in df.columns:
                df['breakout_vp_score'] = 0.0
            if 'consolidation_vp_score' not in df.columns:
                df['consolidation_vp_score'] = 0.0
            
            # 基于市场阶段的加权综合
            phase_weights = {
                'breakout': 0.5,        # 突破阶段权重最高
                'uptrend': 0.3,         # 上涨阶段次之
                'consolidation': 0.15,  # 盘整阶段观望
                'retracement': 0.15     # 回调阶段观望
            }
            
            # 计算综合评分
            for phase, weight in phase_weights.items():
                phase_mask = (df['market_phase'] == phase)
                
                if phase == 'breakout':
                    df.loc[phase_mask, 'volume_price_signal'] += df.loc[phase_mask, 'breakout_vp_score'] * weight
                elif phase == 'uptrend':
                    df.loc[phase_mask, 'volume_price_signal'] += df.loc[phase_mask, 'uptrend_vp_score'] * weight
                else:  # consolidation, retracement
                    df.loc[phase_mask, 'volume_price_signal'] += df.loc[phase_mask, 'consolidation_vp_score'] * weight
            
            # 信号强度分级
            df['vp_signal_strength'] = np.select([
                abs(df['volume_price_signal']) >= 0.3,  # 强信号
                abs(df['volume_price_signal']) >= 0.15, # 中等信号
                abs(df['volume_price_signal']) >= 0.05  # 弱信号
            ], ['强', '中', '弱'], default='无')
            
            # 信号方向描述
            df['vp_signal_direction'] = np.select([
                df['volume_price_signal'] > 0.15,   # 明确买入
                df['volume_price_signal'] > 0.05,   # 偏向买入
                df['volume_price_signal'] < -0.15,  # 明确卖出
                df['volume_price_signal'] < -0.05,  # 偏向卖出
            ], ['买入', '偏多', '卖出', '偏空'], default='中性')
            
            return df
            
        except Exception as e:
            logger.error(f"生成最终量价信号时发生错误: {e}")
            return df
    
    # ===== 价格形态识别系统 =====
    
    def detect_ascending_channel(self, df):
        """
        检测上升通道（优先级3）
        
        计算滚动N日最高价/最低价移动平均线作为上下轨道
        验证正斜率（线性回归系数 > 0.001）
        确认当前价格位于轨道之间
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含OHLCV数据的数据框
            
        Returns
        -------
        tuple
            (channel_status, in_channel, channel_width, slope_upper, slope_lower)
            channel_status: 1(上升), 0(中性), -1(下降)
            in_channel: 价格是否在通道内
            channel_width: 通道宽度百分比
            slope_upper: 上轨斜率
            slope_lower: 下轨斜率
        """
        try:
            if len(df) < self.channel_period:
                logger.warning(f"数据长度不足，需要至少{self.channel_period}个周期")
                return 0, False, 0, 0, 0
            
            # 计算滚动最高价和最低价移动平均线
            upper_rail = df['high'].rolling(window=self.channel_period, min_periods=1).max()
            lower_rail = df['low'].rolling(window=self.channel_period, min_periods=1).min()
            
            # 使用线性回归计算斜率
            def calculate_slope(series):
                if len(series) < 2:
                    return 0
                x = np.arange(len(series))
                y = series.values
                # 过滤NaN值
                mask = ~np.isnan(y)
                if np.sum(mask) < 2:
                    return 0
                slope = np.polyfit(x[mask], y[mask], 1)[0]
                return slope
            
            # 计算上轨和下轨的斜率
            slope_upper = calculate_slope(upper_rail.tail(self.channel_period))
            slope_lower = calculate_slope(lower_rail.tail(self.channel_period))
            
            # 获取当前价格
            current_price = df['close'].iloc[-1]
            current_upper = upper_rail.iloc[-1]
            current_lower = lower_rail.iloc[-1]
            
            # 验证价格是否在通道内
            in_channel = current_lower <= current_price <= current_upper
            
            # 计算通道宽度百分比
            channel_width = (current_upper - current_lower) / current_price
            
            # 判断通道状态
            channel_status = 0
            if (slope_upper > self.channel_slope_threshold and
                slope_lower > self.channel_slope_threshold and
                channel_width > self.channel_width_threshold):
                channel_status = 1  # 上升通道
            elif (slope_upper < -self.channel_slope_threshold and
                  slope_lower < -self.channel_slope_threshold and
                  channel_width > self.channel_width_threshold):
                channel_status = -1  # 下降通道
            
            return channel_status, in_channel, channel_width, slope_upper, slope_lower
            
        except Exception as e:
            logger.error(f"检测上升通道时发生错误: {e}")
            return 0, False, 0, 0, 0
    
    def detect_breakout_pattern(self, df):
        """
        检测突破形态（优先级2）
        
        识别阻力位并检测首次突破
        验证成交量放大和突破确认
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含OHLCV数据的数据框
            
        Returns
        -------
        tuple
            (breakout_signal, breakout_strength, resistance_level, volume_confirmed)
            breakout_signal: 突破信号布尔值
            breakout_strength: 突破强度评分(0-1)
            resistance_level: 阻力位价格
            volume_confirmed: 成交量确认布尔值
        """
        try:
            if len(df) < self.breakout_resistance_period:
                logger.warning(f"数据长度不足，需要至少{self.breakout_resistance_period}个周期")
                return False, 0, 0, False
            
            # 计算阻力位（滚动最高价）
            resistance_level = df['high'].rolling(window=self.breakout_resistance_period, min_periods=1).max().iloc[-1]
            
            # 获取当前和前一日收盘价
            current_close = df['close'].iloc[-1]
            previous_close = df['close'].iloc[-2] if len(df) >= 2 else current_close
            
            # 检测首次突破
            first_time_breakout = (current_close > resistance_level and
                                 previous_close <= resistance_level)
            
            # 计算成交量均线
            if 'volume' in df.columns:
                volume_sma = df['volume'].rolling(window=self.volume_sma_period, min_periods=1).mean()
                current_volume = df['volume'].iloc[-1]
                volume_threshold = volume_sma.iloc[-1] * self.breakout_volume_multiplier
                volume_confirmed = current_volume > volume_threshold
            else:
                volume_confirmed = False
                logger.warning("缺少成交量数据，无法进行成交量确认")
            
            # 突破确认（需要连续N天收盘价高于阻力位）
            if len(df) >= self.breakout_confirmation_days:
                recent_closes = df['close'].tail(self.breakout_confirmation_days)
                breakout_sustained = all(close > resistance_level for close in recent_closes)
            else:
                breakout_sustained = current_close > resistance_level
            
            # 计算突破强度
            breakout_strength = 0
            if first_time_breakout:
                # 基于价格突破幅度和成交量放大程度计算强度
                price_strength = min((current_close - resistance_level) / resistance_level * 10, 0.5)
                volume_strength = 0.3 if volume_confirmed else 0
                confirmation_strength = 0.2 if breakout_sustained else 0
                breakout_strength = price_strength + volume_strength + confirmation_strength
            
            breakout_signal = first_time_breakout and volume_confirmed and breakout_sustained
            
            return breakout_signal, breakout_strength, resistance_level, volume_confirmed
            
        except Exception as e:
            logger.error(f"检测突破形态时发生错误: {e}")
            return False, 0, 0, False
    
    def detect_head_risk_signals(self, df):
        """
        检测顶部风险信号（优先级1）
        
        通过MACD和RSI背离以及成交量确认检测顶部风险
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含OHLCV和技术指标数据的数据框
            
        Returns
        -------
        tuple
            (head_risk_signal, risk_intensity, macd_divergence, rsi_divergence, volume_decline)
            head_risk_signal: 顶部风险信号布尔值
            risk_intensity: 风险强度评分(0-1)
            macd_divergence: MACD背离布尔值
            rsi_divergence: RSI背离布尔值
            volume_decline: 成交量下降布尔值
        """
        try:
            if len(df) < self.head_risk_peak_period:
                logger.warning(f"数据长度不足，需要至少{self.head_risk_peak_period}个周期")
                return False, 0, False, False, False
            
            # 计算MACD（如果不存在）
            if 'macd_dif' not in df.columns:
                df = self.macd_signal(df)
            
            # 计算RSI（如果不存在）
            if 'rsi' not in df.columns:
                df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
            
            # 寻找价格高点
            recent_data = df.tail(self.head_risk_peak_period)
            price_peaks = []
            macd_dif_at_peaks = []
            rsi_at_peaks = []
            
            # 使用滚动窗口找到局部高点
            for i in range(2, len(recent_data) - 2):
                if (recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and
                    recent_data['high'].iloc[i] > recent_data['high'].iloc[i-2] and
                    recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1] and
                    recent_data['high'].iloc[i] > recent_data['high'].iloc[i+2]):
                    
                    price_peaks.append(recent_data['high'].iloc[i])
                    macd_dif_at_peaks.append(recent_data['macd_dif'].iloc[i])
                    rsi_at_peaks.append(recent_data['rsi'].iloc[i])
            
            # 检测MACD背离
            macd_divergence = False
            if len(price_peaks) >= 2:
                # 价格创新高但MACD DIF未创新高
                latest_price_peak = price_peaks[-1]
                previous_price_peak = price_peaks[-2]
                latest_macd_dif = macd_dif_at_peaks[-1]
                previous_macd_dif = macd_dif_at_peaks[-2]
                
                if (latest_price_peak > previous_price_peak and
                    latest_macd_dif < previous_macd_dif):
                    macd_divergence = True
            
            # 检测RSI背离
            rsi_divergence = False
            if len(price_peaks) >= 2:
                latest_rsi = rsi_at_peaks[-1]
                previous_rsi = rsi_at_peaks[-2]
                
                if (price_peaks[-1] > price_peaks[-2] and
                    latest_rsi < previous_rsi):
                    rsi_divergence = True
            
            # 检测成交量下降
            volume_decline = False
            if 'volume' in df.columns and len(df) >= 10:
                recent_volume = df['volume'].tail(5).mean()
                earlier_volume = df['volume'].tail(10).head(5).mean()
                volume_decline = recent_volume < earlier_volume * 0.8
            
            # 计算风险强度
            risk_intensity = 0
            if macd_divergence:
                risk_intensity += 0.4
            if rsi_divergence:
                risk_intensity += 0.3
            if volume_decline:
                risk_intensity += 0.3
            
            # 顶部风险信号触发条件
            head_risk_signal = macd_divergence and (rsi_divergence or volume_decline)
            
            return head_risk_signal, risk_intensity, macd_divergence, rsi_divergence, volume_decline
            
        except Exception as e:
            logger.error(f"检测顶部风险信号时发生错误: {e}")
            return False, 0, False, False, False
    
    def calculate_pattern_score(self, head_risk, breakout_signal, channel_status, in_channel):
        """
        计算形态评分（严格分层逻辑）
        
        Parameters
        ----------
        head_risk : bool
            顶部风险信号
        breakout_signal : bool
            突破信号
        channel_status : int
            通道状态：1(上升), 0(中性), -1(下降)
        in_channel : bool
            价格是否在通道内
            
        Returns
        -------
        dict
            包含状态、评分、优先级的字典
        """
        try:
            # 优先级1：顶部风险（否决权）
            if head_risk:
                return {
                    "status": "顶部风险",
                    "score": -2,
                    "priority": 1,
                    "description": "检测到顶部风险信号，建议减仓或观望"
                }
            
            # 优先级2：突破信号
            if breakout_signal:
                return {
                    "status": "放量突破",
                    "score": +2,
                    "priority": 2,
                    "description": "检测到放量突破信号，建议买入"
                }
            
            # 优先级3：上升通道
            if channel_status == 1 and in_channel:
                return {
                    "status": "上升通道",
                    "score": +1,
                    "priority": 3,
                    "description": "价格处于上升通道内，趋势向好"
                }
            
            # 优先级4：形态不明
            return {
                "status": "形态不明",
                "score": 0,
                "priority": 4,
                "description": "无明确形态信号，建议观望"
            }
            
        except Exception as e:
            logger.error(f"计算形态评分时发生错误: {e}")
            return {
                "status": "计算错误",
                "score": 0,
                "priority": 5,
                "description": f"计算过程中出现错误: {e}"
            }
    
    def analyze_price_patterns(self, df):
        """
        主要形态分析入口函数
        
        执行完整的价格形态识别流程
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含OHLCV数据和技术指标的数据框
            
        Returns
        -------
        dict
            包含所有形态分析结果的字典
        """
        try:
            # 数据验证
            if df.empty:
                logger.warning("输入数据为空")
                return self._empty_pattern_result()
            
            if len(df) < 60:
                logger.warning("数据长度不足60个周期，形态识别可能不准确")
            
            # 必要列检查
            required_cols = ['date', 'open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"缺少必要的数据列: {missing_cols}")
                return self._empty_pattern_result()
            
            logger.info("开始执行价格形态识别分析")
            
            # 1. 检测上升通道
            channel_status, in_channel, channel_width, slope_upper, slope_lower = self.detect_ascending_channel(df)
            
            # 2. 检测突破形态
            breakout_signal, breakout_strength, resistance_level, volume_confirmed = self.detect_breakout_pattern(df)
            
            # 3. 检测顶部风险
            head_risk_signal, risk_intensity, macd_divergence, rsi_divergence, volume_decline = self.detect_head_risk_signals(df)
            
            # 4. 计算形态评分
            pattern_score = self.calculate_pattern_score(head_risk_signal, breakout_signal, channel_status, in_channel)
            
            # 5. 组织结果
            result = {
                # 形态评分
                "pattern_score": pattern_score["score"],
                "pattern_status": pattern_score["status"],
                "pattern_priority": pattern_score["priority"],
                "pattern_description": pattern_score["description"],
                
                # 上升通道详情
                "channel_status": channel_status,
                "in_channel": in_channel,
                "channel_width": channel_width,
                "slope_upper": slope_upper,
                "slope_lower": slope_lower,
                
                # 突破形态详情
                "breakout_signal": breakout_signal,
                "breakout_strength": breakout_strength,
                "resistance_level": resistance_level,
                "volume_confirmed": volume_confirmed,
                
                # 顶部风险详情
                "head_risk_signal": head_risk_signal,
                "risk_intensity": risk_intensity,
                "macd_divergence": macd_divergence,
                "rsi_divergence": rsi_divergence,
                "volume_decline": volume_decline,
                
                # 数据质量信息
                "data_periods": len(df),
                "analysis_timestamp": datetime.now().isoformat(),
                "confidence_level": self._calculate_confidence_level(df, pattern_score["score"])
            }
            
            # 记录分析结果
            logger.info(f"形态分析完成 - 状态: {pattern_score['status']}, "
                       f"评分: {pattern_score['score']}, "
                       f"优先级: {pattern_score['priority']}")
            
            return result
            
        except Exception as e:
            logger.error(f"价格形态分析过程中发生错误: {e}")
            return self._empty_pattern_result()
    
    def _empty_pattern_result(self):
        """返回空的形态分析结果"""
        return {
            "pattern_score": 0,
            "pattern_status": "数据不足",
            "pattern_priority": 5,
            "pattern_description": "数据不足以进行形态分析",
            "channel_status": 0,
            "in_channel": False,
            "channel_width": 0,
            "slope_upper": 0,
            "slope_lower": 0,
            "breakout_signal": False,
            "breakout_strength": 0,
            "resistance_level": 0,
            "volume_confirmed": False,
            "head_risk_signal": False,
            "risk_intensity": 0,
            "macd_divergence": False,
            "rsi_divergence": False,
            "volume_decline": False,
            "data_periods": 0,
            "analysis_timestamp": datetime.now().isoformat(),
            "confidence_level": 0
        }
    
    def _calculate_confidence_level(self, df, pattern_score):
        """计算置信度水平"""
        try:
            confidence = 0.5  # 基础置信度
            
            # 数据量加成
            if len(df) >= 250:  # 一年数据
                confidence += 0.3
            elif len(df) >= 60:  # 三个月数据
                confidence += 0.2
            elif len(df) >= 30:  # 一个月数据
                confidence += 0.1
            
            # 信号强度加成
            if abs(pattern_score) >= 2:
                confidence += 0.2
            elif abs(pattern_score) >= 1:
                confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"计算置信度时发生错误: {e}")
            return 0.5
    
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
            添加了组合信号和仓位的数据框
        """
        if df.empty:
            logger.warning("输入的数据为空")
            return df
        
        # 首先调用各个信号函数生成必要的信号列
        logger.info("开始生成各个策略信号")
        
        # 生成布林带突破信号
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns and 'bb_middle' in df.columns:
            df = self.bollinger_bands_breakout(df)
        
        
        # 生成MACD信号
        df = self.macd_signal(df)
        
        # 生成移动平均线分析信号
        df = self.moving_average_analysis(df,
                                        ma_periods=self.ma_periods,
                                        support_threshold=self.ma_support_threshold,
                                        trend_strength_sensitivity=self.trend_strength_sensitivity)
        
        # 生成量价齐升确认信号
        if 'volume' in df.columns and 'volume_ma_5' in df.columns:
            df = self.volume_price_confirmation(df)
        
        # 执行价格形态识别分析
        logger.info("开始执行价格形态识别分析")
        pattern_analysis = self.analyze_price_patterns(df)
        
        # 将形态分析结果添加到数据框中
        result_df = df.copy()
        result_df['pattern_score'] = pattern_analysis['pattern_score']
        result_df['pattern_status'] = pattern_analysis['pattern_status']
        result_df['pattern_priority'] = pattern_analysis['pattern_priority']
        result_df['head_risk_signal'] = pattern_analysis['head_risk_signal']
        result_df['breakout_signal'] = pattern_analysis['breakout_signal']
        result_df['channel_status'] = pattern_analysis['channel_status']
        result_df['confidence_level'] = pattern_analysis['confidence_level']
        
        # 确保必要的信号列存在
        signal_cols = [
            'bb_signal', 'ema_signal', 'macd_signal', 'ma_signal', 'multi_timeframe_signal',
            'ema_reversal_signal', 'volume_price_signal'
        ]
        
        available_signals = [col for col in signal_cols if col in df.columns]
        
        if not available_signals:
            logger.error("没有可用的策略信号列")
            return df
        
        logger.info(f"开始组合策略信号，可用信号: {available_signals}")
        
        # 将形态分析结果添加到数据框中
        result_df['pattern_score'] = pattern_analysis['pattern_score']
        result_df['pattern_status'] = pattern_analysis['pattern_status']
        result_df['pattern_priority'] = pattern_analysis['pattern_priority']
        result_df['head_risk_signal'] = pattern_analysis['head_risk_signal']
        result_df['breakout_signal'] = pattern_analysis['breakout_signal']
        result_df['channel_status'] = pattern_analysis['channel_status']
        result_df['confidence_level'] = pattern_analysis['confidence_level']
        
        # 按照策略类型分组
        # trend_signals = ['ema_signal', 'bb_signal', 'macd_signal', 'ma_signal']
        trend_signals = ['ema_signal', 'macd_signal']
        # trend_signals = ['bb_signal']

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

        # --- 价格形态识别分层优先级系统 ---
        # 初始化最终仓位列
        result_df['final_position'] = 0.0
        
        # 优先级1：顶部风险信号（否决权 - 最高优先级）
        if pattern_analysis['head_risk_signal']:
            logger.warning(f"检测到顶部风险信号，优先级1启动 - {pattern_analysis['pattern_description']}")
            result_df['final_position'] = -0.5  # 减仓信号
            result_df['pattern_action'] = "顶部风险-减仓"
        
        # 优先级2：突破形态识别（仅在无顶部风险时生效）
        elif pattern_analysis['breakout_signal']:
            logger.info(f"检测到突破形态信号，优先级2启动 - {pattern_analysis['pattern_description']}")
            result_df['final_position'] = 1.0  # 买入信号
            result_df['pattern_action'] = "突破形态-买入"
        
        # 优先级3：上升通道检测（仅在无更高优先级信号时生效）
        elif pattern_analysis['channel_status'] == 1:
            logger.info(f"检测到上升通道信号，优先级3启动 - {pattern_analysis['pattern_description']}")
            # 结合传统信号决定仓位
            if result_df['trend_score'].iloc[-1] >= 1.0:
                result_df['final_position'] = 0.8  # 通道内买入
            elif result_df['trend_score'].iloc[-1] <= -1.0:
                result_df['final_position'] = -0.3  # 通道内轻仓卖出
            else:
                result_df['final_position'] = 0.3  # 通道内持仓
            result_df['pattern_action'] = "上升通道-持仓"
        
        # 优先级4：传统信号组合（仅在无形态信号时生效）
        else:
            logger.info("无明确形态信号，使用传统信号组合")
            # 核心趋势策略
            main_buy_condition = (result_df['trend_score'] >= 1.0)
            main_sell_condition = (result_df['trend_score'] <= -1.0)

            result_df.loc[main_buy_condition, 'final_position'] = 1.0
            result_df.loc[main_sell_condition, 'final_position'] = -1.0
            result_df['pattern_action'] = "传统信号"

        # # 规则 2：反转策略（仅在未触发主要趋势信号时）
        # reversal_buy_condition = (result_df['final_position'] == 0.0) & \
        #                          (result_df['trend_score'] < -1.5) & \
        #                          (result_df['reversal_score'] > 1.0)

        # reversal_sell_condition = (result_df['final_position'] == 0.0) & \
        #                           (result_df['trend_score'] > 1.5) & \
        #                           (result_df['reversal_score'] < -1.0)

        # result_df.loc[reversal_buy_condition, 'final_position'] = 0.3
        # result_df.loc[reversal_sell_condition, 'final_position'] = -0.3

        # # 规则 3：成交量过滤器（否决或减弱）
        # no_volume_condition = (result_df['final_position'] != 0.0) & (~result_df['volume_confirmed'])
        # result_df.loc[no_volume_condition, 'final_position'] *= 0.5

        # --- 将 final_position 映射为信号 ---
        result_df['final_signal'] = 0
        result_df.loc[result_df['final_position'] == 1.0, 'final_signal'] = 1
        result_df.loc[result_df['final_position'] == -1.0, 'final_signal'] = -1
        result_df.loc[(result_df['final_position'] > 0) & (result_df['final_position'] < 1.0), 'final_signal'] = 0.5
        result_df.loc[(result_df['final_position'] < 0) & (result_df['final_position'] > -1.0), 'final_signal'] = -0.5

        # 应用收盘价交易执行
        # result_df = self.close_price_execution(result_df, 'final_signal')

        # 使用最终仓位作为 position 列
        # result_df['final_signal_position'] = result_df['final_position']

        # 统计信号数量
        buy_signals = (result_df['final_signal'] > 0).sum()
        sell_signals = (result_df['final_signal'] < 0).sum()
        
        # 统计形态识别结果
        pattern_actions = result_df['pattern_action'].iloc[-1] if 'pattern_action' in result_df.columns else "无"
        current_pattern_score = result_df['pattern_score'].iloc[-1] if 'pattern_score' in result_df.columns else 0
        current_confidence = result_df['confidence_level'].iloc[-1] if 'confidence_level' in result_df.columns else 0
        
        logger.info(f"=== 价格形态识别结果 ===")
        logger.info(f"形态状态: {pattern_analysis['pattern_status']}")
        logger.info(f"形态评分: {pattern_analysis['pattern_score']}")
        logger.info(f"优先级: {pattern_analysis['pattern_priority']}")
        logger.info(f"置信度: {pattern_analysis['confidence_level']:.2f}")
        logger.info(f"执行动作: {pattern_actions}")
        logger.info(f"=== 信号组合结果 ===")
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
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # 创建策略实例
    strategy = TrendStrategy()
    
    # 测试策略
    print("趋势策略模块测试完成")
