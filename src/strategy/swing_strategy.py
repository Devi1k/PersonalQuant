#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
波段策略模块
实现多周期动量确认、通道回归均值、形态驱动反转、对冲型仓位平衡和量价微观验证等波段交易策略
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path

# 设置日志
logger = logging.getLogger(__name__)

class SwingStrategy:
    """波段策略类"""
    
    def __init__(self, config=None):
        """
        初始化波段策略
        
        Parameters
        ----------
        config : dict, default None
            策略配置参数
        """
        self.config = config or {}
        
        # 从配置中获取策略参数，如果没有则使用默认值
        swing_config = self.config.get('strategy', {}).get('swing', {})
        
        # 多周期动量确认系统参数
        self.rsi_period = swing_config.get('rsi_period', 14)
        self.rsi_oversold = swing_config.get('rsi_oversold', 30)
        self.kdj_j_oversold = swing_config.get('kdj_j_oversold', 20)
        self.macd_bars_increasing = swing_config.get('macd_bars_increasing', 3)
        self.double_bottom_volume_ratio = swing_config.get('double_bottom_volume_ratio', 1.3)
        
        # 通道回归均值策略参数
        self.bollinger_period = swing_config.get('bollinger_period', 20)
        self.bollinger_std_dev = swing_config.get('bollinger_std_dev', 2.0)
        self.bollinger_reversion_bars = swing_config.get('bollinger_reversion_bars', 3)
        self.kdj_params = swing_config.get('kdj_params', (9, 3, 3))  # (K周期, D周期, J周期)
        self.atr_period = swing_config.get('atr_period', 14)
        self.atr_stop_loss_multiplier = swing_config.get('atr_stop_loss_multiplier', 1.5)
        self.max_breakout_minutes = swing_config.get('max_breakout_minutes', 20)
        
        # 形态驱动反转交易参数
        self.head_shoulder_volume_ratio = swing_config.get('head_shoulder_volume_ratio', 0.5)
        self.head_shoulder_breakout_volume_ratio = swing_config.get('head_shoulder_breakout_volume_ratio', 2.0)
        self.triangle_support_volume_ratio = swing_config.get('triangle_support_volume_ratio', 0.7)
        self.flag_pole_min_rise = swing_config.get('flag_pole_min_rise', 0.15)
        self.flag_max_retracement = swing_config.get('flag_max_retracement', 0.382)
        
        # 对冲型仓位平衡参数
        self.ema_channel_period = swing_config.get('ema_channel_period', 144)
        self.td_sequence_count = swing_config.get('td_sequence_count', 13)
        self.hedge_option_delta = swing_config.get('hedge_option_delta', (0.3, 0.4))  # (最小Delta, 最大Delta)
        self.hedge_position_ratio = swing_config.get('hedge_position_ratio', (0.15, 0.25))  # (最小比例, 最大比例)
        
        # 量价微观验证参数
        self.vwap_volume_ratio = swing_config.get('vwap_volume_ratio', 1.2)
        self.large_order_inflow_ratio = swing_config.get('large_order_inflow_ratio', 0.4)
        self.closing_auction_time = swing_config.get('closing_auction_time', "14:55")
        
        # 信号组合权重
        default_weights = {
            'momentum_signal': 2.0,        # 多周期动量确认信号
            'channel_signal': 2.0,         # 通道回归均值信号
            'pattern_signal': 1.5,         # 形态驱动反转信号
            'hedge_signal': 1.0,           # 对冲型仓位平衡信号
            'volume_price_signal': 1.5     # 量价微观验证信号
        }
        self.signal_weights = swing_config.get('signal_weights', default_weights)
        
        logger.info(f"波段策略初始化完成，参数：RSI周期={self.rsi_period}, RSI超卖阈值={self.rsi_oversold}, "
                   f"KDJ-J超卖阈值={self.kdj_j_oversold}, 布林带周期={self.bollinger_period}, "
                   f"布林带标准差={self.bollinger_std_dev}, ATR周期={self.atr_period}, "
                   f"ATR止损倍数={self.atr_stop_loss_multiplier}")
    
    def multi_timeframe_momentum_system(self, df_5min, df_15min, df_60min):
        """
        多周期动量确认系统
        
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
            添加了多周期动量确认信号的数据框（基于5分钟K线）
        """
        if df_5min.empty or df_15min.empty or df_60min.empty:
            logger.warning("输入的数据为空")
            return df_5min
        
        # 确保必要的列存在
        required_cols = ["date", "close", "open", "high", "low", "volume", "rsi_14", "kdj_j", "bb_lower", "bb_middle"]
        for df, timeframe in [(df_5min, "5分钟"), (df_15min, "15分钟"), (df_60min, "60分钟")]:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"{timeframe}K线数据缺失必要的列: {missing_cols}")
                return df_5min
        
        logger.info("开始计算多周期动量确认系统信号")
        
        # 复制数据，避免修改原始数据
        result_df = df_5min.copy()
        
        # 1. 短期动量：5分钟K线RSI（14周期）<30且KDJ-J线<20（超卖）+ 布林带下轨价格回归至中轨上方
        result_df['short_term_oversold'] = (
            (result_df['rsi_14'] < self.rsi_oversold) & 
            (result_df['kdj_j'] < self.kdj_j_oversold)
        )
        
        # 布林带下轨价格回归至中轨上方
        result_df['bb_lower_to_middle_cross'] = (
            (result_df['close'] > result_df['bb_middle']) & 
            (result_df['close'].shift(1) <= result_df['bb_middle'].shift(1)) &
            (result_df['close'].shift(5) < result_df['bb_lower'].shift(5))
        )
        
        # 2. 中期过滤：15分钟K线需满足MACD柱线连续3根递增（防止短期假信号）
        # 将15分钟MACD柱状线数据合并到5分钟数据
        df_15min_copy = df_15min.copy()
        df_15min_copy['date_rounded'] = df_15min_copy['date'].apply(
            lambda x: x.replace(minute=(x.minute // 15) * 15, second=0, microsecond=0)
        )
        
        # 计算15分钟MACD柱状线是否连续递增
        df_15min_copy['macd_hist_increasing'] = False
        for i in range(self.macd_bars_increasing, len(df_15min_copy)):
            increasing = True
            for j in range(1, self.macd_bars_increasing):
                if df_15min_copy['macd_hist'].iloc[i-j] <= df_15min_copy['macd_hist'].iloc[i-j-1]:
                    increasing = False
                    break
            df_15min_copy.loc[df_15min_copy.index[i], 'macd_hist_increasing'] = increasing
        
        # 准备15分钟MACD数据用于合并
        df_15min_macd = df_15min_copy[['date_rounded', 'macd_hist_increasing']].rename(
            columns={'date_rounded': 'date_15min'}
        )
        
        # 创建用于合并的辅助列
        result_df['date_15min'] = result_df['date'].apply(
            lambda x: x.replace(minute=(x.minute // 15) * 15, second=0, microsecond=0)
        )
        
        # 合并15分钟MACD数据
        result_df = pd.merge(result_df, df_15min_macd, on='date_15min', how='left')
        
        # 填充可能的缺失值
        result_df['macd_hist_increasing'] = result_df['macd_hist_increasing'].fillna(False)
        
        # 3. 执行逻辑：观察60分钟K线是否形成双底形态（第二个底不创新低，且右底成交量>左底1.3倍）
        # 将60分钟K线数据合并到5分钟数据
        df_60min_copy = df_60min.copy()
        df_60min_copy['date_rounded'] = df_60min_copy['date'].apply(
            lambda x: x.replace(minute=0, second=0, microsecond=0)
        )
        
        # 识别60分钟K线的双底形态
        df_60min_copy['is_bottom'] = (
            (df_60min_copy['low'] < df_60min_copy['low'].shift(1)) & 
            (df_60min_copy['low'] < df_60min_copy['low'].shift(-1))
        )
        
        # 查找双底形态
        df_60min_copy['double_bottom'] = False
        for i in range(10, len(df_60min_copy)):
            # 查找过去10个周期内的两个底
            bottoms = df_60min_copy['is_bottom'].iloc[i-10:i+1]
            bottom_indices = bottoms[bottoms].index.tolist()
            
            if len(bottom_indices) >= 2:
                # 获取最近的两个底
                right_bottom_idx = bottom_indices[-1]
                left_bottom_idx = bottom_indices[-2]
                
                # 检查第二个底是否不创新低
                if df_60min_copy.loc[right_bottom_idx, 'low'] >= df_60min_copy.loc[left_bottom_idx, 'low']:
                    # 检查右底成交量是否大于左底成交量的1.3倍
                    if (df_60min_copy.loc[right_bottom_idx, 'volume'] > 
                        df_60min_copy.loc[left_bottom_idx, 'volume'] * self.double_bottom_volume_ratio):
                        df_60min_copy.loc[right_bottom_idx, 'double_bottom'] = True
        
        # 准备60分钟双底数据用于合并
        df_60min_bottom = df_60min_copy[['date_rounded', 'double_bottom']].rename(
            columns={'date_rounded': 'date_60min'}
        )
        
        # 创建用于合并的辅助列
        result_df['date_60min'] = result_df['date'].apply(
            lambda x: x.replace(minute=0, second=0, microsecond=0)
        )
        
        # 合并60分钟双底数据
        result_df = pd.merge(result_df, df_60min_bottom, on='date_60min', how='left')
        
        # 填充可能的缺失值
        result_df['double_bottom'] = result_df['double_bottom'].fillna(False)
        
        # 综合多周期动量确认信号
        # 1 = 买入信号（短期超卖 + 布林带回归 + 15分钟MACD柱线递增 + 60分钟双底）
        # 0 = 无信号
        
        result_df['momentum_signal'] = 0
        result_df.loc[
            result_df['short_term_oversold'] & 
            result_df['bb_lower_to_middle_cross'] & 
            result_df['macd_hist_increasing'] & 
            result_df['double_bottom'],
            'momentum_signal'
        ] = 1
        
        # 统计信号数量
        buy_signals = (result_df['momentum_signal'] == 1).sum()
        logger.info(f"多周期动量确认系统信号计算完成，买入信号: {buy_signals}个")
        
        # 删除辅助列
        result_df.drop(['date_15min', 'date_60min'], axis=1, inplace=True)
        
        return result_df
    
    def channel_mean_reversion_strategy(self, df):
        """
        通道回归均值策略
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含价格和技术指标的数据框
            
        Returns
        -------
        pandas.DataFrame
            添加了通道回归均值策略信号的数据框
        """
        if df.empty:
            logger.warning("输入的数据为空")
            return df
        
        # 确保必要的列存在
        required_cols = ["date", "close", "high", "low", "bb_upper", "bb_lower", "bb_middle", "kdj_k", "kdj_d", "atr"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"通道回归均值策略所需的列缺失: {missing_cols}")
            return df
        
        logger.info("开始计算通道回归均值策略信号")
        
        # 复制数据，避免修改原始数据
        result_df = df.copy()
        
        # 1. 布林带（20日，2倍标准差）上轨做空/下轨做多，但仅在价格触碰通道后3根K线内回归中轨时生效
        
        # 上轨触碰
        result_df['touch_upper'] = result_df['high'] >= result_df['bb_upper']
        
        # 下轨触碰
        result_df['touch_lower'] = result_df['low'] <= result_df['bb_lower']
        
        # 中轨向上穿越
        result_df['cross_middle_up'] = (
            (result_df['close'] > result_df['bb_middle']) & 
            (result_df['close'].shift(1) <= result_df['bb_middle'].shift(1))
        )
        
        # 中轨向下穿越
        result_df['cross_middle_down'] = (
            (result_df['close'] < result_df['bb_middle']) & 
            (result_df['close'].shift(1) >= result_df['bb_middle'].shift(1))
        )
        
        # 初始化信号列
        result_df['channel_signal'] = 0
        
        # 买入信号：之前3根K线内触碰下轨，现在向上穿过中轨
        for i in range(self.bollinger_reversion_bars, len(result_df)):
            if (result_df['touch_lower'].iloc[i-self.bollinger_reversion_bars:i].any() and 
                result_df['cross_middle_up'].iloc[i]):
                result_df.loc[result_df.index[i], 'channel_signal'] = 1
        
        # 卖出信号：之前3根K线内触碰上轨，现在向下穿过中轨
        for i in range(self.bollinger_reversion_bars, len(result_df)):
            if (result_df['touch_upper'].iloc[i-self.bollinger_reversion_bars:i].any() and 
                result_df['cross_middle_down'].iloc[i]):
                result_df.loc[result_df.index[i], 'channel_signal'] = -1
        
        # 2. 过滤条件：结合日线级别的KDJ（9,3,3）金叉/死叉提高胜率
        
        # 计算KDJ金叉/死叉
        result_df['kdj_golden_cross'] = (
            (result_df['kdj_k'] > result_df['kdj_d']) & 
            (result_df['kdj_k'].shift(1) <= result_df['kdj_d'].shift(1))
        )
        
        result_df['kdj_death_cross'] = (
            (result_df['kdj_k'] < result_df['kdj_d']) & 
            (result_df['kdj_k'].shift(1) >= result_df['kdj_d'].shift(1))
        )
        
        # 根据KDJ金叉/死叉过滤信号
        # 日线死叉时忽略布林带上轨做空信号
        result_df.loc[result_df['kdj_death_cross'] & (result_df['channel_signal'] == -1), 'channel_signal'] = 0
        
        # 日线金叉时忽略布林带下轨做多信号
        result_df.loc[result_df['kdj_golden_cross'] & (result_df['channel_signal'] == 1), 'channel_signal'] = 0
        
        # 3. 风险控制：单次止损为ATR（14日）的1.5倍，若价格突破通道后20分钟未回归则强制止损
        
        # 计算止损价格
        result_df['long_stop_loss'] = result_df['close'] - (result_df['atr'] * self.atr_stop_loss_multiplier)
        result_df['short_stop_loss'] = result_df['close'] + (result_df['atr'] * self.atr_stop_loss_multiplier)
        
        # 统计信号数量
        buy_signals = (result_df['channel_signal'] == 1).sum()
        sell_signals = (result_df['channel_signal'] == -1).sum()
        logger.info(f"通道回归均值策略信号计算完成，买入信号: {buy_signals}个, 卖出信号: {sell_signals}个")
        
        return result_df
    
    def pattern_driven_reversal_strategy(self, df):
        """
        形态驱动反转交易
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含价格和技术指标的数据框
            
        Returns
        -------
        pandas.DataFrame
            添加了形态驱动反转交易信号的数据框
        """
        if df.empty:
            logger.warning("输入的数据为空")
            return df
        
        # 确保必要的列存在
        required_cols = ["date", "close", "open", "high", "low", "volume", "volume_ma_5", "rsi_14"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"形态驱动反转交易策略所需的列缺失: {missing_cols}")
            return df
        
        logger.info("开始计算形态驱动反转交易信号")
        
        # 复制数据，避免修改原始数据
        result_df = df.copy()
        
        # 初始化形态识别列
        result_df['head_shoulder_bottom'] = False
        result_df['ascending_triangle'] = False
        result_df['flag_pattern'] = False
        
        # 1. 头肩底形态识别（右肩成交量>头部50%，颈线突破时成交量>5日均量2倍）
        for i in range(30, len(result_df)):
            # 简化的头肩底识别逻辑
            # 实际应用中应使用更复杂的算法或专业库
            
            # 查找可能的头肩底形态
            window = result_df.iloc[i-30:i+1]
            
            # 查找三个低点（左肩、头部、右肩）
            lows = window['low'].rolling(5, center=True).min()
            low_points = []
            
            for j in range(2, len(lows)-2):
                if (lows.iloc[j] == window['low'].iloc[j] and 
                    lows.iloc[j] < lows.iloc[j-2] and 
                    lows.iloc[j] < lows.iloc[j+2]):
                    low_points.append(j)
            
            # 如果找到至少三个低点
            if len(low_points) >= 3:
                left_shoulder_idx = low_points[0]
                head_idx = low_points[1]
                right_shoulder_idx = low_points[2]
                
                # 检查头部是否低于两肩
                if (window['low'].iloc[head_idx] < window['low'].iloc[left_shoulder_idx] and 
                    window['low'].iloc[head_idx] < window['low'].iloc[right_shoulder_idx]):
                    
                    # 检查右肩成交量是否大于头部的50%
                    if (window['volume'].iloc[right_shoulder_idx] > 
                        window['volume'].iloc[head_idx] * self.head_shoulder_volume_ratio):
                        
                        # 计算颈线（左肩和右肩之间的高点）
                        neck_line = max(window['high'].iloc[left_shoulder_idx:right_shoulder_idx+1])
                        
                        # 检查是否突破颈线
                        if window['close'].iloc[-1] > neck_line:
                            # 检查突破时成交量是否大于5日均量的2倍
                            if (window['volume'].iloc[-1] > 
                                window['volume_ma_5'].iloc[-1] * self.head_shoulder_breakout_volume_ratio):
                                result_df.loc[result_df.index[i], 'head_shoulder_bottom'] = True
        
        # 2. 上升三角形识别（第3次回踩支撑线缩量至均量70%，突破时放量）
        for i in range(20, len(result_df)):
            # 简化的上升三角形识别逻辑
            window = result_df.iloc[i-20:i+1]
            
            # 查找水平支撑线（至少三次触及）
            lows = window['low'].values
            support_line = None
            support_touches = 0
            
            for level in np.linspace(min(lows), max(lows), 20):
                touches = sum(abs(low - level) < (max(lows) - min(lows)) * 0.03 for low in lows)
                if touches >= 3 and (support_line is None or touches > support_touches):
                    support_line = level
                    support_touches = touches
            
            if support_line is not None:
                # 查找下降阻力线（至少两次触及）
                highs = window['high'].values
                resistance_touches = []
                
                for j in range(len(highs)):
                    if abs(highs[j] - max(highs[:j+1])) < (max(highs) - min(highs)) * 0.03:
                        resistance_touches.append(j)
                
                if len(resistance_touches) >= 2:
                    # 检查第3次回踩支撑线是否缩量
                    support_touches_idx = [j for j in range(len(lows)) if abs(lows[j] - support_line) < (max(lows) - min(lows)) * 0.03]
                    
                    if len(support_touches_idx) >= 3:
                        third_touch_idx = support_touches_idx[2]
                        
                        if (window['volume'].iloc[third_touch_idx] < 
                            window['volume_ma_5'].iloc[third_touch_idx] * self.triangle_support_volume_ratio):
                            
                            # 检查是否突破阻力线
                            if window['close'].iloc[-1] > max(highs[:-1]):
                                # 检查突破时是否放量
                                if window['volume'].iloc[-1] > window['volume_ma_5'].iloc[-1]:
                                    result_df.loc[result_df.index[i], 'ascending_triangle'] = True
        
        # 3. 旗形整理识别（旗杆涨幅>15%，回调幅度<38.2%斐波那契位）
        for i in range(15, len(result_df)):
            # 简化的旗形整理识别逻辑
            window = result_df.iloc[i-15:i+1]
            
            # 查找可能的旗杆（快速上涨）
            for j in range(5, 10):
                if j < len(window):
                    pole_start = window['low'].iloc[0]
                    pole_end = window['high'].iloc[j]
                    pole_rise = (pole_end - pole_start) / pole_start
                    
                    # 检查旗杆涨幅是否大于15%
                    if pole_rise > self.flag_pole_min_rise:
                        # 检查回调幅度是否小于38.2%斐波那契位
                        retracement = (pole_end - window['low'].iloc[j:].min()) / (pole_end - pole_start)
                        
                        if retracement < self.flag_max_retracement:
                            # 检查是否突破旗形上轨
                            if window['close'].iloc[-1] > window['high'].iloc[j:-1].max():
                                result_df.loc[result_df.index[i], 'flag_pattern'] = True
                                break
        
        # 4. 增强信号：形态完成后RSI出现背离（价格新低但RSI未新低，或新高但RSI未新高）
        
        # 计算RSI背离
        result_df['rsi_bullish_divergence'] = False
        result_df['rsi_bearish_divergence'] = False
        
        for i in range(10, len(result_df)):
            # 查找价格新低但RSI未新低（看涨背离）
            if (result_df['low'].iloc[i] < result_df['low'].iloc[i-5:i].min() and 
                result_df['rsi_14'].iloc[i] > result_df['rsi_14'].iloc[i-5:i].min()):
                result_df.loc[result_df.index[i], 'rsi_bullish_divergence'] = True
            
            # 查找价格新高但RSI未新高（看跌背离）
            if (result_df['high'].iloc[i] > result_df['high'].iloc[i-5:i].max() and 
                result_df['rsi_14'].iloc[i] < result_df['rsi_14'].iloc[i-5:i].max()):
                result_df.loc[result_df.index[i], 'rsi_bearish_divergence'] = True
        
        # 综合形态驱动反转信号
        # 1 = 买入信号（头肩底或上升三角形或旗形整理，且有RSI看涨背离）
        # -1 = 卖出信号（RSI看跌背离）
        # 0 = 无信号
        
        result_df['pattern_signal'] = 0
        
        # 买入信号
        result_df.loc[
            ((result_df['head_shoulder_bottom']) | 
             (result_df['ascending_triangle']) | 
             (result_df['flag_pattern'])) & 
            (result_df['rsi_bullish_divergence']),
            'pattern_signal'
        ] = 1
        
        # 卖出信号
        result_df.loc[result_df['rsi_bearish_divergence'], 'pattern_signal'] = -1
        
        # 统计信号数量
        buy_signals = (result_df['pattern_signal'] == 1).sum()
        sell_signals = (result_df['pattern_signal'] == -1).sum()
        logger.info(f"形态驱动反转交易信号计算完成，买入信号: {buy_signals}个, 卖出信号: {sell_signals}个")
        
        return result_df
    def hedge_position_balance(self, df, weekly_df=None):
        """
        对冲型仓位平衡
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含价格和技术指标的数据框
        weekly_df : pandas.DataFrame, default None
            周线数据框，用于TD序列计算
            
        Returns
        -------
        pandas.DataFrame
            添加了对冲型仓位平衡信号的数据框
        """
        if df.empty:
            logger.warning("输入的数据为空")
            return df
        
        # 确保必要的列存在
        required_cols = ["date", "close", "high", "low", "ema_144"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"对冲型仓位平衡策略所需的列缺失: {missing_cols}")
            return df
        
        logger.info("开始计算对冲型仓位平衡信号")
        
        # 复制数据，避免修改原始数据
        result_df = df.copy()
        
        # 1. 对冲触发：当144日EMA通道上轨被突破且同时出现周线级别TD序列13计数，启动对冲模块
        
        # 计算EMA通道上下轨
        result_df['ema_upper'] = result_df['ema_144'] * (1 + 0.05)  # 上轨为EMA的1.05倍
        result_df['ema_lower'] = result_df['ema_144'] * (1 - 0.05)  # 下轨为EMA的0.95倍
        
        # 检测EMA通道上轨突破
        result_df['ema_upper_breakout'] = result_df['close'] > result_df['ema_upper']
        
        # 计算周线TD序列（如果提供了周线数据）
        td_setup_count = 0
        if weekly_df is not None and not weekly_df.empty:
            # 确保周线数据包含必要的列
            if all(col in weekly_df.columns for col in ["date", "close"]):
                # 计算TD序列
                weekly_df['td_setup'] = 0
                
                for i in range(4, len(weekly_df)):
                    # TD Setup: 9个连续的收盘价高于4个周期前的收盘价
                    if weekly_df['close'].iloc[i] > weekly_df['close'].iloc[i-4]:
                        td_setup_count += 1
                        if td_setup_count <= 9:
                            weekly_df.loc[weekly_df.index[i], 'td_setup'] = td_setup_count
                    else:
                        td_setup_count = 0
                        weekly_df.loc[weekly_df.index[i], 'td_setup'] = 0
                
                # TD Countdown: 从TD Setup完成后开始计数，13个收盘价高于2个周期前的收盘价
                weekly_df['td_countdown'] = 0
                td_countdown_count = 0
                td_setup_completed = False
                
                for i in range(len(weekly_df)):
                    if weekly_df['td_setup'].iloc[i] == 9:
                        td_setup_completed = True
                        td_countdown_count = 0
                    
                    if td_setup_completed and i >= 2:
                        if weekly_df['close'].iloc[i] > weekly_df['close'].iloc[i-2]:
                            td_countdown_count += 1
                            if td_countdown_count <= 13:
                                weekly_df.loc[weekly_df.index[i], 'td_countdown'] = td_countdown_count
                        else:
                            # 不重置计数，只是不增加
                            weekly_df.loc[weekly_df.index[i], 'td_countdown'] = td_countdown_count
                
                # 将周线TD序列数据合并到日线数据
                # 创建日期映射函数，将日线日期映射到对应的周
                def map_to_week(date):
                    # 获取日期所在周的周一日期
                    week_start = date - pd.Timedelta(days=date.weekday())
                    return week_start
                
                # 在周线数据中添加周开始日期列
                weekly_df['week_start'] = weekly_df['date'].apply(map_to_week)
                
                # 在日线数据中添加周开始日期列
                result_df['week_start'] = result_df['date'].apply(map_to_week)
                
                # 准备周线TD序列数据用于合并
                weekly_td = weekly_df[['week_start', 'td_countdown']].copy()
                
                # 合并周线TD序列数据到日线数据
                result_df = pd.merge(result_df, weekly_td, on='week_start', how='left')
                
                # 填充可能的缺失值
                result_df['td_countdown'] = result_df['td_countdown'].fillna(0)
            else:
                logger.warning("周线数据缺少必要的列，无法计算TD序列")
                result_df['td_countdown'] = 0
        else:
            logger.warning("未提供周线数据，无法计算TD序列")
            result_df['td_countdown'] = 0
        
        # 2. 对冲信号：EMA通道上轨突破 + TD序列13计数
        result_df['hedge_signal'] = 0
        result_df.loc[
            (result_df['ema_upper_breakout']) &
            (result_df['td_countdown'] >= self.td_sequence_count),
            'hedge_signal'
        ] = -1  # 做空信号
        
        # 3. 对冲仓位计算
        # 基于波动率锥动态调整对冲仓位比例
        
        # 计算20日波动率
        result_df['returns'] = result_df['close'].pct_change()
        result_df['volatility_20d'] = result_df['returns'].rolling(20).std() * np.sqrt(252)
        
        # 计算波动率百分位（相对于过去一年）
        vol_percentile = result_df['volatility_20d'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0
        )
        
        # 根据波动率百分位动态调整对冲仓位比例
        min_ratio, max_ratio = self.hedge_position_ratio
    def volume_price_microvalidation(self, df, level2_data=None):
        """
        量价微观验证
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含价格和成交量数据的数据框
        level2_data : pandas.DataFrame, default None
            Level2数据，包含主力资金流向信息
            
        Returns
        -------
        pandas.DataFrame
            添加了量价微观验证信号的数据框
        """
        if df.empty:
            logger.warning("输入的数据为空")
            return df
        
        # 确保必要的列存在
        required_cols = ["date", "close", "volume", "vwap"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"量价微观验证策略所需的列缺失: {missing_cols}")
            return df
        
        logger.info("开始计算量价微观验证信号")
        
        # 复制数据，避免修改原始数据
        result_df = df.copy()
        
        # 1. 入场校验：价格突破时需满足成交量>当日VWAP的120%
        result_df['volume_gt_vwap'] = result_df['volume'] > (result_df['vwap'] * self.vwap_volume_ratio)
        
        # 2. 主力资金流向监控（如果提供了Level2数据）
        if level2_data is not None and not level2_data.empty:
            # 确保Level2数据包含必要的列
            if all(col in level2_data.columns for col in ["date", "large_order_inflow_ratio"]):
                # 合并Level2数据
                level2_data_copy = level2_data[['date', 'large_order_inflow_ratio']].copy()
                result_df = pd.merge(result_df, level2_data_copy, on='date', how='left')
                
                # 填充可能的缺失值
                result_df['large_order_inflow_ratio'] = result_df['large_order_inflow_ratio'].fillna(0)
            else:
                logger.warning("Level2数据缺少必要的列，无法监控主力资金流向")
                result_df['large_order_inflow_ratio'] = 0
        else:
            logger.warning("未提供Level2数据，无法监控主力资金流向")
            result_df['large_order_inflow_ratio'] = 0
        
        # 3. 量价微观验证信号
        # 1 = 买入信号（成交量>VWAP的120% + 主力资金净流入占比>40%）
        # 0 = 无信号
        
        result_df['volume_price_signal'] = 0
        result_df.loc[
            (result_df['volume_gt_vwap']) &
            (result_df['large_order_inflow_ratio'] > self.large_order_inflow_ratio),
            'volume_price_signal'
        ] = 1
        
        # 4. 出场优化：设定14:55的尾盘集中成交算法（TWAP策略降低滑点）
        # 这部分通常在实盘交易系统中实现，这里只做标记
        
        # 标记接近收盘时间的K线
        result_df['closing_auction'] = result_df['date'].apply(
            lambda x: x.strftime('%H:%M') == self.closing_auction_time
        )
        
        # 统计信号数量
        buy_signals = (result_df['volume_price_signal'] == 1).sum()
        logger.info(f"量价微观验证信号计算完成，买入信号: {buy_signals}个")
        
        return result_df
    
    def combine_signals(self, df):
        """
        组合各策略信号
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含各策略信号的数据框
            
        Returns
        -------
        pandas.DataFrame
            添加了综合信号的数据框
        """
        if df.empty:
            logger.warning("输入的数据为空")
            return df
        
        # 确保必要的列存在
        required_cols = ["date", "close", "momentum_signal", "channel_signal", "pattern_signal", "hedge_signal", "volume_price_signal"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"组合信号所需的列缺失: {missing_cols}")
            return df
        
        logger.info("开始计算综合信号")
        
        # 复制数据，避免修改原始数据
        result_df = df.copy()
        
        # 计算加权综合信号
        result_df['weighted_signal'] = (
            result_df['momentum_signal'] * self.signal_weights.get('momentum_signal', 2.0) +
            result_df['channel_signal'] * self.signal_weights.get('channel_signal', 2.0) +
            result_df['pattern_signal'] * self.signal_weights.get('pattern_signal', 1.5) +
            result_df['hedge_signal'] * self.signal_weights.get('hedge_signal', 1.0) +
            result_df['volume_price_signal'] * self.signal_weights.get('volume_price_signal', 1.5)
        )
        
        # 综合信号
        # 1 = 买入信号（加权信号 > 2）
        # -1 = 卖出信号（加权信号 < -2）
        # 0 = 无信号
        
        result_df['combined_signal'] = 0
        result_df.loc[result_df['weighted_signal'] > 2, 'combined_signal'] = 1
        result_df.loc[result_df['weighted_signal'] < -2, 'combined_signal'] = -1
        
        # 统计信号数量
        buy_signals = (result_df['combined_signal'] == 1).sum()
        sell_signals = (result_df['combined_signal'] == -1).sum()
        logger.info(f"综合信号计算完成，买入信号: {buy_signals}个, 卖出信号: {sell_signals}个")
        
        return result_df
        result_df['hedge_position_ratio'] = min_ratio + (max_ratio - min_ratio) * vol_percentile
        
        # 统计信号数量
        hedge_signals = (result_df['hedge_signal'] == -1).sum()
        logger.info(f"对冲型仓位平衡信号计算完成，对冲信号: {hedge_signals}个")
        
        # 删除辅助列
        if 'week_start' in result_df.columns:
            result_df.drop(['week_start'], axis=1, inplace=True)
        
        return result_df
