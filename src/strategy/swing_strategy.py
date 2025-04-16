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
import warnings  # Import warnings

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
        swing_config = self.config.get("strategy", {}).get("swing", {})

        # 多周期动量确认系统参数
        self.rsi_period = swing_config.get("rsi_period", 14)
        self.rsi_oversold = swing_config.get("rsi_oversold", 35)  # 从30调整到35
        self.kdj_j_oversold = swing_config.get("kdj_j_oversold", 25)  # 从20调整到25
        self.macd_bars_increasing = swing_config.get("macd_bars_increasing", 2)  # 从3调整到2
        self.double_bottom_volume_ratio = swing_config.get(
            "double_bottom_volume_ratio", 1.2  # 从1.3调整到1.2
        )

        # 通道回归均值策略参数
        self.bollinger_period = swing_config.get("bollinger_period", 20)
        self.bollinger_std_dev = swing_config.get("bollinger_std_dev", 2.0)
        self.bollinger_reversion_bars = swing_config.get("bollinger_reversion_bars", 3)
        self.kdj_params = swing_config.get(
            "kdj_params", (9, 3, 3)
        )  # (K周期, D周期, J周期)
        self.atr_period = swing_config.get("atr_period", 14)
        self.atr_stop_loss_multiplier = swing_config.get(
            "atr_stop_loss_multiplier", 1.5
        )
        # self.max_breakout_minutes = swing_config.get('max_breakout_minutes', 20) # 注释掉，因为未在代码中使用

        # 形态驱动反转交易参数
        self.head_shoulder_volume_ratio = swing_config.get(
            "head_shoulder_volume_ratio", 0.5
        )
        self.head_shoulder_breakout_volume_ratio = swing_config.get(
            "head_shoulder_breakout_volume_ratio", 2.0
        )
        self.triangle_support_volume_ratio = swing_config.get(
            "triangle_support_volume_ratio", 0.7
        )
        self.flag_pole_min_rise = swing_config.get("flag_pole_min_rise", 0.15)
        self.flag_max_retracement = swing_config.get("flag_max_retracement", 0.382)

        # 对冲型仓位平衡参数
        # self.ema_channel_period = swing_config.get("ema_channel_period", 144)
        self.td_sequence_count = swing_config.get("td_sequence_count", 13)
        self.hedge_option_delta = swing_config.get(
            "hedge_option_delta", (0.3, 0.4)
        )  # (最小Delta, 最大Delta)
        self.hedge_position_ratio = swing_config.get(
            "hedge_position_ratio", (0.15, 0.25)
        )  # (最小比例, 最大比例)
        self.ema_channel_width = swing_config.get(
            "ema_channel_width", 0.05
        )  # EMA通道宽度

        # 量价微观验证参数
        self.vwap_volume_ratio = swing_config.get("vwap_volume_ratio", 1.2)
        self.large_order_inflow_ratio = swing_config.get(
            "large_order_inflow_ratio", 0.4
        )
        self.closing_auction_time = swing_config.get("closing_auction_time", "14:55")

        # 信号组合权重
        default_weights = {
            "momentum_signal": 2.0,  # 多周期动量确认信号
            "bollinger_signal": 2.0,  # 通道回归均值信号
            "pattern_signal": 1.5,  # 形态驱动反转信号
            "hedge_signal": 1.0,  # 对冲型仓位平衡信号
            "volume_price_signal": 1.5,  # 量价微观验证信号
        }
        self.signal_weights = swing_config.get("signal_weights", default_weights)
        self.combined_signal_threshold = swing_config.get(
            "combined_signal_threshold", 2.0
        )  # 最终信号合并阈值

        logger.info(
            f"波段策略初始化完成，参数：RSI周期={self.rsi_period}, RSI超卖阈值={self.rsi_oversold}, "
            f"KDJ-J超卖阈值={self.kdj_j_oversold}, 布林带周期={self.bollinger_period}, "
            f"布林带标准差={self.bollinger_std_dev}, ATR周期={self.atr_period}, "
            f"ATR止损倍数={self.atr_stop_loss_multiplier}"
        )

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
        required_cols = [
            "date",
            "close",
            "open",
            "high",
            "low",
            "volume",
            "rsi_14",
            "kdj_j",
            "bb_lower",
            "bb_middle",
        ]
        for df, timeframe in [
            (df_5min, "5分钟"),
            (df_15min, "15分钟"),
            (df_60min, "60分钟"),
        ]:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"{timeframe}K线数据缺失必要的列: {missing_cols}")
                return df_5min

        logger.info("开始计算多周期动量确认系统信号")

        # 复制数据，避免修改原始数据
        result_df = df_5min.copy()

        # 1. 短期动量：5分钟K线RSI（14周期）<30且KDJ-J线<20（超卖）+ 布林带下轨价格回归至中轨上方
        result_df["short_term_oversold"] = (result_df["rsi_14"] < self.rsi_oversold) & (
            result_df["kdj_j"] < self.kdj_j_oversold
        )

        # 布林带下轨价格回归至中轨上方
        result_df["bb_lower_to_middle_cross"] = (
            (result_df["close"] > result_df["bb_middle"])
            & (result_df["close"].shift(1) <= result_df["bb_middle"].shift(1))
            & (result_df["close"].shift(5) < result_df["bb_lower"].shift(5))
        )

        # 2. 中期过滤：15分钟K线需满足MACD柱线连续3根递增（防止短期假信号）
        # 将15分钟MACD柱状线数据合并到5分钟数据
        df_15min_copy = df_15min.copy()
        df_15min_copy["date_rounded"] = df_15min_copy["date"].apply(
            lambda x: x.replace(minute=(x.minute // 15) * 15, second=0, microsecond=0)
        )

        # 计算15分钟MACD柱状线是否连续递增
        df_15min_copy["macd_hist_increasing"] = False
        for i in range(self.macd_bars_increasing, len(df_15min_copy)):
            increasing = True
            for j in range(1, self.macd_bars_increasing):
                if (
                    df_15min_copy["macd_hist"].iloc[i - j]
                    <= df_15min_copy["macd_hist"].iloc[i - j - 1]
                ):
                    increasing = False
                    break
            df_15min_copy.loc[df_15min_copy.index[i], "macd_hist_increasing"] = (
                increasing
            )

        # 准备15分钟MACD数据用于合并
        df_15min_macd = df_15min_copy[["date_rounded", "macd_hist_increasing"]].rename(
            columns={"date_rounded": "date_15min"}
        )

        # 创建用于合并的辅助列
        result_df["date_15min"] = result_df["date"].apply(
            lambda x: x.replace(minute=(x.minute // 15) * 15, second=0, microsecond=0)
        )

        # 合并15分钟MACD数据
        result_df = pd.merge(result_df, df_15min_macd, on="date_15min", how="left")

        # 填充可能的缺失值
        result_df["macd_hist_increasing"] = result_df["macd_hist_increasing"].fillna(
            False
        )

        # 3. 执行逻辑：观察60分钟K线是否形成双底形态（第二个底不创新低，且右底成交量>左底1.3倍）
        # 将60分钟K线数据合并到5分钟数据
        df_60min_copy = df_60min.copy()
        df_60min_copy["date_rounded"] = df_60min_copy["date"].apply(
            lambda x: x.replace(minute=0, second=0, microsecond=0)
        )

        # 识别60分钟K线的双底形态
        df_60min_copy["is_bottom"] = (
            df_60min_copy["low"] < df_60min_copy["low"].shift(1)
        ) & (df_60min_copy["low"] < df_60min_copy["low"].shift(-1))

        # 查找双底形态 - 向量化实现
        df_60min_copy["double_bottom"] = False
        
        # 获取所有底部点的索引
        bottom_indices = df_60min_copy.index[df_60min_copy["is_bottom"]].tolist()
        
        # 查找符合条件的双底形态
        for i in range(len(bottom_indices) - 1):
            right_bottom_idx = bottom_indices[i+1]
            left_bottom_idx = bottom_indices[i]
            
            # 检查两个底之间的距离是否在10个周期内
            if right_bottom_idx - left_bottom_idx > 10:
                continue
                
            # 检查第二个底是否不创新低
            if df_60min_copy.loc[right_bottom_idx, "low"] >= df_60min_copy.loc[left_bottom_idx, "low"]:
                # 检查右底成交量是否大于左底成交量的1.2倍
                if (df_60min_copy.loc[right_bottom_idx, "volume"] > 
                    df_60min_copy.loc[left_bottom_idx, "volume"] * self.double_bottom_volume_ratio):
                    df_60min_copy.loc[right_bottom_idx, "double_bottom"] = True

        # 准备60分钟双底数据用于合并
        df_60min_bottom = df_60min_copy[["date_rounded", "double_bottom"]].rename(
            columns={"date_rounded": "date_60min"}
        )

        # 创建用于合并的辅助列
        result_df["date_60min"] = result_df["date"].apply(
            lambda x: x.replace(minute=0, second=0, microsecond=0)
        )

        # 合并60分钟双底数据
        result_df = pd.merge(result_df, df_60min_bottom, on="date_60min", how="left")

        # 填充可能的缺失值
        result_df["double_bottom"] = result_df["double_bottom"].fillna(False)

        # 综合多周期动量确认信号
        # 1 = 买入信号（短期超卖 + 布林带回归 + 15分钟MACD柱线递增 + 60分钟双底）
        # 0 = 无信号

        result_df["momentum_signal"] = 0
        result_df.loc[
            result_df["short_term_oversold"]
            & result_df["bb_lower_to_middle_cross"]
            & result_df["macd_hist_increasing"]
            & result_df["double_bottom"],
            "momentum_signal",
        ] = 1

        # 统计信号数量
        buy_signals = (result_df["momentum_signal"] == 1).sum()
        logger.info(f"多周期动量确认系统信号计算完成，买入信号: {buy_signals}个")

        # 删除辅助列
        result_df.drop(["date_15min", "date_60min"], axis=1, inplace=True)

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
        required_cols = [
            "date",
            "close",
            "open",
            "high",
            "low",
            "volume",
            "volume_ma_5",
            "rsi_14",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"形态驱动反转交易策略所需的列缺失: {missing_cols}")
            return df

        logger.info("开始计算形态驱动反转交易信号")

        # 复制数据，避免修改原始数据
        result_df = df.copy()

        # 初始化形态识别列
        result_df["head_shoulder_bottom"] = False
        result_df["ascending_triangle"] = False
        result_df["flag_pattern"] = False
        result_df["head_shoulder_top"] = False
        
        # 识别局部极值点 (向量化操作)
        result_df['is_local_min'] = (result_df['low'] < result_df['low'].shift(1)) & (result_df['low'] < result_df['low'].shift(-1))
        result_df['is_local_max'] = (result_df['high'] > result_df['high'].shift(1)) & (result_df['high'] > result_df['high'].shift(-1))

        # 1. 头肩底形态识别 - 向量化实现
        # 获取所有局部低点的索引
        min_indices = result_df.index[result_df['is_local_min']].tolist()
        
        # 在每个30条记录的窗口中查找头肩底形态
        for i in range(30, len(result_df)):
            # 获取当前窗口内的局部低点
            window_min_indices = [idx for idx in min_indices if i-30 <= idx <= i]
            
            # 如果找到至少三个低点
            if len(window_min_indices) >= 3:
                # 获取最近的三个低点
                left_shoulder_idx = window_min_indices[-3]
                head_idx = window_min_indices[-2]
                right_shoulder_idx = window_min_indices[-1]
                
                # 检查头部是否低于两肩
                if (result_df.loc[head_idx, 'low'] < result_df.loc[left_shoulder_idx, 'low'] and
                    result_df.loc[head_idx, 'low'] < result_df.loc[right_shoulder_idx, 'low']):
                    
                    # 检查右肩成交量是否大于头部的50%
                    if (result_df.loc[right_shoulder_idx, 'volume'] > 
                        result_df.loc[head_idx, 'volume'] * self.head_shoulder_volume_ratio):
                        
                        # 计算颈线（左肩和右肩之间的高点）
                        neck_line = result_df.loc[left_shoulder_idx:right_shoulder_idx, 'high'].max()
                        
                        # 检查是否突破颈线
                        if result_df.loc[i, 'close'] > neck_line:
                            # 检查突破时成交量是否大于5日均量的2倍
                            if (result_df.loc[i, 'volume'] > 
                                result_df.loc[i, 'volume_ma_5'] * self.head_shoulder_breakout_volume_ratio):
                                result_df.loc[i, 'head_shoulder_bottom'] = True

        # 2. 上升三角形识别 - 向量化实现
        for i in range(20, len(result_df)):
            window = result_df.iloc[i-20:i+1]
            
            # 查找水平支撑线（至少三次触及）
            lows = window["low"].values
            support_levels = np.linspace(min(lows), max(lows), 30)  # 从20增加到30
            
            max_touches = 0
            best_support = None
            
            for level in support_levels:
                # 计算接近该水平线的点数
                threshold = (max(lows) - min(lows)) * 0.02  # 从0.03减小到0.02
                touches = np.sum(np.abs(lows - level) < threshold)
                if touches >= 3 and touches > max_touches:
                    max_touches = touches
                    best_support = level
            
            if best_support is not None:
                # 查找下降阻力线（至少两次触及）
                highs = window["high"].values
                threshold = (max(highs) - min(highs)) * 0.02  # 从0.03减小到0.02
                resistance_touches = np.where(np.abs(highs - np.maximum.accumulate(highs)) < threshold)[0]
                
                if len(resistance_touches) >= 2:
                    # 找出触及支撑线的点
                    support_touches = np.where(np.abs(lows - best_support) < threshold)[0]
                    
                    if len(support_touches) >= 3:
                        third_touch_idx = support_touches[2]
                        
                        # 检查第3次回踩支撑线是否缩量
                        if (window["volume"].iloc[third_touch_idx] < 
                            window["volume_ma_5"].iloc[third_touch_idx] * self.triangle_support_volume_ratio):
                            
                            # 检查是否突破阻力线
                            if window["close"].iloc[-1] > np.max(highs[:-1]):
                                # 检查突破时是否放量
                                if window["volume"].iloc[-1] > window["volume_ma_5"].iloc[-1] * 1.5:
                                    result_df.loc[result_df.index[i], "ascending_triangle"] = True

        # 3. 旗形整理识别 - 向量化实现
        for i in range(15, len(result_df)):
            window = result_df.iloc[i-15:i+1]
            
            # 查找可能的旗杆（快速上涨）
            for j in range(5, min(10, len(window))):
                pole_start = window["low"].iloc[0]
                pole_end = window["high"].iloc[j]
                pole_rise = (pole_end - pole_start) / pole_start
                
                # 检查旗杆涨幅是否大于15%
                if pole_rise > self.flag_pole_min_rise:
                    # 检查回调幅度是否小于38.2%斐波那契位
                    retracement = (pole_end - window["low"].iloc[j:].min()) / (pole_end - pole_start)
                    
                    if retracement < self.flag_max_retracement:
                        # 检查是否突破旗形上轨
                        if window["close"].iloc[-1] > window["high"].iloc[j:-1].max():
                            result_df.loc[result_df.index[i], "flag_pattern"] = True
                            break

        # 4. 头肩顶形态识别 - 向量化实现
        # 获取所有局部高点的索引
        max_indices = result_df.index[result_df['is_local_max']].tolist()
        
        for i in range(30, len(result_df)):
            # 获取当前窗口内的局部高点
            window_max_indices = [idx for idx in max_indices if i-30 <= idx <= i]
            
            # 如果找到至少三个高点
            if len(window_max_indices) >= 3:
                # 获取最近的三个高点
                left_shoulder_idx = window_max_indices[-3]
                head_idx = window_max_indices[-2]
                right_shoulder_idx = window_max_indices[-1]
                
                # 检查头部是否高于两肩
                if (result_df.loc[head_idx, 'high'] > result_df.loc[left_shoulder_idx, 'high'] and
                    result_df.loc[head_idx, 'high'] > result_df.loc[right_shoulder_idx, 'high']):
                    
                    # 检查右肩成交量是否小于头部的50%（缩量特征）
                    if (result_df.loc[right_shoulder_idx, 'volume'] < 
                        result_df.loc[head_idx, 'volume'] * self.head_shoulder_volume_ratio):
                        
                        # 计算颈线（左肩和右肩之间的低点）
                        neck_line = result_df.loc[left_shoulder_idx:right_shoulder_idx, 'low'].min()
                        
                        # 检查是否跌破颈线
                        if result_df.loc[i, 'close'] < neck_line:
                            # 检查跌破时成交量是否大于5日均量1.5倍
                            if result_df.loc[i, 'volume'] > result_df.loc[i, 'volume_ma_5'] * 1.5:
                                result_df.loc[i, 'head_shoulder_top'] = True

        # 5. 计算RSI背离 - 向量化实现
        # 初始化背离列
        result_df["rsi_bullish_divergence"] = False
        result_df["rsi_bearish_divergence"] = False
        
        # 计算价格和RSI的局部极值
        result_df['price_new_low'] = result_df['low'] < result_df['low'].rolling(5, min_periods=1).min().shift(1)
        result_df['price_new_high'] = result_df['high'] > result_df['high'].rolling(5, min_periods=1).max().shift(1)
        
        result_df['rsi_higher_than_min'] = result_df['rsi_14'] > result_df['rsi_14'].rolling(5, min_periods=1).min().shift(1)
        result_df['rsi_lower_than_max'] = result_df['rsi_14'] < result_df['rsi_14'].rolling(5, min_periods=1).max().shift(1)
        
        # 计算背离
        result_df.loc[result_df['price_new_low'] & result_df['rsi_higher_than_min'], 'rsi_bullish_divergence'] = True
        result_df.loc[result_df['price_new_high'] & result_df['rsi_lower_than_max'], 'rsi_bearish_divergence'] = True
        
        # 清理临时列
        result_df.drop(['price_new_low', 'price_new_high', 'rsi_higher_than_min', 'rsi_lower_than_max'], 
                       axis=1, inplace=True, errors='ignore')
        
        # 综合形态驱动反转信号
        # 1 = 买入信号（头肩底或上升三角形或旗形整理，且有RSI看涨背离）
        # -1 = 卖出信号（头肩顶形态 或 RSI看跌背离与其他指标结合）
        # 0 = 无信号

        result_df["pattern_signal"] = 0

        # 买入信号
        result_df.loc[
            (
                (result_df["head_shoulder_bottom"])
                | (result_df["ascending_triangle"])
                | (result_df["flag_pattern"])
            )
            & (result_df["rsi_bullish_divergence"]),
            "pattern_signal",
        ] = 1

        # 卖出信号 - 增强版：需要满足多个条件之一
        # 1. 头肩顶形态确认
        # 2. RSI看跌背离 + 价格跌破20日均线
        # 3. RSI看跌背离 + 成交量放大（大于5日均量1.5倍）
        
        # 计算20日均线
        if "ma_20" not in result_df.columns:
            result_df["ma_20"] = result_df["close"].rolling(20).mean()
            
        # 检测价格跌破20日均线
        result_df["below_ma20"] = result_df["close"] < result_df["ma_20"]
        
        # 检测成交量放大
        result_df["volume_surge"] = result_df["volume"] > result_df["volume_ma_5"] * 1.5
        
        # 设置卖出信号 - 修改为与买入信号对称的逻辑
        result_df.loc[
            (
                (result_df["head_shoulder_top"])  # 头肩顶形态
                | (result_df["below_ma20"] & result_df["volume_surge"])  # 跌破均线 + 放量
            )
            & (result_df["rsi_bearish_divergence"]),  # 必须有RSI看跌背离
            "pattern_signal"
        ] = -1

        # 统计信号数量
        buy_signals = (result_df["pattern_signal"] == 1).sum()
        sell_signals = (result_df["pattern_signal"] == -1).sum()
        logger.info(
            f"形态驱动反转交易信号计算完成，买入信号: {buy_signals}个, 卖出信号: {sell_signals}个"
        )

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

        # 计算EMA通道上下轨 (使用配置参数)
        result_df["ema_upper"] = result_df["ema_144"] * (1 + self.ema_channel_width)
        result_df["ema_lower"] = result_df["ema_144"] * (1 - self.ema_channel_width)

        # 检测EMA通道上轨突破
        result_df["ema_upper_breakout"] = result_df["close"] > result_df["ema_upper"]

        # 计算周线TD序列（如果提供了周线数据）
        td_setup_count = 0
        weekly_df["td_countdown"] = 0  # 初始化列
        if weekly_df is not None and not weekly_df.empty:
            # 确保周线数据包含必要的列
            if all(
                col in weekly_df.columns for col in ["date", "close", "high", "low"]
            ):
                # 向量化实现TD序列计算
                weekly_df["td_setup_sell"] = 0
                
                # 计算价格是否高于4周期前的价格
                weekly_df["price_higher_than_4_ago"] = weekly_df["close"] > weekly_df["close"].shift(4)
                
                # 使用累积和技术计算连续的上涨周数
                # 每当条件不满足时，重置计数
                weekly_df["reset_group"] = (~weekly_df["price_higher_than_4_ago"]).cumsum()
                weekly_df["consecutive_count"] = weekly_df.groupby("reset_group")["price_higher_than_4_ago"].cumsum()
                
                # 将计数限制在9以内
                weekly_df["td_setup_sell"] = weekly_df["consecutive_count"].where(weekly_df["price_higher_than_4_ago"], 0).clip(upper=9)
                
                # 初始化TD Countdown列
                weekly_df["td_countdown"] = 0
                
                # 找出所有TD Setup完成的点 (等于9)
                setup_complete_indices = weekly_df.index[weekly_df["td_setup_sell"] == 9].tolist()
                
                # 对每个完成的setup点开始计算countdown
                for start_idx in setup_complete_indices:
                    if start_idx + 2 >= len(weekly_df):  # 需要至少2个后续周期
                        continue
                        
                    countdown = 0
                    for i in range(start_idx, len(weekly_df)):
                        # Countdown条件: 收盘价高于或等于2周期前的最高价
                        if i >= 2 and weekly_df["close"].iloc[i] >= weekly_df["high"].iloc[i-2]:
                            countdown += 1
                            weekly_df.loc[weekly_df.index[i], "td_countdown"] = countdown
                            
                            # 达到目标计数后重置
                            if countdown >= self.td_sequence_count:
                                break
                
                # 清理临时列
                weekly_df.drop(["price_higher_than_4_ago", "reset_group", "consecutive_count"], 
                              axis=1, inplace=True, errors='ignore')
                
                # 将周线TD序列数据合并到日线数据
                # 创建日期映射函数，将日线日期映射到对应的周
                def map_to_week(date):
                    # 获取日期所在周的周一日期
                    week_start = date - pd.Timedelta(days=date.weekday())
                    return week_start

                # 在周线数据中添加周开始日期列
                weekly_df["week_start"] = weekly_df["date"].apply(map_to_week)

                # 在日线数据中添加周开始日期列
                result_df["week_start"] = result_df["date"].apply(map_to_week)

                # 准备周线TD序列数据用于合并
                weekly_td = weekly_df[["week_start", "td_countdown"]].copy()
                
                # 修复未来函数问题：将TD Countdown值向前移动一周
                # 这样本周的日线数据只能使用上周的TD Countdown结果
                weekly_td["td_countdown_shifted"] = weekly_td["td_countdown"].shift(1)
                
                # 合并周线TD序列数据到日线数据（使用移动后的数据）
                result_df = pd.merge(result_df, 
                                     weekly_td[["week_start", "td_countdown_shifted"]], 
                                     on="week_start", 
                                     how="left")

                # 填充可能的缺失值
                result_df["td_countdown"] = result_df["td_countdown_shifted"].fillna(0)
                
                # 删除临时列
                result_df.drop(["td_countdown_shifted"], axis=1, inplace=True, errors='ignore')
            else:
                logger.warning("周线数据缺少必要的列，无法计算TD序列")
                result_df["td_countdown"] = 0
        else:
            logger.warning("未提供周线数据或周线数据列不全，无法计算TD序列")
            result_df["td_countdown"] = 0

        # 2. 对冲信号：EMA通道上轨突破 + TD序列13计数
        result_df["hedge_signal"] = 0
        result_df.loc[
            (result_df["ema_upper_breakout"])
            & (result_df["td_countdown"] >= self.td_sequence_count),
            "hedge_signal",
        ] = -1  # 做空信号

        # 3. 对冲仓位计算
        # 基于波动率锥动态调整对冲仓位比例

        # 计算20日波动率
        result_df["returns"] = result_df["close"].pct_change()
        result_df["volatility_20d"] = result_df["returns"].rolling(20).std() * np.sqrt(
            252
        )

        # 计算波动率百分位（相对于过去一年）
        vol_percentile = (
            result_df["volatility_20d"]
            .rolling(252)
            .apply(
                lambda x: (
                    pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 1 else np.nan
                ),  # Need at least 2 points to rank
                raw=False,  # rank needs Series
            )
            .fillna(0.5)
        )  # Fill NaN with 0.5 (median assumption)

        # 根据波动率百分位动态调整对冲仓位比例
        min_ratio, max_ratio = self.hedge_position_ratio
        # 线性插值计算对冲比例：波动率低时比例低，波动率高时比例高
        result_df["hedge_ratio"] = min_ratio + (max_ratio - min_ratio) * vol_percentile

        # 对冲信号触发时才应用对冲比例，否则为0
        result_df.loc[result_df["hedge_signal"] != -1, "hedge_ratio"] = 0

        logger.info(
            f"对冲型仓位平衡信号计算完成，对冲信号: {(result_df['hedge_signal'] == -1).sum()}个"
        )

        # 返回结果，包含 hedge_signal 和 hedge_ratio
        return result_df[["date", "hedge_signal", "hedge_ratio"]]

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

        # 1. 入场校验：价格突破时需满足成交量>当日VWAP的120% - 向量化实现
        result_df["volume_gt_vwap"] = result_df["volume"] > (
            result_df["vwap"] * self.vwap_volume_ratio
        )

        # 2. 主力资金流向监控（如果提供了Level2数据）- 向量化实现
        if level2_data is not None and not level2_data.empty:
            # 确保Level2数据包含必要的列
            if all(
                col in level2_data.columns
                for col in ["date", "large_order_inflow_ratio"]
            ):
                # 合并Level2数据
                level2_data_copy = level2_data[
                    ["date", "large_order_inflow_ratio"]
                ].copy()
                result_df = pd.merge(result_df, level2_data_copy, on="date", how="left")

                # 填充可能的缺失值
                result_df["large_order_inflow_ratio"] = result_df[
                    "large_order_inflow_ratio"
                ].fillna(0)
            else:
                logger.warning("Level2数据缺少必要的列，无法监控主力资金流向")
                result_df["large_order_inflow_ratio"] = 0
        else:
            logger.warning("未提供Level2数据，无法监控主力资金流向")
            result_df["large_order_inflow_ratio"] = 0

        # 3. 量价微观验证信号 - 向量化实现
        # 1 = 买入信号（成交量>VWAP的120% + 主力资金净流入占比>40%）
        # 0 = 无信号
        result_df["volume_price_signal"] = 0
        buy_condition = (result_df["volume_gt_vwap"]) & (result_df["large_order_inflow_ratio"] > self.large_order_inflow_ratio)
        result_df.loc[buy_condition, "volume_price_signal"] = 1

        # 4. 出场优化：设定14:55的尾盘集中成交算法（TWAP策略降低滑点）- 向量化实现
        # 标记接近收盘时间的K线
        result_df["closing_auction"] = result_df["date"].apply(
            lambda x: x.strftime("%H:%M") == self.closing_auction_time
        )

        # 统计信号数量
        buy_signals = (result_df["volume_price_signal"] == 1).sum()
        logger.info(f"量价微观验证信号计算完成，买入信号: {buy_signals}个")

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
        required_cols = [
            "date", "close", "open", "high", "low", "volume", 
            "bb_upper", "bb_middle", "bb_lower", "kdj_k", "kdj_d", "kdj_j", "atr"
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"通道回归均值策略所需的列缺失: {missing_cols}")
            return df

        logger.info("开始计算通道回归均值策略信号")

        # 复制数据，避免修改原始数据
        result_df = df.copy()

        # 1. 布林带通道回归策略 - 向量化实现
        # 初始化信号列
        result_df["bollinger_signal"] = 0
        
        # 计算价格相对于布林带的位置
        result_df["bb_position"] = (result_df["close"] - result_df["bb_lower"]) / (result_df["bb_upper"] - result_df["bb_lower"])
        
        # 检测价格触及或突破布林带下轨
        result_df["touch_lower_band"] = result_df["low"] <= result_df["bb_lower"]
        
        # 检测价格从布林带下轨向上回归至中轨的过程
        result_df["lower_to_middle_cross"] = (
            (result_df["close"] > result_df["bb_middle"]) & 
            (result_df["close"].shift(1) <= result_df["bb_middle"].shift(1)) &
            (result_df["low"].shift(1).rolling(window=self.bollinger_reversion_bars).min() <= result_df["bb_lower"].shift(1).rolling(window=self.bollinger_reversion_bars).min())
        )
        
        # 检测价格触及或突破布林带上轨
        result_df["touch_upper_band"] = result_df["high"] >= result_df["bb_upper"]
        
        # 检测价格从布林带上轨向下回归至中轨的过程
        result_df["upper_to_middle_cross"] = (
            (result_df["close"] < result_df["bb_middle"]) & 
            (result_df["close"].shift(1) >= result_df["bb_middle"].shift(1)) &
            (result_df["high"].shift(1).rolling(window=self.bollinger_reversion_bars).max() >= result_df["bb_upper"].shift(1).rolling(window=self.bollinger_reversion_bars).max())
        )
        
        # 2. KDJ指标确认 - 向量化实现
        # 检测KDJ超买超卖状态
        result_df["kdj_oversold"] = (result_df["kdj_j"] < 20)  # 仅使用 J 值判断超卖
        result_df["kdj_overbought"] = (result_df["kdj_j"] > 80)  # 仅使用 J 值判断超买
        
        # 检测KDJ金叉和死叉
        result_df["kdj_golden_cross"] = (result_df["kdj_k"] > result_df["kdj_d"]) & (result_df["kdj_k"].shift(1) <= result_df["kdj_d"].shift(1))
        result_df["kdj_death_cross"] = (result_df["kdj_k"] < result_df["kdj_d"]) & (result_df["kdj_k"].shift(1) >= result_df["kdj_d"].shift(1))
        
        # 3. ATR止损设置 - 仅计算止损价格，不执行止损逻辑
        # 计算ATR止损价格，供回测框架使用
        result_df["long_stop_loss"] = result_df["close"] - (result_df["atr"] * self.atr_stop_loss_multiplier)
        result_df["short_stop_loss"] = result_df["close"] + (result_df["atr"] * self.atr_stop_loss_multiplier)
        
        # 4. 平仓/止盈逻辑
        # 初始化平仓信号列
        result_df["exit_signal"] = 0
        
        # 多头平仓条件：
        # 1. 价格触及布林带上轨
        # 2. 价格从下方穿越中轨向上
        # 3. KDJ超买或死叉
        long_exit_condition = (
            (result_df["touch_upper_band"]) |  # 价格触及上轨
            (result_df["upper_to_middle_cross"]) |  # 价格穿越中轨向下
            (result_df["kdj_overbought"] & result_df["kdj_death_cross"])  # KDJ超买且死叉
        )
        result_df.loc[long_exit_condition, "exit_signal"] = 2  # 2表示多头平仓
        
        # 空头平仓条件：
        # 1. 价格触及布林带下轨
        # 2. 价格从上方穿越中轨向下
        # 3. KDJ超卖或金叉
        short_exit_condition = (
            (result_df["touch_lower_band"]) |  # 价格触及下轨
            (result_df["lower_to_middle_cross"]) |  # 价格穿越中轨向上
            (result_df["kdj_oversold"] & result_df["kdj_golden_cross"])  # KDJ超卖且金叉
        )
        result_df.loc[short_exit_condition, "exit_signal"] = -2  # -2表示空头平仓
        
        # 5. 综合通道回归均值信号
        # 1 = 买入信号（价格从布林带下轨回归至中轨 + KDJ超卖或金叉）
        # -1 = 卖出信号（价格从布林带上轨回归至中轨 + KDJ超买或死叉）
        # 2 = 多头平仓信号
        # -2 = 空头平仓信号
        # 0 = 无信号
        
        # 买入信号
        buy_condition = (
            (result_df["lower_to_middle_cross"]) & 
            (result_df["kdj_oversold"] | result_df["kdj_golden_cross"])
        )
        result_df.loc[buy_condition, "bollinger_signal"] = 1
        
        # 卖出信号
        sell_condition = (
            (result_df["upper_to_middle_cross"]) & 
            (result_df["kdj_overbought"] | result_df["kdj_death_cross"])
        )
        result_df.loc[sell_condition, "bollinger_signal"] = -1
        
        # 将布林带信号作为最终的通道信号
        result_df["channel_signal"] = result_df["bollinger_signal"]
        
        # 合并平仓信号
        # 如果存在平仓信号，覆盖原有的通道信号
        result_df.loc[result_df["exit_signal"] != 0, "channel_signal"] = result_df["exit_signal"]
        
        # 统计信号数量
        buy_signals = (result_df["channel_signal"] == 1).sum()
        sell_signals = (result_df["channel_signal"] == -1).sum()
        long_exit_signals = (result_df["channel_signal"] == 2).sum()
        short_exit_signals = (result_df["channel_signal"] == -2).sum()
        
        logger.info(f"通道回归均值策略信号计算完成，买入信号: {buy_signals}个, 卖出信号: {sell_signals}个, 多头平仓信号: {long_exit_signals}个, 空头平仓信号: {short_exit_signals}个")
        
        return result_df

    def combine_signals(self, df_5min, df_15min, df_60min, weekly_df=None, level2_data=None):
        """
        组合各策略信号，通过调用各个策略方法生成信号并合并

        Parameters
        ----------
        df_5min : pandas.DataFrame
            5分钟K线数据，包含基础指标
        df_15min : pandas.DataFrame
            15分钟K线数据，包含基础指标
        df_60min : pandas.DataFrame
            60分钟K线数据，包含基础指标
        weekly_df : pandas.DataFrame, default None
            周线数据，用于对冲策略
        level2_data : pandas.DataFrame, default None
            Level2数据，用于量价验证

        Returns
        -------
        pandas.DataFrame
            添加了综合信号的数据框（基于5分钟K线）
        """
        if df_5min.empty:
            logger.warning("输入的5分钟K线数据为空")
            return df_5min

        logger.info("开始组合各策略信号")

        # 基础数据框使用5分钟K线
        combined_df = df_5min.copy()

        # --- Call Individual Strategies --- 
        # 1. Multi-Timeframe Momentum
        momentum_df = self.multi_timeframe_momentum_system(df_5min, df_15min, df_60min)
        if not momentum_df.empty and 'momentum_signal' in momentum_df.columns:
            combined_df = pd.merge(combined_df, momentum_df[['date', 'momentum_signal']], on='date', how='left')
            combined_df['momentum_signal'] = combined_df['momentum_signal'].fillna(0)
        else:
            logger.warning("多周期动量信号计算失败或未返回信号列，填充为0")
            combined_df['momentum_signal'] = 0

        # 2. Channel Mean Reversion
        # Call the strategy
        channel_df = self.channel_mean_reversion_strategy(combined_df) # Assuming it now returns bollinger_signal, exit_signal, etc.

        # Define the columns we expect from this strategy
        channel_cols_to_merge = ['date', 'bollinger_signal', 'exit_signal', 'long_stop_loss', 'short_stop_loss']
        signal_cols = ['bollinger_signal', 'exit_signal'] # Columns to fillna(0)

        # Check if the result is not empty and contains the primary signal ('bollinger_signal')
        if not channel_df.empty and 'bollinger_signal' in channel_df.columns:
            # Select only the columns we need for merging, handling missing optional columns
            cols_present = [col for col in channel_cols_to_merge if col in channel_df.columns]
            merge_df = channel_df[cols_present]

            # Merge the signals and stop-loss levels
            combined_df = pd.merge(combined_df, merge_df, on='date', how='left')

            # Fill NaN values for signal columns only
            for col in signal_cols:
                if col in combined_df.columns:
                    combined_df[col] = combined_df[col].fillna(0)
                else:
                     combined_df[col] = 0 # Ensure column exists even if merge failed for it
                     logger.warning(f"列 '{col}' 在合并后未找到或全为NaN，已填充为 0")

            # Ensure stop loss columns exist after merge, fill with NaN if missing
            for col in ['long_stop_loss', 'short_stop_loss']:
                 if col not in combined_df.columns:
                     combined_df[col] = np.nan # Use NaN for missing price levels
                     logger.warning(f"列 '{col}' 在合并后未找到，已填充为 NaN")
                 else:
                     # If column exists but merge resulted in NaNs, keep them as NaN (don't fillna price levels with 0)
                     pass # Already handled by the merge or initial state

        else:
            logger.warning("通道回归均值策略信号计算失败或未返回 'bollinger_signal' 列，相关信号填充为0/NaN")
            # Ensure all expected columns exist and are filled appropriately if the strategy call fails
            for col in signal_cols:
                if col not in combined_df.columns: combined_df[col] = 0
            for col in ['long_stop_loss', 'short_stop_loss']:
                if col not in combined_df.columns: combined_df[col] = np.nan

        # 3. Pattern Driven Reversal
        pattern_df = self.pattern_driven_reversal_strategy(combined_df) # Pass the potentially modified combined_df
        if not pattern_df.empty and 'pattern_signal' in pattern_df.columns:
             # Merge pattern_signal back, avoiding column duplication if pattern_df is combined_df
            if 'pattern_signal' not in combined_df.columns:
                combined_df = pd.merge(combined_df, pattern_df[['date', 'pattern_signal']], on='date', how='left')
            else: # Update the existing column if pattern_df was combined_df
                 combined_df['pattern_signal'] = pattern_df['pattern_signal']
            combined_df['pattern_signal'] = combined_df['pattern_signal'].fillna(0)
        else:
            logger.warning("形态驱动反转信号计算失败或未返回信号列，填充为0")
            combined_df['pattern_signal'] = 0

        # 4. Hedge Position Balance
        # Pass the latest combined_df and weekly_df
        hedge_df = self.hedge_position_balance(combined_df, weekly_df) 
        if not hedge_df.empty and 'hedge_signal' in hedge_df.columns:
            merge_cols = ['date', 'hedge_signal']
            if 'hedge_ratio' in hedge_df.columns:
                 merge_cols.append('hedge_ratio')
            # Merge back into combined_df, handle potential existing columns from hedge_df if it modified combined_df inplace
            update_cols = [col for col in merge_cols if col != 'date']
            if all(col in combined_df.columns for col in update_cols):
                combined_df[update_cols] = hedge_df[update_cols]
            else:
                 combined_df = pd.merge(combined_df, hedge_df[merge_cols], on='date', how='left')

            combined_df['hedge_signal'] = combined_df['hedge_signal'].fillna(0)
            if 'hedge_ratio' in combined_df.columns:
                 combined_df['hedge_ratio'] = combined_df['hedge_ratio'].fillna(0)
            else:
                 combined_df['hedge_ratio'] = 0 # Ensure column exists
        else:
            logger.warning("对冲型仓位平衡信号计算失败或未返回信号列，填充为0")
            combined_df['hedge_signal'] = 0
            combined_df['hedge_ratio'] = 0 # If signal fails, ratio is also 0

        # 5. Volume Price Microvalidation
        # Pass the latest combined_df and level2_data
        volume_price_df = self.volume_price_microvalidation(combined_df, level2_data) 
        if not volume_price_df.empty and 'volume_price_signal' in volume_price_df.columns:
             # Merge back into combined_df, handle potential existing column
            if 'volume_price_signal' in combined_df.columns:
                 combined_df['volume_price_signal'] = volume_price_df['volume_price_signal']
            else:
                combined_df = pd.merge(combined_df, volume_price_df[['date', 'volume_price_signal']], on='date', how='left')
            combined_df['volume_price_signal'] = combined_df['volume_price_signal'].fillna(0)
        else:
            logger.warning("量价微观验证信号计算失败或未返回信号列，填充为0")
            combined_df['volume_price_signal'] = 0

        # --- Signal Filtering and Final Decision ---
        logger.info("生成最终信号，优先处理平仓信号")
        combined_df["combined_signal"] = 0 # Initialize with 0 (no action)

        # 1. Check for Exit Signals first (Priority)
        # Use .loc for potentially faster setting on large dataframes
        long_exit_mask = combined_df["exit_signal"] == 2
        short_exit_mask = combined_df["exit_signal"] == -2

        combined_df.loc[long_exit_mask, "combined_signal"] = 2   # 平多仓
        combined_df.loc[short_exit_mask, "combined_signal"] = -2  # 平空仓

        # 2. If no exit signal, check for Entry Signals
        # Mask for rows where no exit signal was triggered
        no_exit_mask = ~(long_exit_mask | short_exit_mask)

        # a) Check for Buy Signal (weighted sum threshold)
        buy_condition = (combined_df["weighted_signal"] > self.combined_signal_threshold) & no_exit_mask
        combined_df.loc[buy_condition, "combined_signal"] = 1   # 开多仓

        # b) Check for Sell/Hedge Signal (specific hedge signal, overrides weighted if needed)
        # Apply only where no exit and no buy signal was set
        sell_condition = (combined_df["hedge_signal"] == -1) & no_exit_mask # & ~buy_condition (implicitly handled by applying after buy)
        # We only set sell signal if combined_signal is still 0 at this point after checking exits and buy
        combined_df.loc[sell_condition & (combined_df["combined_signal"] == 0), "combined_signal"] = -1 # 开空仓/对冲

        # 统计最终信号
        final_buy_entry_signals = (combined_df["combined_signal"] == 1).sum()
        final_sell_entry_signals = (combined_df["combined_signal"] == -1).sum()
        final_long_exit_signals = (combined_df["combined_signal"] == 2).sum()
        final_short_exit_signals = (combined_df["combined_signal"] == -2).sum()

        logger.info(f"最终信号统计: 开多={final_buy_entry_signals}, 开空={final_sell_entry_signals}, 平多={final_long_exit_signals}, 平空={final_short_exit_signals}")

        logger.info("组合信号计算完成")
        return combined_df
