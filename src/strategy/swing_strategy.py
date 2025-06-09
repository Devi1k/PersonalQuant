#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
波段策略模块
实现通道回归均值、形态驱动反转和对冲型仓位平衡等适用于日线周期的波段交易策略
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
import warnings  # Import warnings
import sys

# 设置日志
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)
# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"swing_strategy_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
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
        swing_config = self.config.get("swing", {})
        logger.info(f"波段策略配置: {swing_config}")
        # 形态驱动反转交易参数 - 适用于日线周期
        self.rsi_period = swing_config.get("rsi_period", 14)
        self.rsi_oversold = swing_config.get("rsi_oversold", 30)  # 从40调整到30，适应日线周期
        self.kdj_j_oversold = swing_config.get("kdj_j_oversold", 20)  # 从30调整到20，适应日线周期
        self.head_shoulder_volume_ratio = swing_config.get(
            "head_shoulder_volume_ratio", 0.6
        )  # 从0.5调整到0.6
        self.head_shoulder_breakout_volume_ratio = swing_config.get(
            "head_shoulder_breakout_volume_ratio", 1.5
        )  # 从2.0调整到1.5
        self.triangle_support_volume_ratio = swing_config.get(
            "triangle_support_volume_ratio", 0.8
        )  # 从0.7调整到0.8
        self.flag_pole_min_rise = swing_config.get("flag_pole_min_rise", 0.05)  # 从0.08调整到0.05，适应日线周期
        self.flag_max_retracement = swing_config.get("flag_max_retracement", 0.5)  # 从0.382调整到0.5

        # 通道回归均值策略参数 - 适用于日线周期
        self.bollinger_period = swing_config.get("bollinger_period", 20)
        self.bollinger_std_dev = swing_config.get("bollinger_std_dev", 2.0)  # 从1.8调整到2.0，适应日线周期
        self.bollinger_reversion_bars = swing_config.get("bollinger_reversion_bars", 5)  # 从3调整到5
        self.kdj_params = swing_config.get(
            "kdj_params", (9, 3, 3)
        )  # (K周期, D周期, J周期)
        self.atr_period = swing_config.get("atr_period", 14)
        self.atr_stop_loss_multiplier = swing_config.get(
            "atr_stop_loss_multiplier", 2.0
        )  # 从1.2调整到2.0，适应日线周期

        # 对冲型仓位平衡参数 - 适用于日线和周线
        self.td_sequence_count = swing_config.get("td_sequence_count", 9)  # 从13调整到9
        self.hedge_option_delta = swing_config.get(
            "hedge_option_delta", (0.3, 0.4)
        )  # (最小Delta, 最大Delta)
        self.hedge_position_ratio = swing_config.get(
            "hedge_position_ratio", (0.15, 0.25)
        )  # (最小比例, 最大比例)，从(0.1, 0.2)调整到(0.15, 0.25)
        self.ema_channel_width = swing_config.get(
            "ema_channel_width", 0.05
        )  # EMA通道宽度，从0.03调整到0.05

        # 信号组合权重 - 移除不适用于日线的策略
        default_weights = {
            "bollinger_signal": 2.5,  # 通道回归均值信号
            "pattern_signal": 1.5,  # 形态驱动反转信号，从1.0提高到1.5
            "hedge_signal": 1.0,  # 对冲型仓位平衡信号
        }
        self.signal_weights = swing_config.get("signal_weights", default_weights)
        self.combined_signal_threshold = swing_config.get(
            "combined_signal_threshold", 2.0
        )  # 最终信号合并阈值，从1.8调整到2.0

        logger.info(
            f"波段策略初始化完成，参数：RSI周期={self.rsi_period}, RSI超卖阈值={self.rsi_oversold}, "
            f"KDJ-J超卖阈值={self.kdj_j_oversold}, 布林带周期={self.bollinger_period}, "
            f"布林带标准差={self.bollinger_std_dev}, ATR周期={self.atr_period}, "
            f"ATR止损倍数={self.atr_stop_loss_multiplier}"
        )

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
                    
                    # 移除右肩成交量检查
                    # 原代码：
                    # # 检查右肩成交量是否大于头部的50%
                    # if (result_df.loc[right_shoulder_idx, 'volume'] > 
                    #     result_df.loc[head_idx, 'volume'] * self.head_shoulder_volume_ratio):
                    
                    # 计算颈线（左肩和右肩之间的高点）
                    neck_line = result_df.loc[left_shoulder_idx:right_shoulder_idx, 'high'].max()
                    
                    # 检查是否突破颈线
                    if result_df.loc[i, 'close'] > neck_line:
                        # 移除突破时成交量检查
                        # 原代码：
                        # # 检查突破时成交量是否大于5日均量的2倍
                        # if (result_df.loc[i, 'volume'] > 
                        #     result_df.loc[i, 'volume_ma_5'] * self.head_shoulder_breakout_volume_ratio):
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
                        
                        # 移除第3次回踩支撑线缩量检查
                        # 原代码：
                        # # 检查第3次回踩支撑线是否缩量
                        # if (window["volume"].iloc[third_touch_idx] < 
                        #     window["volume_ma_5"].iloc[third_touch_idx] * self.triangle_support_volume_ratio):
                        
                        # 检查是否突破阻力线
                        if window["close"].iloc[-1] > np.max(highs[:-1]):
                            # 移除突破时放量检查
                            # 原代码：
                            # # 检查突破时是否放量
                            # if window["volume"].iloc[-1] > window["volume_ma_5"].iloc[-1] * 1.5:
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
                    
                    # 移除右肩成交量检查
                    # 原代码：
                    # # 检查右肩成交量是否小于头部的50%（缩量特征）
                    # if (result_df.loc[right_shoulder_idx, 'volume'] < 
                    #     result_df.loc[head_idx, 'volume'] * self.head_shoulder_volume_ratio):
                    
                    # 计算颈线（左肩和右肩之间的低点）
                    neck_line = result_df.loc[left_shoulder_idx:right_shoulder_idx, 'low'].min()
                    
                    # 检查是否跌破颈线
                    if result_df.loc[i, 'close'] < neck_line:
                        # 移除跌破时成交量检查
                        # 原代码：
                        # # 检查跌破时成交量是否大于5日均量1.5倍
                        # if result_df.loc[i, 'volume'] > result_df.loc[i, 'volume_ma_5'] * 1.5:
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
            "bb_upper", "bb_middle", "bb_lower", "kdj_k", "kdj_d", "kdj_j", "atr_14"
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
        result_df["long_stop_loss"] = result_df["close"] - (result_df["atr_14"] * self.atr_stop_loss_multiplier)
        result_df["short_stop_loss"] = result_df["close"] + (result_df["atr_14"] * self.atr_stop_loss_multiplier)
        
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
        result_df.loc[long_exit_condition, "exit_signal"] = 1  # 1表示多头平仓
        
        # 空头平仓条件：
        # 1. 价格触及布林带下轨
        # 2. 价格从上方穿越中轨向下
        # 3. KDJ超卖或金叉
        short_exit_condition = (
            (result_df["touch_lower_band"]) |  # 价格触及下轨
            (result_df["lower_to_middle_cross"]) |  # 价格穿越中轨向上
            (result_df["kdj_oversold"] & result_df["kdj_golden_cross"])  # KDJ超卖且金叉
        )
        result_df.loc[short_exit_condition, "exit_signal"] = -1  # -1表示空头平仓
        
        # 5. 综合通道回归均值信号
        # 0.5 = 买入信号（价格从布林带下轨回归至中轨 + KDJ超卖或金叉）
        # -0.5 = 卖出信号（价格从布林带上轨回归至中轨 + KDJ超买或死叉）
        # 1 = 多头平仓信号
        # -1 = 空头平仓信号
        # 0 = 无信号
        
        # 买入信号
        buy_condition = (
            (result_df["lower_to_middle_cross"]) & 
            (result_df["kdj_oversold"] | result_df["kdj_golden_cross"])
        )
        result_df.loc[buy_condition, "bollinger_signal"] = 0.5
        
        # 卖出信号
        sell_condition = (
            (result_df["upper_to_middle_cross"]) & 
            (result_df["kdj_overbought"] | result_df["kdj_death_cross"])
        )
        result_df.loc[sell_condition, "bollinger_signal"] = -0.5
        
        # 将布林带信号作为最终的通道信号
        result_df["channel_signal"] = result_df["bollinger_signal"]
        
        # 合并平仓信号
        # 如果存在平仓信号，覆盖原有的通道信号
        result_df.loc[result_df["exit_signal"] != 0, "channel_signal"] = result_df["exit_signal"]
        
        # 统计信号数量
        buy_signals = (result_df["channel_signal"] == 0.5).sum()
        sell_signals = (result_df["channel_signal"] == -0.5).sum()
        long_exit_signals = (result_df["channel_signal"] == 1).sum()
        short_exit_signals = (result_df["channel_signal"] == -1).sum()
        
        logger.info(f"通道回归均值策略信号计算完成，买入信号: {buy_signals}个, 卖出信号: {sell_signals}个, 多头平仓信号: {long_exit_signals}个, 空头平仓信号: {short_exit_signals}个")
        
        return result_df

    def combine_signals(self, df, weekly_df=None):
        """
        组合各策略信号，通过调用各个策略方法生成信号并合并

        Parameters
        ----------
        df : pandas.DataFrame
            日线K线数据，包含基础指标
        weekly_df : pandas.DataFrame, default None
            周线数据，用于对冲策略

        Returns
        -------
        pandas.DataFrame
            添加了综合信号的数据框
        """
        if df.empty:
            logger.warning("输入的日线K线数据为空")
            return df

        logger.info("开始组合各策略信号")

        # 基础数据框使用日线K线
        combined_df = df.copy()

        # --- Call Individual Strategies --- 
        # 1. Channel Mean Reversion
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

        # 2. Pattern Driven Reversal
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

        # 3. Hedge Position Balance
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

        # 计算加权信号
        combined_df["weighted_signal"] = (
            combined_df["bollinger_signal"] * self.signal_weights["bollinger_signal"]
            + combined_df["pattern_signal"] * self.signal_weights["pattern_signal"]
            + combined_df["hedge_signal"] * self.signal_weights["hedge_signal"]
        )

        # --- Signal Filtering and Final Decision ---
        logger.info("生成最终信号，优先处理平仓信号")
        combined_df["final_signal"] = 0 # Initialize with 0 (no action)

        # 1. Check for Exit Signals first (Priority)
        # Use .loc for potentially faster setting on large dataframes
        long_exit_mask = combined_df["exit_signal"] == 2
        short_exit_mask = combined_df["exit_signal"] == -2

        combined_df.loc[long_exit_mask, "final_signal"] = 2   # 平多仓
        combined_df.loc[short_exit_mask, "final_signal"] = -2  # 平空仓

        # 2. If no exit signal, check for Entry Signals
        # Mask for rows where no exit signal was triggered
        no_exit_mask = ~(long_exit_mask | short_exit_mask)

        # a) Check for Buy Signal (weighted sum threshold)
        buy_condition = (combined_df["weighted_signal"] > self.combined_signal_threshold) & no_exit_mask
        combined_df.loc[buy_condition, "final_signal"] = 1   # 开多仓

        # b) Check for Sell/Hedge Signal (specific hedge signal, overrides weighted if needed)
        # Apply only where no exit and no buy signal was set
        sell_condition = (combined_df["hedge_signal"] == -1) & no_exit_mask # & ~buy_condition (implicitly handled by applying after buy)
        # We only set sell signal if final_signal is still 0 at this point after checking exits and buy
        combined_df.loc[sell_condition & (combined_df["final_signal"] == 0), "final_signal"] = -1 # 开空仓/对冲

        # 统计最终信号
        final_buy_entry_signals = (combined_df["final_signal"] == 1).sum()
        final_sell_entry_signals = (combined_df["final_signal"] == -1).sum()
        final_long_exit_signals = (combined_df["final_signal"] == 2).sum()
        final_short_exit_signals = (combined_df["final_signal"] == -2).sum()

        logger.info(f"最终信号统计: 开多={final_buy_entry_signals}, 开空={final_sell_entry_signals}, 平多={final_long_exit_signals}, 平空={final_short_exit_signals}")

        logger.info("组合信号计算完成")
        return combined_df
