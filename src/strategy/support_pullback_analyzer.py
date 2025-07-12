#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
支撑位识别和成交量分析模块
实现支撑位识别和成交量萎缩分析，作为独立的交易信号
"""

import pandas as pd
import numpy as np
import logging
import talib
from datetime import datetime
from pathlib import Path
import sys

# 设置项目路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

# 设置日志
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"support_pullback_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SupportPullbackAnalyzer:
    """支撑位识别和成交量分析类"""
    
    def __init__(self, config=None):
        """
        初始化支撑位识别和成交量分析器
        
        Parameters
        ----------
        config : dict, default None
            策略配置参数
        """
        self.config = config or {}
        
        # 从配置中获取参数，如果没有则使用默认值
        support_config = self.config.get('support_pullback', {})
        
        # 移动平均线支撑检测参数
        self.ma_support_threshold = support_config.get('ma_support_threshold', 0.02)  # 价格接近MA的阈值（2%）
        self.ma_touch_threshold = support_config.get('ma_touch_threshold', 0.005)  # 价格触及MA的阈值（0.5%）
        self.ma_periods = support_config.get('ma_periods', [10, 20, 60])  # MA周期
        
        # 平台/颈线支撑识别参数
        self.platform_lookback = support_config.get('platform_lookback', 20)  # 平台识别回看周期
        self.platform_min_duration = support_config.get('platform_min_duration', 5)  # 平台最小持续时间
        self.platform_price_range = support_config.get('platform_price_range', 0.03)  # 平台价格波动范围（3%）
        self.neckline_threshold = support_config.get('neckline_threshold', 0.01)  # 颈线突破阈值（1%）
        
        # 布林带支撑分析参数
        self.bb_period = support_config.get('bb_period', 20)  # 布林带周期
        self.bb_std_dev = support_config.get('bb_std_dev', 2.0)  # 布林带标准差倍数
        self.bb_support_threshold = support_config.get('bb_support_threshold', 0.01)  # 布林带支撑阈值（1%）
        
        # 成交量萎缩分析参数
        self.volume_ma_period = support_config.get('volume_ma_period', 20)  # 成交量均线周期
        self.volume_contraction_ratio = support_config.get('volume_contraction_ratio', 0.7)  # 成交量萎缩比例（70%）
        self.volume_lookback = support_config.get('volume_lookback', 5)  # 成交量回看周期
        
        # RSI分析参数
        self.rsi_period = support_config.get('rsi_period', 14)  # RSI周期
        self.rsi_overbought_threshold = support_config.get('rsi_overbought_threshold', 70)  # 超买阈值
        self.rsi_oversold_threshold = support_config.get('rsi_oversold_threshold', 30)  # 超卖阈值
        self.rsi_extreme_overbought = support_config.get('rsi_extreme_overbought', 80)  # 极度超买阈值
        self.rsi_pullback_zone_low = support_config.get('rsi_pullback_zone_low', 45)  # 回调区间下限
        self.rsi_pullback_zone_high = support_config.get('rsi_pullback_zone_high', 55)  # 回调区间上限
        self.rsi_oversold_recovery_zone = support_config.get('rsi_oversold_recovery_zone', 40)  # 超卖恢复区间
        self.rsi_prolonged_overbought_periods = support_config.get('rsi_prolonged_overbought_periods', 5)  # 持续超买周期数
        
        # MFI分析参数
        self.mfi_period = support_config.get('mfi_period', 14)  # MFI周期
        self.mfi_oversold_threshold = support_config.get('mfi_oversold_threshold', 30)  # MFI超卖阈值
        self.mfi_overbought_threshold = support_config.get('mfi_overbought_threshold', 70)  # MFI超买阈值
        self.mfi_recovery_threshold = support_config.get('mfi_recovery_threshold', 35)  # MFI恢复阈值
        self.mfi_momentum_lookback = support_config.get('mfi_momentum_lookback', 3)  # MFI动能回看周期
        self.mfi_inflow_threshold = support_config.get('mfi_inflow_threshold', 50)  # 资金流入阈值
        self.price_stability_threshold = support_config.get('price_stability_threshold', 0.02)  # 价格稳定阈值（2%）
        
        logger.info(f"支撑位识别和成交量分析器初始化完成，参数：MA支撑阈值={self.ma_support_threshold:.2%}, "
                   f"平台回看周期={self.platform_lookback}, 成交量萎缩比例={self.volume_contraction_ratio:.2%}, "
                   f"RSI周期={self.rsi_period}, MFI周期={self.mfi_period}")
    
    def identify_ma_support(self, df):
        """
        移动平均线支撑检测
        
        识别价格接近或触及MA10、MA20、MA60的情况
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含价格和移动平均线数据的数据框
            
        Returns
        -------
        pandas.DataFrame
            添加了MA支撑识别结果的数据框
        """
        if df.empty:
            logger.warning("输入数据为空")
            return df
        
        # 确保必要的列存在
        required_cols = ['close', 'low', 'high']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"MA支撑检测所需的列缺失: {missing_cols}")
            return df
        
        logger.info("开始识别移动平均线支撑")
        
        # 复制数据
        result_df = df.copy()
        
        # 初始化支撑列
        result_df['ma_support_level'] = 0  # 支撑强度（0-3）
        result_df['ma_support_type'] = ''  # 支撑类型
        result_df['ma_support_price'] = np.nan  # 支撑价格
        result_df['ma_support_signal'] = False  # MA支撑信号
        
        # 检查每个MA周期的支撑
        support_types = []
        support_prices = []
        
        for period in self.ma_periods:
            ma_col = f'ma_{period}'
            
            if ma_col not in result_df.columns:
                # 如果MA列不存在，计算它
                logger.info(f"计算{period}日移动平均线")
                result_df[ma_col] = talib.SMA(result_df['close'].values, timeperiod=period)
            
            # 计算价格与MA的距离
            price_to_ma_ratio = (result_df['close'] - result_df[ma_col]) / result_df[ma_col]
            low_to_ma_ratio = (result_df['low'] - result_df[ma_col]) / result_df[ma_col]
            
            # 检测支撑条件
            # 1. 价格接近MA（在阈值范围内）
            near_support = (abs(price_to_ma_ratio) <= self.ma_support_threshold) & (result_df['close'] >= result_df[ma_col])
            
            # 2. 价格触及MA（低点触及或穿过MA）
            touch_support = (low_to_ma_ratio <= self.ma_touch_threshold) & (result_df['close'] >= result_df[ma_col])
            
            # 3. 价格从MA下方回升
            below_to_above = (result_df['close'] > result_df[ma_col]) & (result_df['close'].shift(1) <= result_df[ma_col].shift(1))
            
            # 综合支撑条件
            ma_support = near_support | touch_support | below_to_above
            
            # 记录支撑信息
            result_df.loc[ma_support, f'ma{period}_support'] = True
            
            # 根据MA周期分配支撑强度
            if period == 10:
                support_strength = 1  # MA10支撑较弱
            elif period == 20:
                support_strength = 2  # MA20支撑中等
            else:  # period == 60
                support_strength = 3  # MA60支撑较强
            
            # 更新支撑强度和类型
            mask = ma_support & (result_df['ma_support_level'] < support_strength)
            result_df.loc[mask, 'ma_support_level'] = support_strength
            result_df.loc[mask, 'ma_support_type'] = f'MA{period}'
            result_df.loc[mask, 'ma_support_price'] = result_df.loc[mask, ma_col]
        
        # 生成MA支撑信号（任何MA支撑都产生信号）
        result_df['ma_support_signal'] = result_df['ma_support_level'] > 0
        
        # 统计支撑情况
        ma10_support_count = result_df.get('ma10_support', pd.Series([False]*len(result_df))).sum()
        ma20_support_count = result_df.get('ma20_support', pd.Series([False]*len(result_df))).sum()
        ma60_support_count = result_df.get('ma60_support', pd.Series([False]*len(result_df))).sum()
        
        logger.info(f"MA支撑识别完成 - MA10支撑: {ma10_support_count}次, "
                   f"MA20支撑: {ma20_support_count}次, MA60支撑: {ma60_support_count}次")
        
        return result_df
    
    def identify_platform_support(self, df):
        """
        平台/颈线支撑识别
        
        检测价格回调到前期平台上边界或突破颈线水平
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含OHLC数据的数据框
            
        Returns
        -------
        pandas.DataFrame
            添加了平台支撑识别结果的数据框
        """
        if df.empty:
            logger.warning("输入数据为空")
            return df
        
        # 确保必要的列存在
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"平台支撑识别所需的列缺失: {missing_cols}")
            return df
        
        logger.info("开始识别平台/颈线支撑")
        
        # 复制数据
        result_df = df.copy()
        
        # 初始化平台支撑列
        result_df['platform_support'] = False
        result_df['platform_support_level'] = np.nan
        result_df['neckline_support'] = False
        result_df['neckline_level'] = np.nan
        result_df['platform_neckline_signal'] = False  # 平台/颈线支撑信号
        
        # 使用滚动窗口识别平台
        for i in range(self.platform_lookback, len(result_df)):
            window = result_df.iloc[i-self.platform_lookback:i]
            
            # 计算窗口内的价格范围
            window_high = window['high'].max()
            window_low = window['low'].min()
            window_range = (window_high - window_low) / window_low
            
            # 检测平台条件
            if window_range <= self.platform_price_range:
                # 找到平台
                platform_high = window_high
                platform_low = window_low
                platform_mid = (platform_high + platform_low) / 2
                
                # 检查当前价格是否回调到平台支撑
                current_low = result_df.iloc[i]['low']
                current_close = result_df.iloc[i]['close']
                
                # 平台上边界支撑
                if (platform_high * 0.99 <= current_low <= platform_high * 1.01 and
                    current_close > platform_high * 0.995):
                    result_df.loc[result_df.index[i], 'platform_support'] = True
                    result_df.loc[result_df.index[i], 'platform_support_level'] = platform_high
                
                # 检测突破后的回踩（颈线支撑）
                # 查找之前是否有突破
                if i > 0:
                    prev_close = result_df.iloc[i-1]['close']
                    if (prev_close > platform_high and
                        current_low <= platform_high * (1 + self.neckline_threshold) and
                        current_close > platform_high):
                        result_df.loc[result_df.index[i], 'neckline_support'] = True
                        result_df.loc[result_df.index[i], 'neckline_level'] = platform_high
        
        # 生成平台/颈线支撑信号
        result_df['platform_neckline_signal'] = result_df['platform_support'] | result_df['neckline_support']
        
        # 统计支撑情况
        platform_support_count = result_df['platform_support'].sum()
        neckline_support_count = result_df['neckline_support'].sum()
        
        logger.info(f"平台/颈线支撑识别完成 - 平台支撑: {platform_support_count}次, "
                   f"颈线支撑: {neckline_support_count}次")
        
        return result_df
    
    def identify_bollinger_support(self, df):
        """
        布林带支撑分析（可选）
        
        识别价格回调到布林带中轨或下轨的情况
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含价格数据的数据框
            
        Returns
        -------
        pandas.DataFrame
            添加了布林带支撑识别结果的数据框
        """
        if df.empty:
            logger.warning("输入数据为空")
            return df
        
        # 确保必要的列存在
        required_cols = ['close', 'low']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"布林带支撑分析所需的列缺失: {missing_cols}")
            return df
        
        logger.info("开始识别布林带支撑")
        
        # 复制数据
        result_df = df.copy()
        
        # 检查布林带列是否存在，如果不存在则计算
        bb_cols = ['bb_upper', 'bb_middle', 'bb_lower']
        if not all(col in result_df.columns for col in bb_cols):
            logger.info("计算布林带指标")
            close_np = result_df['close'].values
            upper, middle, lower = talib.BBANDS(
                close_np,
                timeperiod=self.bb_period,
                nbdevup=self.bb_std_dev,
                nbdevdn=self.bb_std_dev,
                matype=0
            )
            result_df['bb_upper'] = upper
            result_df['bb_middle'] = middle
            result_df['bb_lower'] = lower
        
        # 初始化布林带支撑列
        result_df['bb_support_type'] = ''
        result_df['bb_support_level'] = np.nan
        result_df['bb_support_signal'] = False  # 布林带支撑信号
        
        # 计算价格与布林带的关系
        # 中轨支撑
        middle_distance = abs(result_df['low'] - result_df['bb_middle']) / result_df['bb_middle']
        middle_support = (middle_distance <= self.bb_support_threshold) & (result_df['close'] > result_df['bb_middle'])
        
        # 下轨支撑
        lower_distance = abs(result_df['low'] - result_df['bb_lower']) / result_df['bb_lower']
        lower_support = (lower_distance <= self.bb_support_threshold) & (result_df['close'] > result_df['bb_lower'])
        
        # 从下轨反弹
        lower_bounce = (result_df['low'] <= result_df['bb_lower']) & (result_df['close'] > result_df['bb_lower'])
        
        # 记录支撑类型
        result_df.loc[middle_support, 'bb_support_type'] = 'BB_Middle'
        result_df.loc[middle_support, 'bb_support_level'] = result_df.loc[middle_support, 'bb_middle']
        
        result_df.loc[lower_support | lower_bounce, 'bb_support_type'] = 'BB_Lower'
        result_df.loc[lower_support | lower_bounce, 'bb_support_level'] = result_df.loc[lower_support | lower_bounce, 'bb_lower']
        
        # 生成布林带支撑信号
        result_df['bb_support_signal'] = result_df['bb_support_type'] != ''
        
        # 统计支撑情况
        middle_support_count = middle_support.sum()
        lower_support_count = (lower_support | lower_bounce).sum()
        
        logger.info(f"布林带支撑识别完成 - 中轨支撑: {middle_support_count}次, "
                   f"下轨支撑: {lower_support_count}次")
        
        return result_df
    
    def analyze_volume_contraction(self, df):
        """
        成交量萎缩分析
        
        判断回调过程中成交量是否显著减少
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含价格和成交量数据的数据框
            
        Returns
        -------
        pandas.DataFrame
            添加了成交量分析结果的数据框
        """
        if df.empty:
            logger.warning("输入数据为空")
            return df
        
        # 确保必要的列存在
        required_cols = ['close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"成交量分析所需的列缺失: {missing_cols}")
            return df
        
        logger.info("开始分析成交量萎缩")
        
        # 复制数据
        result_df = df.copy()
        
        # 计算成交量均线
        result_df['volume_ma'] = result_df['volume'].rolling(window=self.volume_ma_period).mean()
        
        # 计算成交量比率
        result_df['volume_ratio'] = result_df['volume'] / result_df['volume_ma']
        
        # 识别价格回调
        result_df['price_pullback'] = result_df['close'] < result_df['close'].shift(1)
        
        # 计算连续回调天数
        result_df['pullback_days'] = 0
        pullback_count = 0
        for i in range(len(result_df)):
            if result_df.iloc[i]['price_pullback']:
                pullback_count += 1
            else:
                pullback_count = 0
            result_df.iloc[i, result_df.columns.get_loc('pullback_days')] = pullback_count
        
        # 判断成交量萎缩
        # 1. 当前成交量低于均量的设定比例
        volume_contraction = result_df['volume_ratio'] < self.volume_contraction_ratio
        
        # 2. 回调期间的平均成交量萎缩
        result_df['pullback_volume_contraction'] = False
        for i in range(self.volume_lookback, len(result_df)):
            if result_df.iloc[i]['pullback_days'] >= 2:  # 至少回调2天
                # 计算回调期间的平均成交量比率
                lookback_start = max(0, i - result_df.iloc[i]['pullback_days'] + 1)
                avg_volume_ratio = result_df.iloc[lookback_start:i+1]['volume_ratio'].mean()
                if avg_volume_ratio < self.volume_contraction_ratio:
                    result_df.iloc[i, result_df.columns.get_loc('pullback_volume_contraction')] = True
        
        # 成交量萎缩信号（独立信号）
        result_df['volume_contraction_signal'] = result_df['pullback_volume_contraction']
        
        # 统计分析结果
        volume_contraction_count = volume_contraction.sum()
        volume_signal_count = result_df['volume_contraction_signal'].sum()
        
        logger.info(f"成交量分析完成 - 成交量萎缩: {volume_contraction_count}次, "
                   f"成交量萎缩信号: {volume_signal_count}次")
        
        return result_df
    
    def identify_hammer_pattern(self, df, recent_days=None):
        """
        锤子线形态识别
        
        锤子线特征：
        1. 实体较小（开盘价与收盘价接近）
        2. 下影线长度至少是实体的2倍
        3. 上影线很短或没有
        4. 出现在下跌趋势中
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含OHLC数据的数据框
        recent_days : int, optional
            只分析最近N个交易日的数据，None表示分析所有数据
            
        Returns
        -------
        pandas.DataFrame
            添加了锤子线识别结果的数据框
        """
        if df.empty:
            logger.warning("输入数据为空")
            return df
        
        logger.info("开始识别锤子线形态")
        
        # 复制数据
        result_df = df.copy()
        
        # 初始化锤子线列
        result_df['hammer_pattern'] = False
        
        # 计算K线实体和影线
        result_df['body'] = abs(result_df['close'] - result_df['open'])
        result_df['upper_shadow'] = result_df['high'] - result_df[['open', 'close']].max(axis=1)
        result_df['lower_shadow'] = result_df[['open', 'close']].min(axis=1) - result_df['low']
        result_df['body_range'] = result_df['body'] / result_df['close']  # 实体占收盘价的比例
        
        # 计算短期趋势（5日）
        result_df['ma5'] = result_df['close'].rolling(window=5).mean()
        result_df['short_trend'] = result_df['close'] < result_df['ma5']  # 短期下跌趋势
        
        # 锤子线条件
        # 1. 实体较小（小于收盘价的2%）
        small_body = result_df['body_range'] < 0.02
        
        # 2. 下影线长度至少是实体的2倍
        long_lower_shadow = result_df['lower_shadow'] >= 2 * result_df['body']
        
        # 3. 上影线很短（小于实体的0.5倍）
        short_upper_shadow = result_df['upper_shadow'] <= 0.5 * result_df['body']
        
        # 4. 出现在下跌趋势中（可选条件，提高准确性）
        in_downtrend = result_df['short_trend']
        
        # 综合判断锤子线
        result_df['hammer_pattern'] = small_body & long_lower_shadow & short_upper_shadow & in_downtrend
        
        # 统计锤子线数量
        hammer_count = result_df['hammer_pattern'].sum()
        logger.info(f"锤子线形态识别完成 - 发现锤子线: {hammer_count}个")
        
        # 如果指定了recent_days，只在最近的数据中标记形态
        if recent_days is not None and len(result_df) > recent_days:
            # 将早期的形态标记清除
            result_df.iloc[:-recent_days, result_df.columns.get_loc('hammer_pattern')] = False
        
        return result_df
    
    def identify_doji_pattern(self, df, recent_days=None):
        """
        十字星形态识别
        
        十字星特征：
        1. 开盘价与收盘价极其接近（几乎相等）
        2. 上下影线相对较长
        3. 实体极小
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含OHLC数据的数据框
        recent_days : int, optional
            只分析最近N个交易日的数据，None表示分析所有数据
            
        Returns
        -------
        pandas.DataFrame
            添加了十字星识别结果的数据框
        """
        if df.empty:
            logger.warning("输入数据为空")
            return df
        
        logger.info("开始识别十字星形态")
        
        # 复制数据
        result_df = df.copy()
        
        # 初始化十字星列
        result_df['doji_pattern'] = False
        
        # 计算K线实体和影线（如果还没有计算）
        if 'body' not in result_df.columns:
            result_df['body'] = abs(result_df['close'] - result_df['open'])
            result_df['upper_shadow'] = result_df['high'] - result_df[['open', 'close']].max(axis=1)
            result_df['lower_shadow'] = result_df[['open', 'close']].min(axis=1) - result_df['low']
        
        # 计算实体占最高最低价差的比例
        result_df['body_to_range_ratio'] = result_df['body'] / (result_df['high'] - result_df['low'])
        
        # 十字星条件
        # 1. 实体极小（小于当日波动范围的10%）
        tiny_body = result_df['body_to_range_ratio'] < 0.1
        
        # 2. 开盘价与收盘价差异极小（小于收盘价的0.3%）
        close_open_close = (abs(result_df['close'] - result_df['open']) / result_df['close']) < 0.003
        
        # 3. 上下影线都存在且相对较长（至少是实体的2倍）
        has_shadows = (result_df['upper_shadow'] > 0) & (result_df['lower_shadow'] > 0)
        long_shadows = ((result_df['upper_shadow'] + result_df['lower_shadow']) >= 2 * result_df['body'])
        
        # 综合判断十字星
        result_df['doji_pattern'] = tiny_body & close_open_close & has_shadows & long_shadows
        
        # 统计十字星数量
        doji_count = result_df['doji_pattern'].sum()
        logger.info(f"十字星形态识别完成 - 发现十字星: {doji_count}个")
        
        # 如果指定了recent_days，只在最近的数据中标记形态
        if recent_days is not None and len(result_df) > recent_days:
            # 将早期的形态标记清除
            result_df.iloc[:-recent_days, result_df.columns.get_loc('doji_pattern')] = False
        
        return result_df
    
    def identify_morning_star_pattern(self, df, recent_days=None):
        """
        晨星形态识别
        
        晨星形态特征（三根K线组合）：
        1. 第一根：长阴线（下跌）
        2. 第二根：小实体（十字星或小阳/小阴线），向下跳空
        3. 第三根：长阳线（上涨），向上跳空
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含OHLC数据的数据框
        recent_days : int, optional
            只分析最近N个交易日的数据，None表示分析所有数据
            
        Returns
        -------
        pandas.DataFrame
            添加了晨星形态识别结果的数据框
        """
        if df.empty or len(df) < 3:
            logger.warning("输入数据不足以识别晨星形态（需要至少3根K线）")
            return df
        
        logger.info("开始识别晨星形态")
        
        # 复制数据
        result_df = df.copy()
        
        # 初始化晨星形态列
        result_df['morning_star_pattern'] = False
        
        # 计算K线实体（如果还没有计算）
        if 'body' not in result_df.columns:
            result_df['body'] = abs(result_df['close'] - result_df['open'])
        
        # 计算实体方向（阳线为正，阴线为负）
        result_df['body_direction'] = result_df['close'] - result_df['open']
        
        # 计算平均实体大小（用于判断长短）
        avg_body = result_df['body'].rolling(window=20, min_periods=10).mean()
        
        # 确定分析范围
        start_idx = 2  # 至少需要3根K线
        if recent_days is not None and len(result_df) > recent_days:
            start_idx = max(2, len(result_df) - recent_days)
        
        # 从第3根K线开始检查（因为需要前两根）
        for i in range(start_idx, len(result_df)):
            # 获取三根K线
            first = result_df.iloc[i-2]
            second = result_df.iloc[i-1]
            third = result_df.iloc[i]
            
            # 获取当前平均实体大小
            current_avg_body = avg_body.iloc[i] if not pd.isna(avg_body.iloc[i]) else result_df['body'].mean()
            
            # 第一根K线条件：长阴线
            first_is_long_bearish = (first['body_direction'] < 0) and (first['body'] > 1.5 * current_avg_body)
            
            # 第二根K线条件：小实体（可以是十字星）
            second_is_small = second['body'] < 0.5 * current_avg_body
            
            # 第二根K线向下跳空（开盘价低于第一根的收盘价）
            second_gap_down = second['open'] < first['close']
            
            # 第三根K线条件：长阳线
            third_is_long_bullish = (third['body_direction'] > 0) and (third['body'] > 1.5 * current_avg_body)
            
            # 第三根K线向上跳空或至少高开（开盘价高于第二根的收盘价）
            third_gap_up = third['open'] > second['close']
            
            # 第三根K线收盘价最好能超过第一根K线的中点
            first_midpoint = (first['open'] + first['close']) / 2
            third_closes_above_midpoint = third['close'] > first_midpoint
            
            # 综合判断晨星形态
            if (first_is_long_bearish and second_is_small and second_gap_down and
                third_is_long_bullish and third_gap_up and third_closes_above_midpoint):
                result_df.iloc[i, result_df.columns.get_loc('morning_star_pattern')] = True
        
        # 统计晨星形态数量
        morning_star_count = result_df['morning_star_pattern'].sum()
        logger.info(f"晨星形态识别完成 - 发现晨星形态: {morning_star_count}个")
        
        return result_df
    
    def identify_bullish_engulfing_pattern(self, df, recent_days=None):
        """
        看涨吞没形态识别
        
        看涨吞没形态特征（两根K线组合）：
        1. 第一根：阴线
        2. 第二根：阳线，实体完全包含（吞没）第一根阴线的实体
        3. 通常出现在下跌趋势末期
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含OHLC数据的数据框
        recent_days : int, optional
            只分析最近N个交易日的数据，None表示分析所有数据
            
        Returns
        -------
        pandas.DataFrame
            添加了看涨吞没形态识别结果的数据框
        """
        if df.empty or len(df) < 2:
            logger.warning("输入数据不足以识别看涨吞没形态（需要至少2根K线）")
            return df
        
        logger.info("开始识别看涨吞没形态")
        
        # 复制数据
        result_df = df.copy()
        
        # 初始化看涨吞没形态列
        result_df['bullish_engulfing_pattern'] = False
        
        # 计算K线实体方向（如果还没有计算）
        if 'body_direction' not in result_df.columns:
            result_df['body_direction'] = result_df['close'] - result_df['open']
        
        # 计算短期趋势（用于确认下跌趋势）
        if 'ma5' not in result_df.columns:
            result_df['ma5'] = result_df['close'].rolling(window=5).mean()
        result_df['in_downtrend'] = result_df['close'] < result_df['ma5']
        
        # 确定分析范围
        start_idx = 1  # 至少需要2根K线
        if recent_days is not None and len(result_df) > recent_days:
            start_idx = max(1, len(result_df) - recent_days)
        
        # 从第2根K线开始检查
        for i in range(start_idx, len(result_df)):
            # 获取两根K线
            first = result_df.iloc[i-1]
            second = result_df.iloc[i]
            
            # 第一根K线条件：阴线
            first_is_bearish = first['body_direction'] < 0
            
            # 第二根K线条件：阳线
            second_is_bullish = second['body_direction'] > 0
            
            # 吞没条件：第二根阳线的实体完全包含第一根阴线的实体
            # 即：第二根的开盘价低于或等于第一根的收盘价，且第二根的收盘价高于或等于第一根的开盘价
            engulfing_condition = (second['open'] <= first['close']) and (second['close'] >= first['open'])
            
            # 可选条件：出现在下跌趋势中（提高准确性）
            in_downtrend = result_df.iloc[i]['in_downtrend']
            
            # 综合判断看涨吞没形态
            if first_is_bearish and second_is_bullish and engulfing_condition and in_downtrend:
                result_df.iloc[i, result_df.columns.get_loc('bullish_engulfing_pattern')] = True
        
        # 统计看涨吞没形态数量
        bullish_engulfing_count = result_df['bullish_engulfing_pattern'].sum()
        logger.info(f"看涨吞没形态识别完成 - 发现看涨吞没形态: {bullish_engulfing_count}个")
        
        return result_df
    
    def identify_candlestick_patterns(self, df, recent_days=None):
        """
        识别所有K线形态
        
        综合识别锤子线、十字星、晨星和看涨吞没形态
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含OHLC数据的数据框
        recent_days : int, optional
            只分析最近N个交易日的数据，None表示分析所有数据
            默认为None（分析所有数据）
            
        Returns
        -------
        pandas.DataFrame
            添加了所有K线形态识别结果的数据框
        """
        if df.empty:
            logger.warning("输入数据为空")
            return df
        
        logger.info("开始识别K线形态")
        
        # 识别各种形态
        df = self.identify_hammer_pattern(df, recent_days)
        df = self.identify_doji_pattern(df, recent_days)
        df = self.identify_morning_star_pattern(df, recent_days)
        df = self.identify_bullish_engulfing_pattern(df, recent_days)
        
        # 添加综合K线形态信号
        df['candlestick_pattern_signal'] = (
            df.get('hammer_pattern', False) |
            df.get('doji_pattern', False) |
            df.get('morning_star_pattern', False) |
            df.get('bullish_engulfing_pattern', False)
        )
        
        # 生成K线形态类型列表
        df['candlestick_patterns'] = ''
        for i in range(len(df)):
            patterns = []
            if df.iloc[i].get('hammer_pattern', False):
                patterns.append('锤子线')
            if df.iloc[i].get('doji_pattern', False):
                patterns.append('十字星')
            if df.iloc[i].get('morning_star_pattern', False):
                patterns.append('晨星')
            if df.iloc[i].get('bullish_engulfing_pattern', False):
                patterns.append('看涨吞没')
            
            df.iloc[i, df.columns.get_loc('candlestick_patterns')] = ','.join(patterns)
        
        # 统计各形态数量
        total_patterns = df['candlestick_pattern_signal'].sum()
        logger.info(f"K线形态识别完成 - 共发现看涨反转形态: {total_patterns}个")
        
        return df
    
    def generate_support_summary(self, df):
        """
        生成支撑位汇总信息
        
        汇总所有支撑位信号，提供综合的支撑位信息
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含所有分析结果的数据框
            
        Returns
        -------
        pandas.DataFrame
            添加了支撑位汇总信息的数据框
        """
        if df.empty:
            logger.warning("输入数据为空")
            return df
        
        logger.info("生成支撑位汇总信息")
        
        # 复制数据
        result_df = df.copy()
        
        # 初始化汇总列
        result_df['support_count'] = 0  # 支撑位数量
        result_df['support_types'] = ''  # 支撑类型列表
        result_df['any_support_signal'] = False  # 任意支撑信号
        
        # 统计支撑位
        for i in range(len(result_df)):
            support_types = []
            support_count = 0
            
            # MA支撑
            if 'ma_support_signal' in result_df.columns and result_df.iloc[i]['ma_support_signal']:
                support_types.append(result_df.iloc[i]['ma_support_type'])
                support_count += 1
            
            # 平台/颈线支撑
            if 'platform_support' in result_df.columns and result_df.iloc[i]['platform_support']:
                support_types.append('平台')
                support_count += 1
            
            if 'neckline_support' in result_df.columns and result_df.iloc[i]['neckline_support']:
                support_types.append('颈线')
                support_count += 1
            
            # 布林带支撑
            if 'bb_support_signal' in result_df.columns and result_df.iloc[i]['bb_support_signal']:
                support_types.append(result_df.iloc[i]['bb_support_type'])
                support_count += 1
            
            # K线形态
            if 'candlestick_patterns' in result_df.columns and result_df.iloc[i]['candlestick_patterns']:
                patterns = result_df.iloc[i]['candlestick_patterns']
                if patterns:
                    support_types.append(f"K线形态({patterns})")
                    support_count += 1
            
            # 更新汇总信息
            result_df.iloc[i, result_df.columns.get_loc('support_count')] = support_count
            result_df.iloc[i, result_df.columns.get_loc('support_types')] = ','.join(support_types)
            result_df.iloc[i, result_df.columns.get_loc('any_support_signal')] = support_count > 0
        
        # 统计信号
        support_signal_count = result_df['any_support_signal'].sum()
        
        logger.info(f"支撑位汇总完成 - 出现支撑信号的数据点: {support_signal_count}个")
        
        return result_df
    
    def analyze(self, df):
        """
        执行完整的支撑位识别和成交量分析
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含OHLCV数据的数据框
            
        Returns
        -------
        pandas.DataFrame
            包含所有分析结果的数据框
        """
        if df.empty:
            logger.warning("输入数据为空")
            return df
        
        logger.info("开始执行支撑位识别和成交量分析")
        
        # 1. 移动平均线支撑检测
        df = self.identify_ma_support(df)
        
        # 2. 平台/颈线支撑识别
        df = self.identify_platform_support(df)
        
        # 3. 布林带支撑分析
        df = self.identify_bollinger_support(df)
        
        # 4. 成交量萎缩分析
        df = self.analyze_volume_contraction(df)
        
        # 5. K线形态识别
        df = self.identify_candlestick_patterns(df)
        
        # 6. 生成支撑位汇总
        df = self.generate_support_summary(df)
        
        # 添加分析时间戳
        df['analysis_timestamp'] = datetime.now()
        
        logger.info("支撑位识别和成交量分析完成")
        
        return df
    
    def has_hammer_pattern(self, df, lookback=1):
        """
        检查最近是否出现锤子线形态
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含形态识别结果的数据框
        lookback : int, default 1
            回看的K线数量
            
        Returns
        -------
        bool
            是否存在锤子线形态
        """
        if 'hammer_pattern' not in df.columns or df.empty:
            return False
        
        return df['hammer_pattern'].iloc[-lookback:].any()
    
    def has_doji_pattern(self, df, lookback=1):
        """
        检查最近是否出现十字星形态
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含形态识别结果的数据框
        lookback : int, default 1
            回看的K线数量
            
        Returns
        -------
        bool
            是否存在十字星形态
        """
        if 'doji_pattern' not in df.columns or df.empty:
            return False
        
        return df['doji_pattern'].iloc[-lookback:].any()
    
    def has_morning_star_pattern(self, df, lookback=1):
        """
        检查最近是否出现晨星形态
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含形态识别结果的数据框
        lookback : int, default 1
            回看的K线数量
            
        Returns
        -------
        bool
            是否存在晨星形态
        """
        if 'morning_star_pattern' not in df.columns or df.empty:
            return False
        
        return df['morning_star_pattern'].iloc[-lookback:].any()
    
    def has_bullish_engulfing_pattern(self, df, lookback=1):
        """
        检查最近是否出现看涨吞没形态
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含形态识别结果的数据框
        lookback : int, default 1
            回看的K线数量
            
        Returns
        -------
        bool
            是否存在看涨吞没形态
        """
        if 'bullish_engulfing_pattern' not in df.columns or df.empty:
            return False
        
        return df['bullish_engulfing_pattern'].iloc[-lookback:].any()
    
    def has_any_bullish_pattern(self, df, lookback=1):
        """
        检查最近是否出现任何看涨反转形态
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含形态识别结果的数据框
        lookback : int, default 1
            回看的K线数量
            
        Returns
        -------
        bool
            是否存在任何看涨反转形态
        """
        return (self.has_hammer_pattern(df, lookback) or
                self.has_doji_pattern(df, lookback) or
                self.has_morning_star_pattern(df, lookback) or
                self.has_bullish_engulfing_pattern(df, lookback))
    
    def get_latest_patterns(self, df):
        """
        获取最新的K线形态
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含形态识别结果的数据框
            
        Returns
        -------
        dict
            包含各种形态是否存在的字典
        """
        if df.empty:
            return {
                'hammer': False,
                'doji': False,
                'morning_star': False,
                'bullish_engulfing': False,
                'any_pattern': False,
                'pattern_names': []
            }
        
        latest_row = df.iloc[-1]
        patterns = []
        
        hammer = latest_row.get('hammer_pattern', False)
        doji = latest_row.get('doji_pattern', False)
        morning_star = latest_row.get('morning_star_pattern', False)
        bullish_engulfing = latest_row.get('bullish_engulfing_pattern', False)
        
        if hammer:
            patterns.append('锤子线')
        if doji:
            patterns.append('十字星')
        if morning_star:
            patterns.append('晨星')
        if bullish_engulfing:
            patterns.append('看涨吞没')
        
        return {
            'hammer': hammer,
            'doji': doji,
            'morning_star': morning_star,
            'bullish_engulfing': bullish_engulfing,
            'any_pattern': bool(patterns),
            'pattern_names': patterns
        }
    
    def get_pattern_statistics(self, df):
        """
        获取K线形态统计信息
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含形态识别结果的数据框
            
        Returns
        -------
        dict
            包含各种形态统计信息的字典
        """
        if df.empty:
            return {
                'total_patterns': 0,
                'hammer_count': 0,
                'doji_count': 0,
                'morning_star_count': 0,
                'bullish_engulfing_count': 0,
                'pattern_distribution': {}
            }
        
        hammer_count = df.get('hammer_pattern', pd.Series([False]*len(df))).sum()
        doji_count = df.get('doji_pattern', pd.Series([False]*len(df))).sum()
        morning_star_count = df.get('morning_star_pattern', pd.Series([False]*len(df))).sum()
        bullish_engulfing_count = df.get('bullish_engulfing_pattern', pd.Series([False]*len(df))).sum()
        
        total_patterns = df.get('candlestick_pattern_signal', pd.Series([False]*len(df))).sum()
        
        return {
            'total_patterns': total_patterns,
            'hammer_count': hammer_count,
            'doji_count': doji_count,
            'morning_star_count': morning_star_count,
            'bullish_engulfing_count': bullish_engulfing_count,
            'pattern_distribution': {
                '锤子线': hammer_count,
                '十字星': doji_count,
                '晨星': morning_star_count,
                '看涨吞没': bullish_engulfing_count
            }
        }
    
    def analyze_rsi_pullback_conditions(self, df):
        """
        分析RSI指标的回调条件，在DataFrame中添加相关布尔值列
        
        此函数检查多种RSI相关的技术条件，直接在df中添加布尔值列。
        主要评估场景：
        1. 从超买区域回调到中性区间（强势回调）
        2. 从超卖区域开始的动能反转
        3. 避免在长期超买状态下入场
        
        Parameters
        ----------
        df : pd.DataFrame
            包含OHLCV数据的DataFrame，必须包含'close'列
            函数会在此DataFrame上添加以下列：
            - rsi: RSI值
            - rsi_pullback_from_overbought: 从超买区域的强势回调
            - rsi_oversold_reversal: 超卖动能反转
            - rsi_avoid_prolonged_overbought: 应避免（长期超买）
            - rsi_pullback_favorable: 综合评估是否适合回调入场
            
        Returns
        -------
        pd.DataFrame
            添加了RSI分析列的DataFrame
        """
        try:
            # 检查数据有效性
            if df is None or df.empty:
                logger.warning("RSI分析：输入数据为空")
                return df
            
            # 检查必需的列
            if 'close' not in df.columns:
                logger.error("RSI分析：缺少必需的'close'列")
                return df
            
            # 检查数据量是否足够
            if len(df) < self.rsi_period + 10:
                logger.warning(f"RSI分析：数据量不足，需要至少{self.rsi_period + 10}个数据点")
                # 添加空列
                df['rsi'] = np.nan
                df['rsi_pullback_from_overbought'] = False
                df['rsi_oversold_reversal'] = False
                df['rsi_avoid_prolonged_overbought'] = False
                df['rsi_pullback_favorable'] = False
                return df
            
            # 计算RSI
            df['rsi'] = talib.RSI(df['close'].values, timeperiod=self.rsi_period)
            
            # 初始化布尔列
            df['rsi_pullback_from_overbought'] = False
            df['rsi_oversold_reversal'] = False
            df['rsi_avoid_prolonged_overbought'] = False
            df['rsi_pullback_favorable'] = False
            
            # 需要足够的历史数据进行分析
            min_lookback = 20
            
            # 对每一行进行分析（从有足够历史数据的行开始）
            for i in range(min_lookback, len(df)):
                if pd.isna(df['rsi'].iloc[i]):
                    continue
                
                current_rsi = df['rsi'].iloc[i]
                # 获取历史RSI数据
                recent_rsi = df['rsi'].iloc[max(0, i-20):i+1].dropna()
                
                if len(recent_rsi) < 10:
                    continue
                
                # 场景1：从超买区域的强势回调
                max_rsi_recent = recent_rsi.iloc[-10:].max()
                if max_rsi_recent > self.rsi_overbought_threshold:
                    if 45 <= current_rsi <= 55:
                        # 确认是下降趋势
                        if len(recent_rsi) >= 10:
                            rsi_ma5 = recent_rsi.iloc[-5:].mean()
                            rsi_ma10 = recent_rsi.iloc[-10:].mean()
                            if rsi_ma5 < rsi_ma10:
                                df.loc[df.index[i], 'rsi_pullback_from_overbought'] = True
                
                # 场景2：超卖动能反转
                min_rsi_recent = recent_rsi.iloc[-10:].min()
                if min_rsi_recent < self.rsi_oversold_threshold:
                    if self.rsi_oversold_threshold <= current_rsi <= 40:
                        # 确认上升趋势（连续3个周期上升）
                        if len(recent_rsi) >= 3:
                            if (recent_rsi.iloc[-1] > recent_rsi.iloc[-2] and 
                                recent_rsi.iloc[-2] > recent_rsi.iloc[-3]):
                                df.loc[df.index[i], 'rsi_oversold_reversal'] = True
                
                # 场景3：避免长期超买
                if current_rsi > 80:
                    # 检查最近5个周期是否都在超买区
                    if len(recent_rsi) >= 5:
                        if (recent_rsi.iloc[-5:] > self.rsi_overbought_threshold).all():
                            df.loc[df.index[i], 'rsi_avoid_prolonged_overbought'] = True
                
                # 综合评估
                if ((df.loc[df.index[i], 'rsi_pullback_from_overbought'] or 
                     df.loc[df.index[i], 'rsi_oversold_reversal']) and 
                    not df.loc[df.index[i], 'rsi_avoid_prolonged_overbought']):
                    df.loc[df.index[i], 'rsi_pullback_favorable'] = True
            
            logger.info(f"RSI分析完成，添加了5个分析列")
            return df
            
        except Exception as e:
            logger.error(f"RSI分析过程中出错: {str(e)}")
            # 确保添加了列（即使是空的）
            for col in ['rsi', 'rsi_pullback_from_overbought', 'rsi_oversold_reversal', 
                       'rsi_avoid_prolonged_overbought', 'rsi_pullback_favorable']:
                if col not in df.columns:
                    df[col] = False if col != 'rsi' else np.nan
            return df
    
    def analyze_mfi_pullback_conditions(self, df):
        """
        分析MFI指标的回调条件，在DataFrame中添加相关布尔值列
        
        此函数检查多种MFI相关的技术条件，直接在df中添加布尔值列。
        主要评估场景：
        1. 从超卖区域恢复并显示上升动能
        2. 价格稳定期间的资金流入模式
        3. 当前MFI读数是否支持回调入场时机
        
        Parameters
        ----------
        df : pd.DataFrame
            包含OHLCV数据的DataFrame，必须包含'high', 'low', 'close', 'volume'列
            函数会在此DataFrame上添加以下列：
            - mfi: MFI值
            - mfi_recovery_from_oversold: 从超卖区恢复
            - mfi_capital_inflow_stability: 价格稳定期资金流入
            - mfi_supports_pullback: MFI支持回调入场
            - mfi_overall_favorable: MFI综合评估有利
            
        Returns
        -------
        pd.DataFrame
            添加了MFI分析列的DataFrame
        """
        try:
            # 检查数据有效性
            if df is None or df.empty:
                logger.warning("MFI分析：输入数据为空")
                return df
            
            # 检查必需的列
            required_cols = ['high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"MFI分析：缺少必需的列: {missing_cols}")
                # 添加空列
                df['mfi'] = np.nan
                df['mfi_recovery_from_oversold'] = False
                df['mfi_capital_inflow_stability'] = False
                df['mfi_supports_pullback'] = False
                df['mfi_overall_favorable'] = False
                return df
            
            # 检查数据量是否足够
            if len(df) < self.mfi_period + 10:
                logger.warning(f"MFI分析：数据量不足，需要至少{self.mfi_period + 10}个数据点")
                # 添加空列
                df['mfi'] = np.nan
                df['mfi_recovery_from_oversold'] = False
                df['mfi_capital_inflow_stability'] = False
                df['mfi_supports_pullback'] = False
                df['mfi_overall_favorable'] = False
                return df
            
            # 计算MFI
            df['mfi'] = talib.MFI(df['high'].values, df['low'].values,
                                 df['close'].values, df['volume'].values,
                                 timeperiod=self.mfi_period)
            
            # 初始化布尔列
            df['mfi_recovery_from_oversold'] = False
            df['mfi_capital_inflow_stability'] = False
            df['mfi_supports_pullback'] = False
            df['mfi_overall_favorable'] = False
            
            # 需要足够的历史数据进行分析
            min_lookback = 20
            
            # 对每一行进行分析（从有足够历史数据的行开始）
            for i in range(min_lookback, len(df)):
                if pd.isna(df['mfi'].iloc[i]):
                    continue
                
                current_mfi = df['mfi'].iloc[i]
                # 获取历史MFI数据
                recent_mfi = df['mfi'].iloc[max(0, i-20):i+1].dropna()
                
                if len(recent_mfi) < 10:
                    continue
                
                # 场景1：从超卖区恢复并显示上升动能
                min_mfi_recent = recent_mfi.iloc[-10:].min()
                if min_mfi_recent < self.mfi_oversold_threshold:
                    # 检查是否已经恢复
                    if current_mfi > self.mfi_recovery_threshold:
                        # 检查最近几期的MFI动能
                        if len(recent_mfi) >= self.mfi_momentum_lookback:
                            recent_momentum = recent_mfi.iloc[-self.mfi_momentum_lookback:]
                            if len(recent_momentum) >= 2:
                                momentum_slope = np.polyfit(range(len(recent_momentum)),
                                                          recent_momentum.values, 1)[0]
                                if momentum_slope > 0:
                                    df.loc[df.index[i], 'mfi_recovery_from_oversold'] = True
                
                # 场景2：价格稳定期间的资金流入
                if i >= 10:
                    # 计算价格波动率
                    recent_prices = df['close'].iloc[i-9:i+1].values
                    price_returns = np.diff(recent_prices) / recent_prices[:-1]
                    price_volatility = np.std(price_returns)
                    
                    # 判断价格是否稳定
                    if price_volatility < self.price_stability_threshold:
                        # 检查MFI是否显示资金流入（高于50且上升）
                        if current_mfi > self.mfi_inflow_threshold:
                            if len(recent_mfi) >= 5:
                                recent_mfi_trend = recent_mfi.iloc[-5:]
                                mfi_trend_slope = np.polyfit(range(len(recent_mfi_trend)),
                                                           recent_mfi_trend.values, 1)[0]
                                if mfi_trend_slope > 0:
                                    df.loc[df.index[i], 'mfi_capital_inflow_stability'] = True
                
                # 场景3：判断当前MFI是否支持回调入场
                if self.mfi_recovery_threshold <= current_mfi <= self.mfi_overbought_threshold:
                    # 检查短期动能
                    if len(recent_mfi) >= 3:
                        short_term_trend = recent_mfi.iloc[-3:]
                        short_term_slope = np.polyfit(range(3), short_term_trend.values, 1)[0]
                        if short_term_slope >= 0:  # 非负斜率表示稳定或上升
                            df.loc[df.index[i], 'mfi_supports_pullback'] = True
                
                # 综合评估
                if (df.loc[df.index[i], 'mfi_recovery_from_oversold'] or
                    df.loc[df.index[i], 'mfi_capital_inflow_stability'] or
                    df.loc[df.index[i], 'mfi_supports_pullback']):
                    df.loc[df.index[i], 'mfi_overall_favorable'] = True
            
            logger.info(f"MFI分析完成，添加了5个分析列")
            return df
            
        except Exception as e:
            logger.error(f"MFI分析过程中出错: {str(e)}")
            # 确保添加了列（即使是空的）
            for col in ['mfi', 'mfi_recovery_from_oversold', 'mfi_capital_inflow_stability',
                       'mfi_supports_pullback', 'mfi_overall_favorable']:
                if col not in df.columns:
                    df[col] = False if col != 'mfi' else np.nan
            return df


# 测试代码
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # 创建分析器实例
    analyzer = SupportPullbackAnalyzer()
    
    print("=== 支撑位识别和成交量分析模块测试 ===\n")
    
    # 创建测试数据
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # 生成模拟OHLCV数据
    np.random.seed(42)
    base_price = 100
    returns = np.random.randn(100) * 0.02  # 2%的日收益率标准差
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    # 生成OHLCV数据
    df = pd.DataFrame({
        'date': dates,
        'open': close_prices * (1 + np.random.randn(100) * 0.005),
        'high': close_prices * (1 + np.abs(np.random.randn(100)) * 0.01),
        'low': close_prices * (1 - np.abs(np.random.randn(100)) * 0.01),
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    
    print("1. 测试数据生成完成")
    print(f"   数据范围: {df['date'].min()} 到 {df['date'].max()}")
    print(f"   数据点数: {len(df)}")
    print(f"   价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}\n")
    
    # 测试RSI分析
    print("2. 测试RSI回调条件分析")
    df_with_rsi = analyzer.analyze_rsi_pullback_conditions(df.copy())
    
    print("   RSI分析结果:")
    print(f"   - 添加了RSI列: {'rsi' in df_with_rsi.columns}")
    if 'rsi' in df_with_rsi.columns:
        latest_rsi = df_with_rsi['rsi'].iloc[-1]
        if not pd.isna(latest_rsi):
            print(f"   - 当前RSI: {latest_rsi:.2f}")
        else:
            print("   - 当前RSI: N/A")
    
    # 统计布尔结果
    pullback_count = df_with_rsi['rsi_pullback_from_overbought'].sum()
    reversal_count = df_with_rsi['rsi_oversold_reversal'].sum()
    avoid_count = df_with_rsi['rsi_avoid_prolonged_overbought'].sum()
    favorable_count = df_with_rsi['rsi_pullback_favorable'].sum()
    
    print(f"   - 从超买回调到中性: {pullback_count}次")
    print(f"   - 超卖动能反转: {reversal_count}次")
    print(f"   - 应避免（长期超买）: {avoid_count}次")
    print(f"   - 适合回调入场: {favorable_count}次\n")
    
    # 测试MFI分析
    print("3. 测试MFI回调条件分析")
    df_with_mfi = analyzer.analyze_mfi_pullback_conditions(df.copy())
    
    print("   MFI分析结果:")
    print(f"   - 添加了MFI列: {'mfi' in df_with_mfi.columns}")
    if 'mfi' in df_with_mfi.columns:
        latest_mfi = df_with_mfi['mfi'].iloc[-1]
        if not pd.isna(latest_mfi):
            print(f"   - 当前MFI: {latest_mfi:.2f}")
        else:
            print("   - 当前MFI: N/A")
    
    # 统计布尔结果
    recovery_count = df_with_mfi['mfi_recovery_from_oversold'].sum()
    inflow_count = df_with_mfi['mfi_capital_inflow_stability'].sum()
    support_count = df_with_mfi['mfi_supports_pullback'].sum()
    overall_count = df_with_mfi['mfi_overall_favorable'].sum()
    
    print(f"   - 从超卖恢复: {recovery_count}次")
    print(f"   - 价格稳定期资金流入: {inflow_count}次")
    print(f"   - MFI支持回调入场: {support_count}次")
    print(f"   - MFI综合评估有利: {overall_count}次\n")
    
    # 测试完整的分析流程
    print("4. 测试完整分析流程（包括原有功能）")
    full_analysis_df = analyzer.analyze(df)
    
    # 同时应用RSI和MFI分析
    full_analysis_df = analyzer.analyze_rsi_pullback_conditions(full_analysis_df)
    full_analysis_df = analyzer.analyze_mfi_pullback_conditions(full_analysis_df)
    
    # 显示最后几行的分析结果
    print("\n   最近5个交易日的综合分析结果:")
    display_cols = ['date', 'close']
    
    # 检查并添加存在的列
    optional_cols = ['rsi', 'mfi', 'rsi_pullback_favorable', 'mfi_overall_favorable',
                    'support_count', 'support_types', 'volume_contraction_signal',
                    'candlestick_pattern_signal', 'any_support_signal']
    for col in optional_cols:
        if col in full_analysis_df.columns:
            display_cols.append(col)
    
    if len(display_cols) > 2:
        # 只显示有限的列以保持可读性
        key_cols = ['date', 'close', 'rsi', 'mfi', 'rsi_pullback_favorable',
                   'mfi_overall_favorable', 'any_support_signal']
        display_cols = [col for col in key_cols if col in full_analysis_df.columns]
        print(full_analysis_df[display_cols].tail())
    else:
        print("   注意：某些分析列可能未生成")
    
    # 综合评估
    print("\n5. 综合评估总结:")
    
    # 获取最新数据
    latest_row = full_analysis_df.iloc[-1]
    
    rsi_favorable = latest_row.get('rsi_pullback_favorable', False)
    mfi_favorable = latest_row.get('mfi_overall_favorable', False)
    support_signal = latest_row.get('any_support_signal', False)
    volume_signal = latest_row.get('volume_contraction_signal', False)
    
    print(f"   - RSI指标建议入场: {rsi_favorable}")
    print(f"   - MFI指标显示有利: {mfi_favorable}")
    print(f"   - 存在支撑信号: {support_signal}")
    print(f"   - 成交量萎缩: {volume_signal}")
    
    # 策略建议
    print("\n6. 回调入场策略建议:")
    
    if rsi_favorable and mfi_favorable:
        print("   ✓ RSI和MFI指标均支持回调入场")
        print("   建议：可以考虑建立仓位，但需结合其他技术指标确认")
    elif rsi_favorable:
        print("   ✓ RSI指标支持回调入场")
        print("   △ MFI指标未给出明确信号")
        print("   建议：谨慎观察，等待MFI确认")
    elif mfi_favorable:
        print("   △ RSI指标未给出明确信号")
        print("   ✓ MFI指标显示资金流入")
        print("   建议：继续观察RSI走势")
    else:
        print("   ✗ RSI和MFI指标均不支持入场")
        print("   建议：继续等待更好的入场时机")
    
    # 测试边界情况
    print("\n7. 测试边界情况")
    
    # 测试数据不足的情况
    small_df = df.head(10)
    df_small_rsi = analyzer.analyze_rsi_pullback_conditions(small_df)
    print(f"   - 数据不足时RSI分析: 添加了列但值为空")
    print(f"     RSI列存在: {'rsi' in df_small_rsi.columns}")
    print(f"     RSI值数量: {df_small_rsi['rsi'].notna().sum()}")
    
    # 测试空数据
    empty_df = pd.DataFrame()
    df_empty_mfi = analyzer.analyze_mfi_pullback_conditions(empty_df)
    print(f"   - 空数据时MFI分析: 返回原DataFrame")
    print(f"     返回空DataFrame: {df_empty_mfi.empty}")
    
    # 测试缺少必要列的情况
    incomplete_df = df[['date', 'close']].copy()
    df_incomplete_mfi = analyzer.analyze_mfi_pullback_conditions(incomplete_df)
    print(f"   - 缺少列时MFI分析: 添加了空列")
    print(f"     MFI列存在: {'mfi' in df_incomplete_mfi.columns}")
    print(f"     MFI值数量: {df_incomplete_mfi['mfi'].notna().sum()}")
    
    print("\n=== 测试完成 ===")
    print("\n功能总结:")
    print("1. RSI分析功能：")
    print("   - 检测从超买区域(>70)回调到中性区间(45-55)的强势回调")
    print("   - 识别从超卖区域(30-40)的动能反转")
    print("   - 警示长期超买状态(>80)")
    print("   - 直接在DataFrame上添加布尔值列")
    print("\n2. MFI分析功能：")
    print("   - 检测从超卖区(<30)的恢复和上升动能")
    print("   - 识别价格稳定期的资金流入模式")
    print("   - 评估当前MFI水平是否支持入场")
    print("   - 直接在DataFrame上添加布尔值列")
    print("\n3. 两个函数都直接修改传入的DataFrame，添加分析结果列")