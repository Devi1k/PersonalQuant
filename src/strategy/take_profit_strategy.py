#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
止盈策略模块

根据预设的规则（如目标收益、移动止盈、趋势衰竭信号等）判断是否应该执行止盈操作。
"""

import pandas as pd
import talib
import logging
from typing import Dict, Any, Tuple
from datetime import datetime
import os
import sys
from pathlib import Path

# -- 项目路径设置 --
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

# -- 日志配置 --
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"take_profit_strategy_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TakeProfitStrategy:
    """
    止盈策略类，用于评估多种止盈条件。
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化止盈策略管理器。

        Args:
            config (Dict[str, Any], optional): 止盈策略的配置参数。默认为 None。
        """
        self.config = config or {}
        logger.info(f"止盈策略初始化，配置: {self.config}")
        
        # -- 目标止盈 --
        tp_config = self.config.get('target_profit', {})
        self.tp_active = tp_config.get('active', False)
        self.tp_percentage = tp_config.get('percentage', 20.0)

        # -- 移动止盈 --
        trailing_config = self.config.get('trailing_profit', {})
        self.trailing_active = trailing_config.get('active', False)
        
        # 百分比移动止盈
        pct_trailing_config = trailing_config.get('percentage_stop', {})
        self.pct_trailing_active = pct_trailing_config.get('active', False)
        self.trailing_stop_pct = pct_trailing_config.get('percentage', 10.0)

        # MA 移动止盈
        ma_trailing_config = trailing_config.get('ma_stop', {})
        self.ma_trailing_active = ma_trailing_config.get('active', False)
        self.ma_trailing_stop_period = ma_trailing_config.get('period', 20)

        # -- 趋势衰竭信号 --
        ex_config = self.config.get('trend_exhaustion', {})
        self.exhaustion_active = ex_config.get('active', False)

        # MA 突破
        ma_break_cfg = ex_config.get('ma_breakdown', {})
        self.ma_break_active = ma_break_cfg.get('active', True)
        self.ma_break_period = ma_break_cfg.get('period', 10)

        # MACD 背离与死叉
        macd_cfg = ex_config.get('macd_divergence_cross', {})
        self.macd_div_active = macd_cfg.get('active', True)
        self.macd_fast = macd_cfg.get('fastperiod', 12)
        self.macd_slow = macd_cfg.get('slowperiod', 26)
        self.macd_signal = macd_cfg.get('signalperiod', 9)

        # 异常放量
        vol_cfg = ex_config.get('anomalous_volume', {})
        self.vol_active = vol_cfg.get('active', True)
        self.vol_window = vol_cfg.get('window', 20)
        self.vol_factor = vol_cfg.get('factor', 2.0)

        # K线反转形态
        candle_cfg = ex_config.get('candlestick_reversal', {})
        self.candle_active = candle_cfg.get('active', True)
        self.candle_patterns = candle_cfg.get('patterns', ['CDLSHOOTINGSTAR', 'CDLDARKCLOUDCOVER'])

        # RSI 回落
        rsi_cfg = ex_config.get('rsi_pullback', {})
        self.rsi_pullback_active = rsi_cfg.get('active', True)
        self.rsi_period = rsi_cfg.get('period', 14)
        self.rsi_threshold = rsi_cfg.get('threshold', 80)

    def check_signals(self, position: Any, ohlcv_df: pd.DataFrame) -> Tuple[bool, str]:
        """
        检查所有激活的止盈条件，返回第一个触发的信号。
        优先级: 目标止盈 -> 移动止盈 -> 趋势衰竭。

        Args:
            position (Any): 持仓对象，应包含 entry_price, peak_price_since_entry 等属性。
            ohlcv_df (pd.DataFrame): 包含 OHLCV 数据的 DataFrame。

        Returns:
            Tuple[bool, str]: (是否止盈, 触发原因)
        """
        if ohlcv_df.empty or position is None:
            return False, ""

        # 1. 检查目标止盈
        if self.tp_active:
            should_exit, reason = self._check_target_profit(position, ohlcv_df)
            if should_exit:
                return True, reason
        
        # 2. 检查移动止盈
        # 2.1 基于MA的移动止盈
        if self.ma_trailing_active:
            should_exit, reason = self._check_ma_trailing_stop(ohlcv_df)
            if should_exit:
                return True, reason
        
        # 2.2 基于百分比的移动止盈
        if self.pct_trailing_active:
            should_exit, reason = self._check_percentage_trailing_stop(position, ohlcv_df)
            if should_exit:
                return True, reason

        # 3. 检查趋势衰竭信号
        if self.exhaustion_active:
            should_exit, reason = self._check_trend_exhaustion(ohlcv_df)
            if should_exit:
                return True, reason

        return False, ""

    def _check_target_profit(self, position: Any, ohlcv_df: pd.DataFrame) -> Tuple[bool, str]:
        """
        检查是否达到目标盈利点。
        如果最新K线的最高价超过或等于目标价位，则触发。
        """
        if not hasattr(position, 'entry_price'):
            return False, ""
            
        latest_high = ohlcv_df['high'].iloc[-1]
        profit_target_price = position.entry_price * (1 + self.tp_percentage / 100)
        
        if latest_high >= profit_target_price:
            reason = f"TAKE_PROFIT: 达到目标盈利 {self.tp_percentage}% (目标价: {profit_target_price:.2f})"
            logger.info(f"触发信号: {reason} (入场价: {position.entry_price}, 最高价: {latest_high})")
            return True, reason
        return False, ""

    def _check_ma_trailing_stop(self, ohlcv_df: pd.DataFrame) -> Tuple[bool, str]:
        """
        检查是否触发MA移动止盈。
        如果收盘价跌破指定周期的MA，则触发。
        """
        ma = talib.SMA(ohlcv_df['close'], timeperiod=self.ma_trailing_stop_period)
        if len(ma) < self.ma_trailing_stop_period:
            return False, ""

        latest_close = ohlcv_df['close'].iloc[-1]
        ma_value = ma.iloc[-1]

        if not pd.isna(ma_value) and latest_close < ma_value:
            reason = f"TAKE_PROFIT: 股价跌破 MA{self.ma_trailing_stop_period} 移动止盈线"
            logger.info(f"触发信号: {reason} (当前收盘价: {latest_close:.2f}, MA: {ma_value:.2f})")
            return True, reason
        return False, ""

    def _check_percentage_trailing_stop(self, position: Any, ohlcv_df: pd.DataFrame) -> Tuple[bool, str]:
        """
        检查是否触发百分比回撤移动止盈。
        如果当前价格从持仓期间的最高价回撤超过指定百分比，则触发。
        """
        if not hasattr(position, 'peak_price_since_entry'):
            return False, ""

        peak_price = position.peak_price_since_entry
        # 使用最低价进行更保守的判断
        latest_low = ohlcv_df['low'].iloc[-1]
        
        drawdown_pct = (peak_price - latest_low) / peak_price * 100
        
        if drawdown_pct >= self.trailing_stop_pct:
            reason = f"TAKE_PROFIT: 从高点回撤超过 {self.trailing_stop_pct}%"
            logger.info(f"触发信号: {reason} (最高价: {peak_price:.2f}, 当前最低价: {latest_low:.2f})")
            return True, reason
        return False, ""

    def _check_trend_exhaustion(self, ohlcv_df: pd.DataFrame) -> Tuple[bool, str]:
        """
        检查所有趋势衰竭信号，并根据优先级返回第一个触发的信号。
        """
        signal_checks = [
            (self._check_ma_breakdown, self.ma_break_active),
            (self._check_macd_divergence_cross, self.macd_div_active),
            (self._check_anomalous_volume, self.vol_active),
            (self._check_candlestick_reversal, self.candle_active),
            (self._check_rsi_pullback, self.rsi_pullback_active),
        ]

        for check_func, is_active in signal_checks:
            if is_active:
                signal = check_func(ohlcv_df)
                if signal:
                    logger.info(f"触发趋势衰竭信号: {signal['reason']} (详情: {signal.get('details')})")
                    return True, signal['reason']
        
        return False, ""

    def _check_ma_breakdown(self, ohlcv_df: pd.DataFrame) -> Dict[str, Any] | None:
        """
        检查价格是否跌破短期移动平均线。
        - 条件: 前一根K线收盘价在MA之上，最新K线收盘价在MA之下。
        """
        if len(ohlcv_df) < self.ma_break_period:
            return None
        
        ma = talib.SMA(ohlcv_df['close'], timeperiod=self.ma_break_period)
        if len(ma) < 2:
            return None

        latest_close = ohlcv_df['close'].iloc[-1]
        prev_close = ohlcv_df['close'].iloc[-2]
        latest_ma = ma.iloc[-1]
        prev_ma = ma.iloc[-2]
        
        if not pd.isna(latest_ma) and not pd.isna(prev_ma):
             if prev_close > prev_ma and latest_close < latest_ma:
                return {
                    "reason": f"TAKE_PROFIT: 价格跌破 MA{self.ma_break_period}",
                    "details": f"Close: {latest_close:.2f}, MA: {latest_ma:.2f}"
                }
        return None

    def _check_macd_divergence_cross(self, ohlcv_df: pd.DataFrame) -> Dict[str, Any] | None:
        """
        检查看跌MACD背离及随后的死叉确认。
        - 顶背离: 价格创下更高的高点，但MACD柱状图未创下更高的高点。
        - 死叉: MACD线（DIF）下穿信号线（DEA）。
        （注意：这是一个简化的实现，真实的背离检测可能更复杂）
        """
        if len(ohlcv_df) < self.macd_slow:
            return None
            
        macd, macdsignal, macdhist = talib.MACD(
            ohlcv_df['close'],
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )
        if len(macd) < 2:
            return None

        # 检查死叉
        prev_macd = macd.iloc[-2]
        prev_signal = macdsignal.iloc[-2]
        latest_macd = macd.iloc[-1]
        latest_signal = macdsignal.iloc[-1]

        if not pd.isna(latest_macd) and not pd.isna(latest_signal):
            if prev_macd > prev_signal and latest_macd < latest_signal:
                # 在此可以加入更复杂的背离逻辑，例如查找最近的N个周期内价格和MACD的峰值
                return {
                    "reason": "TAKE_PROFIT: MACD死叉",
                    "details": f"MACD({latest_macd:.2f}) < Signal({latest_signal:.2f})"
                }
        return None

    def _check_anomalous_volume(self, ohlcv_df: pd.DataFrame) -> Dict[str, Any] | None:
        """
        检查成交量是否异常放大但价格未相应上涨（量价背离）。
        - 条件: 最新成交量显著高于近期平均成交量，但K线实体（收盘价-开盘价）很小或为负。
        """
        if len(ohlcv_df) < self.vol_window:
            return None

        latest_bar = ohlcv_df.iloc[-1]
        avg_volume = ohlcv_df['volume'].iloc[-self.vol_window:-1].mean()
        
        price_change_ratio = abs(latest_bar['close'] - latest_bar['open']) / latest_bar['open']
        
        if latest_bar['volume'] > avg_volume * self.vol_factor and price_change_ratio < 0.01:
             return {
                "reason": "TAKE_PROFIT: 异常放量但价格无明显上涨",
                "details": f"Volume: {latest_bar['volume']}, AvgVol: {avg_volume:.0f}, PriceChange: {price_change_ratio:.2%}"
            }
        return None

    def _check_candlestick_reversal(self, ohlcv_df: pd.DataFrame) -> Dict[str, Any] | None:
        """
        使用TA-Lib检查预定义的看跌K线反转形态。
        """
        for pattern_name in self.candle_patterns:
            pattern_func = getattr(talib, pattern_name, None)
            if pattern_func:
                result = pattern_func(ohlcv_df['open'], ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'])
                if not result.empty and result.iloc[-1] < 0:  # 看跌形态通常返回-100
                    return {
                        "reason": f"TAKE_PROFIT: 检测到看跌K线形态 {pattern_name}",
                        "details": f"Pattern value: {result.iloc[-1]}"
                    }
        return None

    def _check_rsi_pullback(self, ohlcv_df: pd.DataFrame) -> Dict[str, Any] | None:
        """
        检查RSI是否从超买区域回落。
        - 条件: 前一根K线的RSI在超买阈值之上，最新K线的RSI在阈值之下。
        """
        if len(ohlcv_df) < self.rsi_period:
            return None

        rsi = talib.RSI(ohlcv_df['close'], timeperiod=self.rsi_period)
        if len(rsi) < 2:
            return None
        
        prev_rsi = rsi.iloc[-2]
        latest_rsi = rsi.iloc[-1]

        if not pd.isna(latest_rsi) and not pd.isna(prev_rsi):
            if prev_rsi > self.rsi_threshold and latest_rsi < self.rsi_threshold:
                return {
                    "reason": f"TAKE_PROFIT: RSI从超买区({self.rsi_threshold})回落",
                    "details": f"RSI: {prev_rsi:.1f} -> {latest_rsi:.1f}"
                }
        return None

if __name__ == '__main__':
    # ==============================================================================
    # 完整用法示例：模拟交易循环
    # ==============================================================================
    logger.info("="*20 + " 完整用法示例: 模拟交易循环 " + "="*20)

    # 1. 创建样本数据
    # 创建一个模拟的30天市场数据流
    entry_price_example = 100.0
    data = {
        'open': [entry_price_example + i * 0.5 for i in range(30)],
        'high': [entry_price_example + i * 0.5 + 2 for i in range(30)],
        'low': [entry_price_example + i * 0.5 - 1 for i in range(30)],
        'close': [entry_price_example + i * 0.5 + 1.5 for i in range(30)],
        'volume': [10000 + i * 100 for i in range(30)]
    }
    # 模拟一次价格回撤来触发移动止盈
    data['high'][20] = 115
    data['close'][21] = 105
    data['low'][21] = 104

    sample_df = pd.DataFrame(data, index=pd.to_datetime(pd.date_range('2024-01-01', periods=30)))

    # 2. 实例化策略
    # 使用一个更实际的配置，例如仅激活目标止盈和百分比移动止盈
    strategy_config = {
        "target_profit": {"active": False, "percentage": 25.0}, # 设置为False，以便测试移动止盈
        "trailing_profit": {
            "active": True,
            "percentage_stop": {"active": True, "percentage": 5.0}, # 从高点回撤5%止盈
            "ma_stop": {"active": False}
        },
        "trend_exhaustion": {"active": False}
    }
    take_profit_strategy = TakeProfitStrategy(config=strategy_config)

    # 3. 模拟交易循环
    # 假设我们在第二天开盘时入场
    entry_index = 1
    entry_price = sample_df['open'].iloc[entry_index]
    
    # 使用一个简单的字典来管理持仓状态
    # 在实际应用中，这可能是一个更复杂的持仓对象
    position_info = {
        'entry_price': entry_price,
        'peak_price_since_entry': entry_price # 初始化入场期间的最高价
    }

    logger.info(f"模拟交易开始，入场价: {entry_price:.2f} at {sample_df.index[entry_index].date()}")

    # 从入场后的第一根K线开始迭代
    for i in range(entry_index + 1, len(sample_df)):
        # 获取截至当前时间点的所有数据
        current_df = sample_df.iloc[:i+1]
        current_bar = current_df.iloc[-1]
        
        # 更新持仓期间的最高价
        position_info['peak_price_since_entry'] = max(
            position_info['peak_price_since_entry'],
            current_bar['high']
        )

        logger.info(
            f"[{current_bar.name.date()}] "
            f"Close: {current_bar['close']:.2f}, "
            f"Peak Price: {position_info['peak_price_since_entry']:.2f}"
        )

        # 检查止盈信号
        # 注意：在实际使用中，position_info 可以直接作为 position 参数传入
        # 因为 check_signals 方法通过 `getattr` 访问属性，也兼容字典的 `get` 方法（如果 position 是字典）
        #
        # 为了代码清晰和类型提示友好，我们用一个简单的对象来模拟
        class TempPosition:
            pass
        
        pos_obj = TempPosition()
        pos_obj.entry_price = position_info['entry_price']
        pos_obj.peak_price_since_entry = position_info['peak_price_since_entry']

        should_take_profit, reason = take_profit_strategy.check_signals(pos_obj, current_df)

        if should_take_profit:
            logger.info(f"!!! 止盈信号触发 at {current_bar.name.date()} !!!")
            logger.info(f"原因: {reason}")
            logger.info(f"详情: 当前收盘价 {current_bar['close']:.2f}, "
                        f"入场以来最高价 {position_info['peak_price_since_entry']:.2f}")
            break # 退出循环
    
    if not should_take_profit:
        logger.info("模拟结束，未触发任何止盈信号。")


    # ==============================================================================
    # 原有的单元测试，用于独立验证每个止盈逻辑
    # ==============================================================================
    logger.info("\n" + "="*20 + " 单元测试: 独立检查各项功能 " + "="*20)
    
    # 1. 配置策略
    tp_config = {
        "target_profit": {"active": True, "percentage": 15.0},
        "trailing_profit": {
            "active": True,
            "percentage_stop": {"active": True, "percentage": 8.0},
            "ma_stop": {"active": True, "period": 5}
        },
        "trend_exhaustion": {
            "active": True,
            "rsi_overbought": {"active": True, "threshold": 75},
            "ma_break": {"active": True, "period": 10}
        }
    }
    tp_strategy = TakeProfitStrategy(tp_config)

    # 2. 准备持仓和市场数据
    class MockPosition:
        def __init__(self, entry_price, peak_price_since_entry):
            self.entry_price = entry_price
            self.peak_price_since_entry = peak_price_since_entry

    position_data = MockPosition(entry_price=100, peak_price_since_entry=120)
    
    market_data = pd.DataFrame({
        'open': [102, 105, 108, 115, 118, 115, 112],
        'high': [104, 107, 110, 117, 120, 116, 113],
        'low': [101, 104, 107, 114, 117, 114, 110],
        'close': [103, 106, 109, 116, 118, 115, 111],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600]
    }, index=pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06', '2024-01-07']))

    # 3. 运行总检查
    logger.info("\n--- 运行主信号检查 ---")
    should_exit, reason = tp_strategy.check_signals(position_data, market_data)
    logger.info(f"最终决策: {'退出' if should_exit else '持仓'}, 原因: {reason}")
    

    # 4. 单独测试各个逻辑
    logger.info("\n--- 单独测试 ---")
    # 测试目标止盈
    pos_target = MockPosition(entry_price=100, peak_price_since_entry=115)
    data_target = market_data.copy()
    data_target.loc[data_target.index[-1], 'high'] = 115.1
    exit_target, reason_target = tp_strategy._check_target_profit(pos_target, data_target)
    logger.info(f"目标止盈测试(应触发): {'是' if exit_target else '否'} -> {reason_target}")

    # 测试百分比移动止盈
    pos_trail_pct = MockPosition(entry_price=100, peak_price_since_entry=120)
    data_trail_pct = market_data.copy()
    data_trail_pct.loc[data_trail_pct.index[-1], 'low'] = 110 # 120 -> 110 is a >8% drop
    exit_trail_pct, reason_trail_pct = tp_strategy._check_percentage_trailing_stop(pos_trail_pct, data_trail_pct)
    logger.info(f"百分比移动止盈测试(应触发): {'是' if exit_trail_pct else '否'} -> {reason_trail_pct}")

    # 测试MA移动止盈
    data_ma = market_data.copy()
    # 手动计算前5个周期的收盘价均值: (103+106+109+116+118)/5 = 110.4
    # 最新收盘价为 111，不触发
    exit_ma, reason_ma = tp_strategy._check_ma_trailing_stop(data_ma.head(5)) # use first 5
    logger.info(f"MA移动止盈测试(不触发): {'是' if exit_ma else '否'} -> Expected close > MA(5)")
    
    # 准备触发MA的数据
    data_ma_trigger = data_ma.copy()
    data_ma_trigger.loc[data_ma_trigger.index[-1], 'close'] = 109 # set close to 109 to trigger
    ma_value_for_last_6 = (106+109+116+118+115)/5 # MA for the 6th point
    logger.info(f"Debug MA trigger: close={data_ma_trigger['close'].iloc[-2]}, ma={ma_value_for_last_6:.2f}")

    exit_ma_2, reason_ma_2 = tp_strategy._check_ma_trailing_stop(data_ma_trigger.head(6))
    logger.info(f"MA移动止盈测试(应触发): {'是' if exit_ma_2 else '否'} -> {reason_ma_2}")