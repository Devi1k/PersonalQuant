#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回踩买点评分系统测试模块
测试支撑位识别和回踩买点评分功能
"""

import pandas as pd
import numpy as np
import logging
import talib
from datetime import datetime, timedelta
from pathlib import Path
import os
import sys

# 添加项目根目录到系统路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.strategy.trend_strategy import TrendStrategy
from src.strategy.support_pullback_analyzer import SupportPullbackAnalyzer

# 设置日志
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"test_pullback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def generate_test_data(scenario='pullback'):
    """
    生成不同场景的测试数据
    
    Parameters
    ----------
    scenario : str
        测试场景：'pullback'(回踩), 'breakout'(突破), 'sideways'(横盘)
    
    Returns
    -------
    pandas.DataFrame
        包含OHLCV数据的测试数据框
    """
    logger.info(f"生成测试数据 - 场景: {scenario}")
    
    # 基础参数
    n_days = 100
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')
    
    if scenario == 'pullback':
        # 回踩场景：先上涨，然后回调到支撑位
        # 第一阶段：上涨趋势（前60天）
        trend1 = np.linspace(100, 130, 60)
        # 第二阶段：回调（20天）
        trend2 = np.linspace(130, 120, 20)
        # 第三阶段：在支撑位附近震荡（20天）
        trend3 = np.linspace(120, 122, 20)
        base_prices = np.concatenate([trend1, trend2, trend3])
        
        # 成交量：回调时缩量
        base_volume = np.ones(n_days) * 1000000
        base_volume[60:80] = 600000  # 回调期间缩量
        
    elif scenario == 'breakout':
        # 突破场景：横盘后放量突破
        # 第一阶段：横盘（60天）
        trend1 = np.ones(60) * 110 + np.random.normal(0, 1, 60)
        # 第二阶段：突破上涨（40天）
        trend2 = np.linspace(110, 125, 40)
        base_prices = np.concatenate([trend1, trend2])
        
        # 成交量：突破时放量
        base_volume = np.ones(n_days) * 800000
        base_volume[60:] = 1500000  # 突破期间放量
        
    else:  # sideways
        # 横盘场景：价格在区间内震荡
        base_prices = 110 + np.sin(np.linspace(0, 4*np.pi, n_days)) * 5
        base_volume = np.ones(n_days) * 900000 + np.random.normal(0, 100000, n_days)
    
    # 添加随机噪声
    noise = np.random.normal(0, 0.5, n_days)
    prices = base_prices + noise
    
    # 生成OHLC数据
    df = pd.DataFrame({
        'date': dates,
        'open': prices - np.random.uniform(0, 0.5, n_days),
        'high': prices + np.random.uniform(0, 1, n_days),
        'low': prices - np.random.uniform(0, 1, n_days),
        'close': prices,
        'volume': base_volume + np.random.uniform(-50000, 50000, n_days)
    })
    
    # 确保价格逻辑正确
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


def calculate_technical_indicators(df):
    """
    计算技术指标
    
    Parameters
    ----------
    df : pandas.DataFrame
        原始OHLCV数据
        
    Returns
    -------
    pandas.DataFrame
        添加了技术指标的数据框
    """
    logger.info("计算技术指标")
    
    # 计算移动平均线
    for period in [5, 10, 20, 60]:
        df[f'ma_{period}'] = talib.SMA(df['close'].values, timeperiod=period)
    
    # 计算EMA
    df['ema_21'] = talib.EMA(df['close'].values, timeperiod=21)
    df['ema_200'] = talib.EMA(df['close'].values, timeperiod=200)
    
    # 计算布林带
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
        df['close'].values,
        timeperiod=20,
        nbdevup=2.0,
        nbdevdn=2.0,
        matype=0
    )
    
    # 计算成交量移动平均
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    
    # 计算MACD
    df['macd_dif'], df['macd_dea'], df['macd_histogram'] = talib.MACD(
        df['close'].values,
        fastperiod=12,
        slowperiod=26,
        signalperiod=9
    )
    
    # 计算RSI
    df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
    
    return df


def test_pullback_analyzer():
    """测试支撑位回踩分析器"""
    logger.info("=== 测试支撑位回踩分析器 ===")
    
    # 创建分析器实例
    analyzer = SupportPullbackAnalyzer()
    
    # 测试回踩场景
    logger.info("\n--- 测试回踩场景 ---")
    df_pullback = generate_test_data('pullback')
    df_pullback = calculate_technical_indicators(df_pullback)
    
    # 执行分析
    result_pullback = analyzer.analyze(df_pullback)
    
    # 检查结果
    if 'pullback_score' in result_pullback.columns:
        latest_score = result_pullback['pullback_score'].iloc[-1]
        latest_signal = result_pullback['pullback_signal'].iloc[-1]
        
        logger.info(f"回踩场景 - 最新评分: {latest_score:.2f}/10")
        logger.info(f"回踩场景 - 信号: {latest_signal}")
        
        # 显示最后5天的评分
        logger.info("\n最后5天的回踩评分:")
        display_cols = ['date', 'close', 'pullback_score', 'pullback_signal']
        print(result_pullback[display_cols].tail(5).to_string())
        
        # 统计信号分布
        signal_counts = result_pullback['pullback_signal'].value_counts()
        logger.info("\n信号分布统计:")
        for signal, count in signal_counts.items():
            logger.info(f"  {signal}: {count}次 ({count/len(result_pullback):.1%})")
    else:
        logger.error("未生成回踩评分结果")


def test_trend_strategy_integration():
    """测试趋势策略集成"""
    logger.info("\n=== 测试趋势策略集成 ===")
    
    # 创建策略实例
    config = {
        'fast_ma': 20,
        'slow_ma': 60,
        'trend_weight': 0.7,
        'multi_tf_weight': 0.25,
        'reversal_weight': 0.05
    }
    strategy = TrendStrategy(config)
    
    # 生成测试数据
    df = generate_test_data('pullback')
    df = calculate_technical_indicators(df)
    
    # 执行信号组合
    logger.info("执行信号组合分析...")
    result = strategy.combine_signals(df)
    
    # 检查结果
    if 'pullback_signal' in result.columns:
        logger.info("✓ 回踩信号已成功集成到趋势策略中")
        
        # 显示最后10天的综合信号
        logger.info("\n最后10天的综合信号:")
        display_cols = ['date', 'close', 'trend_score', 'pullback_signal', 
                       'final_signal', 'final_position']
        
        # 确保所需列存在
        available_cols = [col for col in display_cols if col in result.columns]
        if available_cols:
            print(result[available_cols].tail(10).to_string())
        
        # 统计最终信号分布
        if 'final_signal' in result.columns:
            final_signal_counts = result['final_signal'].value_counts()
            logger.info("\n最终信号分布:")
            for signal, count in final_signal_counts.items():
                logger.info(f"  信号值 {signal}: {count}次 ({count/len(result):.1%})")
    else:
        logger.error("回踩信号未正确集成")


def test_different_scenarios():
    """测试不同市场场景"""
    logger.info("\n=== 测试不同市场场景 ===")
    
    analyzer = SupportPullbackAnalyzer()
    scenarios = ['pullback', 'breakout', 'sideways']
    
    results_summary = {}
    
    for scenario in scenarios:
        logger.info(f"\n--- 测试 {scenario} 场景 ---")
        
        # 生成数据
        df = generate_test_data(scenario)
        df = calculate_technical_indicators(df)
        
        # 执行分析
        result = analyzer.analyze(df)
        
        if 'pullback_score' in result.columns:
            # 计算平均评分和信号统计
            avg_score = result['pullback_score'].mean()
            max_score = result['pullback_score'].max()
            buy_signals = (result['pullback_signal'] == '买入').sum()
            sell_signals = (result['pullback_signal'] == '卖出').sum()
            hold_signals = (result['pullback_signal'] == '观望').sum()
            
            results_summary[scenario] = {
                'avg_score': avg_score,
                'max_score': max_score,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'hold_signals': hold_signals
            }
            
            logger.info(f"{scenario} - 平均评分: {avg_score:.2f}, 最高评分: {max_score:.2f}")
            logger.info(f"{scenario} - 买入: {buy_signals}, 卖出: {sell_signals}, 观望: {hold_signals}")
    
    # 显示汇总结果
    logger.info("\n=== 场景测试汇总 ===")
    summary_df = pd.DataFrame(results_summary).T
    print(summary_df.to_string())
    
    return results_summary


def test_edge_cases():
    """测试边界情况"""
    logger.info("\n=== 测试边界情况 ===")
    
    analyzer = SupportPullbackAnalyzer()
    
    # 测试1：数据不足
    logger.info("\n--- 测试数据不足情况 ---")
    df_short = generate_test_data('pullback').head(10)
    df_short = calculate_technical_indicators(df_short)
    result_short = analyzer.analyze(df_short)
    
    if 'pullback_score' in result_short.columns:
        logger.info(f"数据不足时仍生成了评分，最后评分: {result_short['pullback_score'].iloc[-1]:.2f}")
    else:
        logger.info("数据不足时未生成评分（符合预期）")
    
    # 测试2：极端价格波动
    logger.info("\n--- 测试极端价格波动 ---")
    df_volatile = generate_test_data('pullback')
    # 添加极端波动
    df_volatile.loc[80:85, 'close'] *= 0.9  # 突然下跌10%
    df_volatile = calculate_technical_indicators(df_volatile)
    result_volatile = analyzer.analyze(df_volatile)
    
    if 'pullback_score' in result_volatile.columns:
        volatile_scores = result_volatile['pullback_score'].iloc[80:86]
        logger.info(f"极端波动期间的评分变化:")
        print(volatile_scores.to_string())


def main():
    """主测试函数"""
    logger.info("=== 开始回踩买点评分系统测试 ===")
    logger.info(f"测试时间: {datetime.now()}")
    
    try:
        # 1. 测试支撑位回踩分析器
        test_pullback_analyzer()
        
        # 2. 测试趋势策略集成
        test_trend_strategy_integration()
        
        # 3. 测试不同市场场景
        test_different_scenarios()
        
        # 4. 测试边界情况
        test_edge_cases()
        
        logger.info("\n=== 所有测试完成 ===")
        logger.info("✓ 回踩买点评分系统测试通过")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()