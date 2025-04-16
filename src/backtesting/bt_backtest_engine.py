#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于 Backtrader 的回测引擎
提供完整的回测功能和性能评估
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import yaml
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import json
import backtrader as bt
import backtrader.analyzers as btanalyzers
import backtrader.feeds as btfeeds
import backtrader.indicators as btind
import backtrader.strategies as btstrats

# 添加项目根目录到sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入自定义模块 - 使用绝对导入
from src.utils.db_utils import get_db_engine, df_to_sql, load_config, query_etf_history, query_etf_list
from src.data.data_processor import DataProcessor
from src.strategy import FusionStrategy, TrendStrategy, SwingStrategy

# 设置日志
logger = logging.getLogger(__name__)

# 数据源适配器 - 将 DataFrame 转换为 Backtrader 可用的数据源
class PandasDataExtend(btfeeds.PandasData):
    """扩展 Backtrader 的 PandasData 数据源，添加自定义列"""
    
    # 添加自定义列
    lines = ('volume_ma5', 'volume_ma10', 'ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'ma250',
             'rsi', 'kdj_k', 'kdj_d', 'kdj_j', 'macd', 'macd_signal', 'macd_hist',
             'boll_upper', 'boll_middle', 'boll_lower', 'atr', 'adx', 'signal', 'long_stop_loss', 'short_stop_loss')
    
    # 设置列的默认值
    params = (
        ('datetime', None),
        ('open', -1),
        ('high', -1),
        ('low', -1),
        ('close', -1),
        ('volume', -1),
        ('openinterest', None),
        ('volume_ma5', -1),
        ('volume_ma10', -1),
        ('ma5', -1),
        ('ma10', -1),
        ('ma20', -1),
        ('ma60', -1),
        ('ma120', -1),
        ('ma250', -1),
        ('rsi', -1),
        ('kdj_k', -1),
        ('kdj_d', -1),
        ('kdj_j', -1),
        ('macd', -1),
        ('macd_signal', -1),
        ('macd_hist', -1),
        ('boll_upper', -1),
        ('boll_middle', -1),
        ('boll_lower', -1),
        ('atr', -1),
        ('adx', -1),
        ('signal', 0),  # 0: 无信号, 1: 买入, -1: 卖出
        ('long_stop_loss', -1),
        ('short_stop_loss', -1),
    )


# 策略类 - 基于现有策略的 Backtrader 实现
class FusionStrategyBT(bt.Strategy):
    """基于融合策略的 Backtrader 策略实现"""
    
    params = (
        ('trend_weight', 0.6),  # 趋势策略权重
        ('swing_weight', 0.4),  # 波段策略权重
        ('adx_threshold', 25),  # ADX 阈值，用于判断趋势强度
        ('confirmation_multiplier', 1.5),  # 协同确认增强系数
        ('risk_pct', 0.02),  # 单笔风险比例
        ('max_positions', 5),  # 最大持仓数量
        ('use_custom_signals', True),  # 是否使用自定义信号
        ('atr_stop_loss_multiplier', 1.5),  # ATR止损乘数
        ('trailing_stop', False),  # 是否启用跟踪止损
        ('trailing_percent', 0.5),  # 跟踪止损百分比(0-1)，表示ATR的比例
    )
    
    def __init__(self):
        """初始化策略"""
        
        # 记录订单和持仓
        self.orders = {}  # 订单跟踪
        self.positions_info = {}  # 持仓信息
        
        # 使用自定义信号或计算指标
        if self.params.use_custom_signals:
            # 使用数据源中已有的信号
            self.signal = self.data.signal
        else:
            # 计算自定义指标
            # 移动平均线
            self.ma5 = self.data.ma5 if 'ma5' in self.data.lines._getlinealias() else bt.indicators.SMA(self.data.close, period=5)
            self.ma10 = self.data.ma10 if 'ma10' in self.data.lines._getlinealias() else bt.indicators.SMA(self.data.close, period=10)
            self.ma20 = self.data.ma20 if 'ma20' in self.data.lines._getlinealias() else bt.indicators.SMA(self.data.close, period=20)
            self.ma60 = self.data.ma60 if 'ma60' in self.data.lines._getlinealias() else bt.indicators.SMA(self.data.close, period=60)
            
            # RSI
            self.rsi = self.data.rsi if 'rsi' in self.data.lines._getlinealias() else bt.indicators.RSI(self.data.close, period=14)
            
            # MACD
            self.macd = bt.indicators.MACD(
                self.data.close, 
                period_me1=12, 
                period_me2=26, 
                period_signal=9
            )
            
            # 布林带
            self.bollinger = bt.indicators.BollingerBands(
                self.data.close, 
                period=20, 
                devfactor=2.0
            )
            
            # ADX
            self.adx = self.data.adx if 'adx' in self.data.lines._getlinealias() else bt.indicators.ADX(self.data, period=14)
            
            # ATR
            self.atr = self.data.atr if 'atr' in self.data.lines._getlinealias() else bt.indicators.ATR(self.data, period=14)
    
    def log(self, txt, dt=None):
        """日志函数"""
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f'{dt.isoformat()} {txt}')
    
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            # 订单提交或接受，不做任何处理
            return
        
        # 检查订单是否完成
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行: 价格={order.executed.price:.2f}, 成本={order.executed.value:.2f}, 手续费={order.executed.comm:.2f}')
                # 记录买入价格和止损价格
                data = order.data
                atr_value = data.atr[0] if hasattr(data.lines, 'atr') and data.atr[0] > 0 else self.atr[0] if hasattr(self, 'atr') else 0
                
                # 计算止损价格
                stop_loss_price = order.executed.price - (atr_value * self.params.atr_stop_loss_multiplier)
                
                self.positions_info[order.data._name] = {
                    'price': order.executed.price,
                    'size': order.executed.size,
                    'value': order.executed.value,
                    'commission': order.executed.comm,
                    'time': self.data.datetime.datetime(0),
                    'stop_loss': stop_loss_price,  # 记录初始止损价格
                    'highest_price': order.executed.price,  # 记录最高价格，用于跟踪止损
                }
                self.log(f'设置止损价格: {stop_loss_price:.2f}')
            else:
                self.log(f'卖出执行: 价格={order.executed.price:.2f}, 成本={order.executed.value:.2f}, 手续费={order.executed.comm:.2f}')
                # 计算收益
                if order.data._name in self.positions_info:
                    buy_price = self.positions_info[order.data._name]['price']
                    profit = (order.executed.price - buy_price) * order.executed.size
                    self.log(f'收益: {profit:.2f}')
                    # 清除持仓信息
                    del self.positions_info[order.data._name]
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'订单取消/拒绝/保证金不足')
        
        # 移除订单
        self.orders[order.data._name] = None
    
    def notify_trade(self, trade):
        """交易结果通知"""
        if not trade.isclosed:
            return
        
        self.log(f'交易利润: 毛利={trade.pnl:.2f}, 净利={trade.pnlcomm:.2f}')
    
    def next(self):
        """策略核心逻辑，每个交易日调用一次"""
        
        # 遍历所有数据源
        for i, data in enumerate(self.datas):
            ticker = data._name
            
            # 检查是否有未完成的订单
            if self.orders.get(ticker, None):
                continue
            
            # 获取当前持仓
            position = self.getposition(data)
            
            # 处理多头持仓
            if position and position.size > 0:
                current_price = data.close[0]
                
                # 更新持仓信息中的最高价格（用于跟踪止损）
                if ticker in self.positions_info:
                    if current_price > self.positions_info[ticker]['highest_price']:
                        self.positions_info[ticker]['highest_price'] = current_price
                    
                    # 如果启用跟踪止损，动态调整止损价格
                    if self.params.trailing_stop:
                        initial_stop = self.positions_info[ticker]['stop_loss']
                        entry_price = self.positions_info[ticker]['price']
                        highest_price = self.positions_info[ticker]['highest_price']
                        
                        # 计算潜在的新止损价格（基于最高价格）
                        atr_value = data.atr[0] if hasattr(data.lines, 'atr') and data.atr[0] > 0 else self.atr[0] if hasattr(self, 'atr') else 0
                        trailing_distance = atr_value * self.params.atr_stop_loss_multiplier * self.params.trailing_percent
                        potential_stop = highest_price - trailing_distance
                        
                        # 只有当新止损价格高于当前止损价格时才更新
                        if potential_stop > self.positions_info[ticker]['stop_loss']:
                            self.positions_info[ticker]['stop_loss'] = potential_stop
                            self.log(f'调整止损价格: {ticker}, 新止损价={potential_stop:.2f}')
                    
                    # 检查是否触发止损
                    stop_loss_price = self.positions_info[ticker]['stop_loss']
                    
                    # 如果价格低于止损价，执行止损
                    if current_price <= stop_loss_price:
                        self.log(f'止损触发: {ticker}, 当前价格={current_price:.2f}, 止损价={stop_loss_price:.2f}')
                        self.orders[ticker] = self.sell(data=data, size=position.size)
                        continue  # 已经执行了止损，跳过后续信号检查
                
                # 如果没有持仓信息但有long_stop_loss列，使用数据源中的止损价格
                elif hasattr(data.lines, 'long_stop_loss') and data.long_stop_loss[0] > 0:
                    stop_loss_price = data.long_stop_loss[0]
                    
                    # 如果价格低于止损价，执行止损
                    if current_price <= stop_loss_price:
                        self.log(f'数据源止损触发: {ticker}, 当前价格={current_price:.2f}, 止损价={stop_loss_price:.2f}')
                        self.orders[ticker] = self.sell(data=data, size=position.size)
                        continue  # 已经执行了止损，跳过后续信号检查
            
            # 处理空头持仓的止损条件
            elif position and position.size < 0:
                # 空头止损逻辑类似，但方向相反
                # 这里简化处理，只使用数据源中的止损价格
                if hasattr(data.lines, 'short_stop_loss') and data.short_stop_loss[0] > 0:
                    current_price = data.close[0]
                    stop_loss_price = data.short_stop_loss[0]
                    
                    # 如果价格高于止损价，执行止损
                    if current_price >= stop_loss_price:
                        self.log(f'空头止损触发: {ticker}, 当前价格={current_price:.2f}, 止损价={stop_loss_price:.2f}')
                        self.orders[ticker] = self.buy(data=data, size=abs(position.size))
                        continue  # 已经执行了止损，跳过后续信号检查
            
            # 使用自定义信号
            if self.params.use_custom_signals:
                signal = self.data.signal[0]
                
                # 买入信号
                if signal == 1 and not position:
                    # 计算买入数量
                    price = data.close[0]
                    cash = self.broker.getcash()
                    risk_amount = cash * self.params.risk_pct
                    size = int(risk_amount / price)
                    
                    if size > 0:
                        self.log(f'买入信号: {ticker}, 价格={price:.2f}, 数量={size}')
                        self.orders[ticker] = self.buy(data=data, size=size)
                
                # 卖出信号
                elif signal == -1 and position:
                    self.log(f'卖出信号: {ticker}, 价格={data.close[0]:.2f}, 数量={position.size}')
                    self.orders[ticker] = self.sell(data=data, size=position.size)
            
            # 使用自定义指标计算信号
            else:
                # 这里可以实现自定义的信号生成逻辑
                # 例如：趋势策略 + 波段策略的融合
                
                # 趋势信号 (简化示例)
                trend_signal = 0
                if self.ma5[0] > self.ma20[0] and self.ma5[-1] <= self.ma20[-1]:
                    trend_signal = 1  # 金叉买入
                elif self.ma5[0] < self.ma20[0] and self.ma5[-1] >= self.ma20[-1]:
                    trend_signal = -1  # 死叉卖出
                
                # 波段信号 (简化示例)
                swing_signal = 0
                if self.rsi[0] < 30 and self.rsi[-1] < self.rsi[0]:  # RSI 超卖反弹
                    swing_signal = 1
                elif self.rsi[0] > 70 and self.rsi[-1] > self.rsi[0]:  # RSI 超买回落
                    swing_signal = -1
                
                # 判断市场状态
                is_trend_market = self.adx[0] > self.params.adx_threshold
                
                # 根据市场状态调整权重
                if is_trend_market:
                    trend_weight = 0.8
                    swing_weight = 0.2
                else:
                    trend_weight = 0.3
                    swing_weight = 0.7
                
                # 融合信号
                final_signal = trend_weight * trend_signal + swing_weight * swing_signal
                
                # 协同确认增强
                if trend_signal == swing_signal and trend_signal != 0:
                    final_signal *= self.params.confirmation_multiplier
                
                # 执行交易
                if not position and final_signal > 0.5:  # 买入阈值
                    # 计算买入数量
                    price = data.close[0]
                    cash = self.broker.getcash()
                    risk_amount = cash * self.params.risk_pct
                    size = int(risk_amount / price)
                    
                    if size > 0:
                        self.log(f'买入信号: {ticker}, 价格={price:.2f}, 数量={size}')
                        self.orders[ticker] = self.buy(data=data, size=size)
                
                elif position and final_signal < -0.5:  # 卖出阈值
                    self.log(f'卖出信号: {ticker}, 价格={data.close[0]:.2f}, 数量={position.size}')
                    self.orders[ticker] = self.sell(data=data, size=position.size)


# 回测引擎类
class BTBacktestEngine:
    """基于 Backtrader 的回测引擎类"""
    
    def __init__(self, config_file):
        """
        初始化回测引擎
        
        Parameters
        ----------
        config_file : str
            配置文件路径
        """
        self.config_file = config_file
        self.config = self._load_config()
        
        # 回测参数
        self.start_date = self.config['backtest'].get('start_date', '2018-01-01')
        self.end_date = self.config['backtest'].get('end_date', '2024-03-01')
        self.initial_capital = self.config['backtest'].get('initial_capital', 1000000)
        self.commission = self.config['backtest'].get('commission', 0.0003)
        self.slippage = self.config['backtest'].get('slippage', 0.001)
        self.benchmark = self.config['backtest'].get('benchmark', '000300')
        
        # 数据处理器
        self.data_processor = DataProcessor()
        
        # 初始化数据库引擎
        self.config_db = load_config()
        self.engine = get_db_engine(self.config_db)
        
        # 策略配置
        self.strategy_config = self.config.get('strategy', {})
        
        # 回测结果
        self.results = {
            'equity_curve': None,
            'trades': [],
            'performance': {}
        }
        
        # 创建 Backtrader 回测实例
        self.cerebro = bt.Cerebro()
        
        # 设置初始资金
        self.cerebro.broker.setcash(self.initial_capital)
        
        # 设置手续费
        self.cerebro.broker.setcommission(commission=self.commission)
        
        # 设置滑点
        self.cerebro.broker.set_slippage_perc(self.slippage)
        
        # 添加分析器
        self._add_analyzers()
        
        logger.info(f"回测引擎初始化完成，回测区间: {self.start_date} 至 {self.end_date}")
    
    def _load_config(self):
        """
        加载配置文件
        
        Returns
        -------
        dict
            配置参数
        """
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载配置文件: {self.config_file}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return {}
    
    def _add_analyzers(self):
        """添加回测分析器"""
        # 添加回测分析器
        self.cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02, annualize=True)
        self.cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(btanalyzers.SQN, _name='sqn')
        self.cerebro.addanalyzer(btanalyzers.TimeReturn, _name='time_return', timeframe=bt.TimeFrame.Days)
    
    def load_data(self, symbol):
        """
        从数据库加载历史数据并转换为 Backtrader 数据源
        
        Parameters
        ----------
        symbol : str
            交易品种代码
            
        Returns
        -------
        backtrader.feeds.PandasData
            Backtrader 数据源
        """
        try:
            logger.info(f"开始从数据库加载 {symbol} 的历史数据，时间区间: {self.start_date} 至 {self.end_date}")
            
            if not self.engine:
                logger.error("数据库引擎未初始化，无法加载数据")
                return None
            
            # 使用 query_etf_history 函数从数据库查询数据
            df = query_etf_history(
                engine=self.engine,
                etf_code=symbol,
                start_date=self.start_date,
                end_date=self.end_date,
                fields=None  # 获取所有字段
            )
            
            if df.empty:
                logger.error(f"未能从数据库获取到 {symbol} 的历史数据")
                return None
            
            # 确保日期列名一致
            if 'trade_date' in df.columns and 'date' not in df.columns:
                df['date'] = df['trade_date']
            
            # 过滤日期范围
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
            
            # 使用融合策略生成信号
            fusion_strategy = FusionStrategy(self.strategy_config)
            df = fusion_strategy.combine_strategies(df)
            
            # 准备 Backtrader 数据源
            df.set_index('date', inplace=True)
            
            # 确保列名与 PandasDataExtend 中定义的一致
            column_mapping = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'ma5': 'ma5',
                'ma10': 'ma10',
                'ma20': 'ma20',
                'ma60': 'ma60',
                'ma120': 'ma120',
                'ma250': 'ma250',
                'volume_ma5': 'volume_ma5',
                'volume_ma10': 'volume_ma10',
                'rsi': 'rsi',
                'kdj_k': 'kdj_k',
                'kdj_d': 'kdj_d',
                'kdj_j': 'kdj_j',
                'macd': 'macd',
                'macd_signal': 'macd_signal',
                'macd_hist': 'macd_hist',
                'boll_upper': 'boll_upper',
                'boll_middle': 'boll_middle',
                'boll_lower': 'boll_lower',
                'atr': 'atr',
                'adx': 'adx',
                'signal': 'signal',
                'long_stop_loss': 'long_stop_loss',
                'short_stop_loss': 'short_stop_loss'
            }
            
            # 重命名列
            rename_dict = {}
            for bt_col, df_col in column_mapping.items():
                if df_col in df.columns:
                    rename_dict[df_col] = bt_col
            
            df = df.rename(columns=rename_dict)
            
            # 创建 Backtrader 数据源
            data = PandasDataExtend(
                dataname=df,
                name=symbol,
                fromdate=datetime.strptime(self.start_date, '%Y-%m-%d'),
                todate=datetime.strptime(self.end_date, '%Y-%m-%d'),
                plot=True
            )
            
            logger.info(f"成功从数据库加载 {symbol} 的历史数据，共 {len(df)} 条记录")
            return data
        except Exception as e:
            logger.error(f"从数据库加载历史数据失败: {e}")
            return None
    
    def run_backtest(self, symbols):
        """
        运行回测
        
        Parameters
        ----------
        symbols : list
            交易品种代码列表
            
        Returns
        -------
        dict
            回测结果
        """
        logger.info(f"开始回测，交易品种: {symbols}")
        
        # 添加数据
        for symbol in symbols:
            data = self.load_data(symbol)
            if data:
                self.cerebro.adddata(data)
        
        # 添加策略
        strategy_params = {
            'trend_weight': self.strategy_config.get('trend_weight', 0.6),
            'swing_weight': self.strategy_config.get('swing_weight', 0.4),
            'adx_threshold': self.strategy_config.get('adx_threshold', 25),
            'confirmation_multiplier': self.strategy_config.get('confirmation_multiplier', 1.5),
            'risk_pct': self.strategy_config.get('risk_pct', 0.02),
            'max_positions': self.strategy_config.get('max_positions', 5),
            'use_custom_signals': self.strategy_config.get('use_custom_signals', True),
            'atr_stop_loss_multiplier': self.strategy_config.get('atr_stop_loss_multiplier', 1.5),
            'trailing_stop': self.strategy_config.get('trailing_stop', False),
            'trailing_percent': self.strategy_config.get('trailing_percent', 0.5)
        }
        
        self.cerebro.addstrategy(FusionStrategyBT, **strategy_params)
        
        # 运行回测
        logger.info("开始运行回测")
        results = self.cerebro.run()
        strategy = results[0]
        
        # 提取回测结果
        self._extract_results(strategy)
        
        logger.info("回测完成")
        
        return self.results
    
    def _extract_results(self, strategy):
        """
        提取回测结果
        
        Parameters
        ----------
        strategy : backtrader.Strategy
            回测策略实例
        """
        # 提取性能指标
        sharpe = strategy.analyzers.sharpe.get_analysis()
        drawdown = strategy.analyzers.drawdown.get_analysis()
        returns = strategy.analyzers.returns.get_analysis()
        trades = strategy.analyzers.trades.get_analysis()
        sqn = strategy.analyzers.sqn.get_analysis()
        time_return = strategy.analyzers.time_return.get_analysis()
        
        # 转换为 pandas.Series 以便于后续处理
        equity_curve = pd.Series(time_return).cumsum()
        
        # 保存结果
        self.results['equity_curve'] = equity_curve
        
        # 性能指标
        self.results['performance'] = {
            'initial_capital': self.initial_capital,
            'final_value': self.cerebro.broker.getvalue(),
            'total_return': (self.cerebro.broker.getvalue() / self.initial_capital - 1) * 100,
            'annual_return': returns.get('ravg', 0) * 100,
            'sharpe_ratio': sharpe.get('sharperatio', 0),
            'max_drawdown': drawdown.get('max', {}).get('drawdown', 0) * 100,
            'max_drawdown_length': drawdown.get('max', {}).get('len', 0),
            'sqn': sqn.get('sqn', 0),
            'trade_count': trades.get('total', {}).get('total', 0),
            'win_count': trades.get('won', {}).get('total', 0),
            'loss_count': trades.get('lost', {}).get('total', 0),
            'win_rate': trades.get('won', {}).get('total', 0) / max(1, trades.get('total', {}).get('total', 1)) * 100,
            'avg_trade_pnl': trades.get('pnl', {}).get('average', 0),
            'max_trade_pnl': trades.get('pnl', {}).get('max', 0),
            'min_trade_pnl': trades.get('pnl', {}).get('min', 0),
            'avg_trade_length': trades.get('len', {}).get('average', 0)
        }
        
        logger.info(f"回测结果: 总收益率 {self.results['performance']['total_return']:.2f}%, 夏普比率 {self.results['performance']['sharpe_ratio']:.2f}, 最大回撤 {self.results['performance']['max_drawdown']:.2f}%")
    
    def plot_results(self, filename=None):
        """
        绘制回测结果
        
        Parameters
        ----------
        filename : str, optional
            保存图表的文件名，如果为 None 则显示图表
        """
        try:
            # 绘制回测结果
            self.cerebro.plot(style='candle', barup='red', bardown='green', 
                             volup='red', voldown='green', 
                             grid=True, volume=True, 
                             subplot=True, iplot=False)
            
            if filename:
                plt.savefig(filename)
                logger.info(f"回测结果图表已保存至: {filename}")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"绘制回测结果失败: {e}")
    
    def save_results(self, output_dir):
        """
        保存回测结果
        
        Parameters
        ----------
        output_dir : str
            输出目录
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 保存权益曲线
            if self.results['equity_curve'] is not None:
                equity_curve_file = output_path / 'equity_curve.csv'
                self.results['equity_curve'].to_csv(equity_curve_file)
                logger.info(f"权益曲线已保存至: {equity_curve_file}")
            
            # 保存性能指标
            performance_file = output_path / 'performance.json'
            with open(performance_file, 'w', encoding='utf-8') as f:
                json.dump(self.results['performance'], f, ensure_ascii=False, indent=4)
            logger.info(f"性能指标已保存至: {performance_file}")
            
            # 保存图表
            plot_file = output_path / 'backtest_plot.png'
            self.plot_results(filename=str(plot_file))
            
            logger.info(f"回测结果已保存至: {output_dir}")
        except Exception as e:
            logger.error(f"保存回测结果失败: {e}")


# 运行回测的入口函数
def run_backtest(config_file, symbols=None, start_date=None, end_date=None, output_dir=None):
    """
    运行回测的入口函数
    
    Parameters
    ----------
    config_file : str
        配置文件路径
    symbols : list, optional
        交易品种代码列表，如果为 None 则使用配置文件中的设置
    start_date : str, optional
        回测开始日期，格式YYYY-MM-DD，如果为None则使用配置文件中的设置
    end_date : str, optional
        回测结束日期，格式YYYY-MM-DD，如果为None则使用配置文件中的设置
    output_dir : str, optional
        输出目录，如果为None则使用当前目录下的 'backtest_results'
    
    Returns
    -------
    dict
        回测结果
    """
    try:
        # 加载配置
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 覆盖配置
        if start_date:
            config['backtest']['start_date'] = start_date
        if end_date:
            config['backtest']['end_date'] = end_date
        
        # 保存修改后的配置
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        # 初始化回测引擎
        engine = BTBacktestEngine(config_file)
        
        # 如果未指定交易品种，则使用配置中的设置
        if symbols is None:
            symbols = config.get('symbols', [])
        
        # 运行回测
        results = engine.run_backtest(symbols)
        
        # 保存结果
        if output_dir is None:
            output_dir = 'backtest_results'
        engine.save_results(output_dir)
        
        return results
    except Exception as e:
        logger.error(f"运行回测失败: {e}")
        return {}


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行回测
    config_file = 'config/config.yaml'
    symbols = ['510300', '510500']  # 沪深300ETF, 中证500ETF
    
    results = run_backtest(
        config_file=config_file,
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2023-12-31',
        output_dir='backtest_results'
    )
    
    print(f"回测完成，总收益率: {results['performance']['total_return']:.2f}%")
