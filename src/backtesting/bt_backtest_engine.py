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

log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)
# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"backtest_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
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
        ('open', 'open'),        # Explicitly map 'open' line to 'open' column
        ('high', 'high'),        # Explicitly map 'high' line to 'high' column
        ('low', 'low'),          # Explicitly map 'low' line to 'low' column
        ('close', 'close'),      # Explicitly map 'close' line to 'close' column
        ('volume', 'volume'),    # Explicitly map 'volume' line to 'volume' column
        ('openinterest', None),
        ('volume_ma5', 'volume_ma5'),
        ('volume_ma10', 'volume_ma10'),
        ('ma5', 'ma5'),
        ('ma10', 'ma10'),
        ('ma20', 'ma20'),
        ('ma60', 'ma60'),
        ('ma120', 'ma120'),
        ('ma250', 'ma250'),
        ('rsi', 'rsi'),
        ('kdj_k', 'kdj_k'),
        ('kdj_d', 'kdj_d'),
        ('kdj_j', 'kdj_j'),
        ('macd', 'macd'),
        ('macd_signal', 'macd_signal'),
        ('macd_hist', 'macd_hist'),
        ('boll_upper', 'boll_upper'),
        ('boll_middle', 'boll_middle'),
        ('boll_lower', 'boll_lower'),
        ('atr', 'atr'),
        ('adx', 'adx'),
        ('signal', 'signal'), # Explicitly map 'signal' line to 'signal' column
        ('long_stop_loss', 'long_stop_loss'),
        ('short_stop_loss', 'short_stop_loss'),
    )


# 策略类 - 基于现有策略的 Backtrader 实现
class FusionStrategyBT(bt.Strategy):
    """基于融合策略的 Backtrader 策略实现"""
    
    params = (
        ('trend_weight', 0.6),  # 趋势策略权重
        ('swing_weight', 0.4),  # 波段策略权重
        ('adx_threshold', 25),  # ADX 阈值，用于判断趋势强度
        ('confirmation_multiplier', 1.5),  # 协同确认增强系数
        ('risk_pct', 0.02),  # 单笔风险比例 (基于总资产)
        ('max_positions', 5),  # 最大持仓数量
        ('use_custom_signals', True),  # 是否使用自定义信号
        # 下面的止损参数主要在 use_custom_signals=False 时使用，或作为信号驱动模式的备用
        ('atr_stop_loss_multiplier', 1.5),
        ('trailing_stop', False),
        ('trailing_percent', 0.5),
    )

    def __init__(self):
        """初始化策略"""
        super().__init__()
        self.order = None  # 初始化订单状态
        self.positions_info = {}

        if self.params.use_custom_signals:
            if 'signal' not in self.data.lines.getlinealiases():
                 raise ValueError("数据源缺少 'signal' 列，无法使用自定义信号模式。")
            self.signal = self.data.signal
            # 保留ATR计算，可能用于其他地方（如分析或备用止损）
            self.atr = self.data.atr if hasattr(self.data.lines, 'atr') else bt.indicators.ATR(self.data, period=14)
            logger.info("策略初始化：使用数据源中的自定义信号。")
        else:
            logger.info("策略初始化：使用内置指标计算信号。")
            # 此处省略了内置指标的计算代码，保持不变
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
        logger.info(f'{txt}')

    def notify_order(self, order):
        """订单状态通知"""
        ticker = order.data._name if order.data else 'Unknown'
        current_date = self.data.datetime.date(0)

        if order.status in [order.Submitted, order.Accepted]:
            # 订单已提交或已接受，无需操作
            if self.order and self.order.ref == order.ref:
                self.log(f'{current_date} - Order {order.Status[order.status]} - Ref: {order.ref}, Ticker: {ticker}')
            return

        if order.status in [order.Completed]:
            if self.order and self.order.ref == order.ref:
                if order.isbuy():
                    self.log(f'{current_date} - BUY EXECUTED - Ref: {order.ref}, Ticker: {ticker}, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                elif order.issell():
                    self.log(f'{current_date} - SELL EXECUTED - Ref: {order.ref}, Ticker: {ticker}, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.order = None # Clear the completed order reference

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.order and self.order.ref == order.ref:
                self.log(f'{current_date} - Order {order.Status[order.status]} - Ref: {order.ref}, Ticker: {ticker}')
                self.order = None # Clear the failed order reference

    def notify_trade(self, trade):
        """交易状态通知"""
        if not trade.isclosed:
            return
        ticker = trade.data._name
        self.log(f'--- 交易结束 @ {ticker} ---')
        self.log(f'  毛利润: {trade.pnl:.2f}')
        self.log(f'  净利润: {trade.pnlcomm:.2f}')
        self.log(f'-------------------------')


    def calculate_position_size(self):
        """根据风险和现金计算仓位大小"""
        current_price = self.data.close[0]
        total_value = self.broker.getvalue()
        cash = self.broker.getcash()
        
        # 计算当前持仓数量 (需要改进以支持多品种)
        # current_positions = sum(1 for pos in self.broker.positions if self.broker.positions[pos] and self.broker.getposition(self.broker.positions[pos]).size != 0)
        # 简化处理：假设只交易一种资产或平均分配风险
        current_positions = 1 if self.position else 0 # 简化：如果已有仓位则认为占用了1个，否则0个
        remaining_slots = max(1, self.params.max_positions - current_positions) # 至少留1个槽位

        # 1. 基于总资产风险计算
        risk_per_position_value = total_value * self.params.risk_pct / self.params.max_positions
        size_by_risk = int(risk_per_position_value / current_price) if current_price > 0 else 0
        
        # 2. 基于可用现金计算
        cash_per_slot = (cash * 0.95) / remaining_slots # 预留5%现金
        size_by_cash = int(cash_per_slot / current_price) if current_price > 0 else 0
        
        # 取两者最小值
        final_size = max(0, min(size_by_risk, size_by_cash)) 
        
        self.log(f'Calculate Size: Price={current_price:.2f}, RiskSize={size_by_risk}, CashSize={size_by_cash} -> FinalSize={final_size}')
        return final_size

    def next(self):
        """策略核心逻辑，每个交易日调用一次"""
        for i, data in enumerate(self.datas):
            ticker = data._name
            current_date = data.datetime.date(0) # Use loop variable 'data'
            current_close = data.close[0]       # Use loop variable 'data'
            position_size = self.position.size
            
            if self.order:
                # 如果有挂单，不再执行新的交易逻辑
                return
            
            if self.params.use_custom_signals:
                # 使用来自数据源的自定义信号
                signal = data.signal[0]         # Use loop variable 'data'
                
                # --- 添加日志 --- 
                self.log(f'{current_date} - Close: {current_close:.2f}, Signal: {signal}, Position: {position_size}')
                # --------------- 
                
                # 基于信号执行交易
                if signal > 0:  # 买入信号 (1 或 0.5)
                    if not self.position:  # 如果没有持仓
                        # --- 添加日志 ---
                        self.log(f'{current_date} - BUY SIGNAL DETECTED (Signal: {signal}), No current position. Placing Buy Order.')
                        # ---------------
                        size = self.calculate_position_size()
                        if size > 0:
                            self.log(f'BUY CREATE, {current_close:.2f}, Size: {size}')
                            self.order = self.buy(size=size)
                    else:
                        # 如果已有持仓，根据信号强度决定是否加仓 (此处简化为不加仓)
                        self.log(f'{current_date} - Buy Signal ({signal}) but already in position. Holding.')
                        pass 

                elif signal < 0:  # 卖出信号 (-1 或 -0.5)
                    if self.position:  # 如果持有多头仓位
                        # --- 添加日志 ---
                        self.log(f'{current_date} - SELL SIGNAL DETECTED (Signal: {signal}), Current position exists. Placing Sell/Close Order.')
                        # ---------------
                        self.log(f'SELL CREATE (Close Position), {current_close:.2f}, Size: {self.position.size}')
                        self.order = self.close() # 平掉所有多头仓位
                    else:
                        self.log(f'{current_date} - Sell Signal ({signal}) but no current position. Doing nothing.')
                
                # 可以在这里添加其他信号处理逻辑，例如 signal == 0 (无操作)
                # else: # signal == 0 or other conditions
                #     self.log(f'{current_date} - No trade signal (Signal: {signal}). Holding.')
                
            else:
                # 使用策略内置的指标生成信号 (这部分逻辑已被简化或注释，因为当前重点是 use_custom_signals)
                # ... (原有的基于指标的买卖逻辑，当前 focus 下可以暂时忽略) ...
                pass


    def stop(self):
        """策略结束时调用"""
        mode = "自定义信号" if self.params.use_custom_signals else "内置指标"
        self.log(f'策略模式: {mode} | 运行结束。最终组合价值: {self.broker.getvalue():.2f}')


# 回测引擎类
class BTBacktestEngine:
    """基于 Backtrader 的回测引擎类"""
    
    # Define custom lines statically here to avoid modification issues
    _EXPECTED_CUSTOM_LINES = (
        'volume_ma5', 'volume_ma10', 'ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'ma250',
        'rsi', 'kdj_k', 'kdj_d', 'kdj_j', 'macd', 'macd_signal', 'macd_hist',
        'boll_upper', 'boll_middle', 'boll_lower', 'atr', 'adx', 'signal',
        'long_stop_loss', 'short_stop_loss'
    )

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
    
    def load_data(self, symbol, custom_lines):
        """
        从数据库加载历史数据并转换为 Backtrader 数据源
        
        Parameters
        ----------
        symbol : str
            交易品种代码
        custom_lines : tuple
            A tuple containing the names of the custom lines expected by the PandasDataExtend class.
        
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
            
            # --- Revised Column Handling --- 
            # 1. Ensure temporary SL columns exist (assuming bb_lower/upper exist)
            if 'bb_lower' in df.columns and 'bb_upper' in df.columns:
                df['_long_sl'] = df['bb_lower']
                df['_short_sl'] = df['bb_upper']
            else:
                logger.warning(f"Columns 'bb_lower' or 'bb_upper' not found for {symbol}. Stop loss lines will be missing.")
                # Create dummy columns if needed by lines definition
                if 'long_stop_loss' in custom_lines:
                    logger.warning(f"Column 'bb_lower' not found for {symbol}. Creating dummy 'long_stop_loss'.")
                    df['_long_sl'] = float('nan')
                if 'short_stop_loss' in custom_lines:
                    logger.warning(f"Column 'bb_upper' not found for {symbol}. Creating dummy 'short_stop_loss'.") 
                    df['_short_sl'] = float('nan')

            # 2. Define ALL columns Backtrader expects (Standard + Custom Lines)
            bt_standard_cols = ['open', 'high', 'low', 'close', 'volume']
            
            # Use the passed custom lines parameter
            bt_custom_lines = list(custom_lines)
 
            all_required_bt_cols = bt_standard_cols + bt_custom_lines
            
            # 3. Define mapping from SOURCE column names in df to TARGET BT names
            #    Only include columns that NEED renaming or are custom lines.
            #    Standard OHLCV are assumed to have correct names already.
            rename_map = {
                # Source Name in df : Target Name in all_required_bt_cols
                'fusion_signal': 'signal', # Map the calculated signal
                '_long_sl': 'long_stop_loss', # Map the temp SL column
                '_short_sl': 'short_stop_loss',# Map the temp SL column
                
                # --- Add mappings for other custom lines if their names in df differ --- 
                # Example: if df has 'rsi_14' but lines has 'rsi'
                # 'rsi_14': 'rsi',
                # Example: if df has 'bb_upper' but lines has 'boll_upper'
                'bb_upper': 'boll_upper', 
                'bb_middle': 'boll_middle',
                'bb_lower': 'boll_lower', 
                'ATR': 'atr', # Assuming df has 'ATR'
                'ADX': 'adx', # Assuming df has 'ADX'
                # Assuming volume_ma5, volume_ma10, ma5, ma10 etc. already exist with correct names
            }
            
            # 4. Identify source columns needed: Standard OHLCV + keys from rename_map + any custom lines NOT in rename_map
            source_cols_to_select = list(bt_standard_cols) # Start with standard cols
            source_cols_to_select.extend(rename_map.keys()) # Add source names that will be renamed
            # Add custom lines whose names are already correct in df
            for line in bt_custom_lines: # Iterate through custom lines expected by BT
                # Check if this line name is NOT a target name in the rename map
                if line not in rename_map.values() and line in df.columns:
                    source_cols_to_select.append(line) # Add it to the list of source columns to select
            
            # Ensure unique columns and check if they exist in df
            source_cols_to_select = list(dict.fromkeys(source_cols_to_select)) # Keep order, remove duplicates
            missing_source_cols = [col for col in source_cols_to_select if col not in df.columns]
            if missing_source_cols:
                  # If a rename source is missing, remove it from map and selection
                  cleaned_source_cols = []
                  for col in source_cols_to_select:
                      if col in df.columns:
                          cleaned_source_cols.append(col)
                      else:
                          logger.warning(f"Required source column '{col}' not found in DataFrame for {symbol}. It will be excluded.")
                          # Remove corresponding entry from rename_map if necessary
                          keys_to_remove = [k for k, v in rename_map.items() if k == col]
                          for k_rem in keys_to_remove: del rename_map[k_rem]
                  source_cols_to_select = cleaned_source_cols
            
            # 5. Select only the necessary source columns
            df_selected = df[source_cols_to_select]
            
            # 6. Rename the selected columns to match Backtrader's expectations
            df_renamed = df_selected.rename(columns=rename_map)
            
            # 7. Reindex to ensure exact column set and order, filling missing with NaN
            #    This guarantees the DataFrame structure matches PandasDataExtend
            df_final = df_renamed.reindex(columns=all_required_bt_cols, fill_value=float('nan'))
            print(f"DataFrame 列名: {df_final.columns.tolist()}")
            print(f"DataFrame 前5行: \n{df_final.head()}")
            # 创建 Backtrader 数据源 (使用 df_final)
            data = PandasDataExtend(
                dataname=df_final, # 使用整理好的 DataFrame
                datetime=None,  # 使用索引
                open='open', high='high', low='low', close='close', volume='volume', # 显式指定标准列以防万一
            )
            self.cerebro.adddata(data, name=symbol)
            self.log(f'成功从数据库加载 {symbol} 的历史数据，共 {len(df_final)} 条记录')
            return data
        except Exception as e:
            logger.error(f"从数据库加载历史数据失败: {e}")
            import traceback
            traceback.print_exc()
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
        
        Notes
        -----
        1. 该函数会将交易品种代码列表中的每只 ETF 加载到 Backtrader 数据源中，并添加到 cerebro 实例中。
        2. 然后，会将策略 FusionStrategyBT 添加到 cerebro 实例中，并将其参数从 self.strategy_config 中读取出来。
        3. 接下来，会运行回测，并将回测结果保存到 self.results 中。
        4. 最后，会将回测结果返回。
        """
        
        logger.info(f"开始回测，交易品种: {symbols}")
        
        # 添加数据
        for symbol in symbols:
            data = self.load_data(symbol, self._EXPECTED_CUSTOM_LINES)
            if data is not None:
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
            'avg_trade_pnl': trades.get('pnl', {}).get('net', {}).get('average', 0),
            'max_trade_pnl': trades.get('won', {}).get('pnl', {}).get('max', 0),
            'min_trade_pnl': trades.get('lost', {}).get('pnl', {}).get('max', 0),
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
        
        # 创建一个唯一的输出目录，用于存储本次回测的结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 可以在文件夹名称中加入策略参数等信息，以便更好地识别
        run_output_dir = os.path.join(output_dir, f"run_{timestamp}")
        
        # 保存结果到新的唯一目录
        engine.save_results(run_output_dir)
        
        return results
    except Exception as e:
        logger.error(f"运行回测失败: {e}", exc_info=True)
        return {}


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # 运行回测
    config_file = 'config/config.yaml'
    symbols = ['159998']  
    
    results = run_backtest(
        config_file=config_file,
        symbols=symbols,
        start_date='2022-01-01',
        end_date='2024-12-31',
        output_dir='backtest_results'
    )
    
    print(f"回测完成，总收益率: {results['performance']['total_return']:.2f}%")
