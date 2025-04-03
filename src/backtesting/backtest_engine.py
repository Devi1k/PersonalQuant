#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测引擎
用于策略回测和性能评估
"""

import os
import pandas as pd
import numpy as np
import logging
import yaml
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import json

# 导入自定义模块
from src.data.data_processor import DataProcessor
from src.strategy.trend_strategy import TrendStrategy

# 设置日志
logger = logging.getLogger(__name__)

class BacktestEngine:
    """回测引擎类"""
    
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
        
        # 策略
        self.strategy = TrendStrategy(self.config)
        
        # 回测结果
        self.results = {
            'equity_curve': None,
            'trades': [],
            'performance': {}
        }
        
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
    
    def load_data(self, symbol):
        """
        加载历史数据
        
        Parameters
        ----------
        symbol : str
            交易品种代码
            
        Returns
        -------
        pandas.DataFrame
            历史数据
        """
        # 构建数据文件路径
        file_path = Path(f"data/raw/{symbol}_{self.start_date.replace('-', '')}_{self.end_date.replace('-', '')}.csv")
        
        if not file_path.exists():
            logger.error(f"数据文件不存在: {file_path}")
            return pd.DataFrame()
        
        try:
            # 加载数据
            df = pd.read_csv(file_path)
            
            # 处理数据
            df = self.data_processor.process_etf_history(df)
            
            # 计算技术指标，传入配置参数
            df = self.data_processor.calculate_technical_indicators(df, self.config)
            
            # 过滤日期范围
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
            
            logger.info(f"成功加载并处理 {symbol} 的历史数据，共 {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"加载历史数据失败: {e}")
            return pd.DataFrame()
    
    def generate_signals(self, df):
        """
        生成交易信号
        
        Parameters
        ----------
        df : pandas.DataFrame
            历史数据
            
        Returns
        -------
        pandas.DataFrame
            添加了交易信号的数据
        """
        if df.empty:
            logger.warning("输入的历史数据为空")
            return df
        
        logger.info("开始生成交易信号")
        
        # 应用布林带突破策略
        df = self.strategy.bollinger_bands_breakout(df)
        
        # 应用MACD长短周期收敛发散策略
        df = self.strategy.macd_convergence_divergence(df)
        
        # 应用EMA通道反转策略
        df = self.strategy.ema_channel_reversal(df)
        
        # 应用量价齐升确认策略
        df = self.strategy.volume_price_confirmation(df)
        
        # 组合信号
        df = self.strategy.combine_signals(df)
        
        logger.info("交易信号生成完成")
        
        return df
    
    def run_backtest(self, symbol):
        """
        运行回测
        
        Parameters
        ----------
        symbol : str
            交易品种代码
            
        Returns
        -------
        dict
            回测结果
        """
        logger.info(f"开始回测 {symbol}")
        
        # 加载数据
        df = self.load_data(symbol)
        if df.empty:
            logger.error("回测失败：无法加载历史数据")
            return self.results
        
        # 生成信号
        df = self.generate_signals(df)
        
        # 模拟交易
        self._simulate_trades(df, symbol)
        
        # 计算回测结果
        self._calculate_performance()
        
        logger.info(f"{symbol} 回测完成")
        
        return self.results
    
    def _simulate_trades(self, df, symbol):
        """
        模拟交易
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含交易信号的数据
        symbol : str
            交易品种代码
        """
        if df.empty or 'final_signal_execution' not in df.columns:
            logger.warning("无法模拟交易：数据为空或没有交易信号")
            return
        
        logger.info("开始模拟交易")
        
        # 初始化资金和持仓
        capital = self.initial_capital
        position = 0
        trades = []
        equity_curve = []
        
        # 遍历每个交易日
        for i, row in df.iterrows():
            date = row['date']
            close = row['close']
            signal = row['final_signal_execution']
            
            # 计算当前市值
            market_value = capital + position * close
            equity_curve.append({
                'date': date,
                'capital': capital,
                'position': position,
                'close': close,
                'market_value': market_value
            })
            
            # 处理交易信号
            if signal == 1 and position <= 0:  # 买入信号
                # 计算买入数量（全仓买入）
                buy_price = close * (1 + self.slippage)  # 考虑滑点
                shares = int(capital / buy_price)
                cost = shares * buy_price
                commission_fee = cost * self.commission  # 手续费
                
                if shares > 0:
                    # 更新资金和持仓
                    capital -= (cost + commission_fee)
                    position += shares
                    
                    # 记录交易
                    trades.append({
                        'date': date,
                        'type': 'buy',
                        'price': buy_price,
                        'shares': shares,
                        'cost': cost,
                        'commission': commission_fee
                    })
                    
                    logger.info(f"买入 {shares} 股 {symbol}，价格 {buy_price:.2f}，成本 {cost:.2f}，手续费 {commission_fee:.2f}")
            
            elif signal == -1 and position > 0:  # 卖出信号
                # 计算卖出金额
                sell_price = close * (1 - self.slippage)  # 考虑滑点
                proceeds = position * sell_price
                commission_fee = proceeds * self.commission  # 手续费
                
                # 更新资金和持仓
                capital += (proceeds - commission_fee)
                
                # 记录交易
                trades.append({
                    'date': date,
                    'type': 'sell',
                    'price': sell_price,
                    'shares': position,
                    'proceeds': proceeds,
                    'commission': commission_fee
                })
                
                logger.info(f"卖出 {position} 股 {symbol}，价格 {sell_price:.2f}，收入 {proceeds:.2f}，手续费 {commission_fee:.2f}")
                
                # 清空持仓
                position = 0
        
        # 最后一天，如果还有持仓，强制平仓
        if position > 0:
            last_date = df['date'].iloc[-1]
            last_close = df['close'].iloc[-1]
            sell_price = last_close * (1 - self.slippage)
            proceeds = position * sell_price
            commission_fee = proceeds * self.commission
            
            capital += (proceeds - commission_fee)
            
            trades.append({
                'date': last_date,
                'type': 'sell',
                'price': sell_price,
                'shares': position,
                'proceeds': proceeds,
                'commission': commission_fee,
                'note': '回测结束平仓'
            })
            
            logger.info(f"回测结束平仓：卖出 {position} 股 {symbol}，价格 {sell_price:.2f}，收入 {proceeds:.2f}，手续费 {commission_fee:.2f}")
            
            # 更新最后一天的市值
            equity_curve[-1]['capital'] = capital
            equity_curve[-1]['position'] = 0
            equity_curve[-1]['market_value'] = capital
        
        # 保存交易记录和权益曲线
        self.results['trades'] = trades
        self.results['equity_curve'] = pd.DataFrame(equity_curve)
        
        logger.info(f"模拟交易完成，共执行 {len(trades)} 笔交易")
    
    def _calculate_performance(self):
        """计算回测性能指标"""
        if self.results['equity_curve'] is None:
            logger.warning("无法计算性能指标：权益曲线为空")
            return
        
        logger.info("开始计算回测性能指标")
        
        equity_curve = self.results['equity_curve']
        
        # 计算收益率
        initial_value = self.initial_capital
        final_value = equity_curve['market_value'].iloc[-1]
        total_return = (final_value / initial_value - 1) * 100
        
        # 计算年化收益率
        days = (equity_curve['date'].iloc[-1] - equity_curve['date'].iloc[0]).days
        annual_return = (final_value / initial_value) ** (365 / days) - 1
        annual_return_pct = annual_return * 100
        
        # 计算最大回撤
        equity_curve['cummax'] = equity_curve['market_value'].cummax()
        equity_curve['drawdown'] = (equity_curve['market_value'] / equity_curve['cummax'] - 1) * 100
        max_drawdown = equity_curve['drawdown'].min()
        
        # 计算夏普比率
        equity_curve['daily_return'] = equity_curve['market_value'].pct_change()
        daily_returns = equity_curve['daily_return'].dropna()
        risk_free_rate = 0.02 / 252  # 假设无风险年化收益率为2%
        excess_returns = daily_returns - risk_free_rate
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
        
        # 计算胜率
        trades = self.results['trades']
        if trades:
            buy_trades = [t for t in trades if t['type'] == 'buy']
            sell_trades = [t for t in trades if t['type'] == 'sell']
            
            profits = []
            for i in range(min(len(buy_trades), len(sell_trades))):
                buy = buy_trades[i]
                sell = sell_trades[i]
                profit = (sell['price'] - buy['price']) / buy['price'] * 100
                profits.append(profit)
            
            win_trades = sum(1 for p in profits if p > 0)
            win_rate = win_trades / len(profits) * 100 if profits else 0
        else:
            win_rate = 0
        
        # 保存性能指标
        performance = {
            'initial_capital': initial_value,
            'final_capital': final_value,
            'total_return_pct': total_return,
            'annual_return_pct': annual_return_pct,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate_pct': win_rate,
            'trade_count': len(trades)
        }
        
        self.results['performance'] = performance
        
        logger.info(f"回测性能指标计算完成：总收益率 {total_return:.2f}%，年化收益率 {annual_return_pct:.2f}%，最大回撤 {max_drawdown:.2f}%，夏普比率 {sharpe_ratio:.2f}，胜率 {win_rate:.2f}%")
    
    def plot_results(self, save_path=None):
        """
        绘制回测结果图表
        
        Parameters
        ----------
        save_path : str, default None
            图表保存路径，如果为None则显示图表
        """
        if self.results['equity_curve'] is None:
            logger.warning("无法绘制图表：权益曲线为空")
            return
        
        logger.info("开始绘制回测结果图表")
        
        equity_curve = self.results['equity_curve']
        performance = self.results['performance']
        
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # 绘制权益曲线
        axes[0].plot(equity_curve['date'], equity_curve['market_value'], label='策略市值')
        axes[0].set_title('回测结果')
        axes[0].set_ylabel('市值')
        axes[0].grid(True)
        axes[0].legend()
        
        # 添加性能指标文本
        text = f"总收益率: {performance['total_return_pct']:.2f}%\n" \
               f"年化收益率: {performance['annual_return_pct']:.2f}%\n" \
               f"最大回撤: {performance['max_drawdown_pct']:.2f}%\n" \
               f"夏普比率: {performance['sharpe_ratio']:.2f}\n" \
               f"胜率: {performance['win_rate_pct']:.2f}%\n" \
               f"交易次数: {performance['trade_count']}"
        
        axes[0].text(0.02, 0.95, text, transform=axes[0].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 绘制回撤曲线
        axes[1].fill_between(equity_curve['date'], equity_curve['drawdown'], 0, 
                            color='red', alpha=0.3)
        axes[1].set_title('回撤')
        axes[1].set_ylabel('回撤 (%)')
        axes[1].set_ylim(equity_curve['drawdown'].min() * 1.1, 5)
        axes[1].grid(True)
        
        plt.tight_layout()
        
        # 保存或显示图表
        if save_path:
            plt.savefig(save_path)
            logger.info(f"回测结果图表已保存至 {save_path}")
        else:
            plt.show()
            logger.info("回测结果图表已显示")
    
    def save_results(self, symbol, save_dir='results/backtest'):
        """
        保存回测结果
        
        Parameters
        ----------
        symbol : str
            交易品种代码
        save_dir : str, default 'results/backtest'
            结果保存目录
        """
        if self.results['equity_curve'] is None:
            logger.warning("无法保存结果：回测结果为空")
            return
        
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 构建文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{symbol}_{self.start_date.replace('-', '')}_{self.end_date.replace('-', '')}_{timestamp}"
        
        # 保存权益曲线
        equity_curve_file = os.path.join(save_dir, f"{base_name}_equity.csv")
        self.results['equity_curve'].to_csv(equity_curve_file, index=False)
        
        # 保存交易记录
        trades_file = os.path.join(save_dir, f"{base_name}_trades.csv")
        pd.DataFrame(self.results['trades']).to_csv(trades_file, index=False)
        
        # 保存性能指标
        performance_file = os.path.join(save_dir, f"{base_name}_performance.json")
        with open(performance_file, 'w', encoding='utf-8') as f:
            json.dump(self.results['performance'], f, indent=4)
        
        # 保存图表
        plot_file = os.path.join(save_dir, f"{base_name}_plot.png")
        self.plot_results(save_path=plot_file)
        
        logger.info(f"回测结果已保存至 {save_dir} 目录")


def run_backtest(config_file, start_date=None, end_date=None):
    """
    运行回测的入口函数
    
    Parameters
    ----------
    config_file : str
        配置文件路径
    start_date : str, default None
        回测开始日期，格式YYYY-MM-DD，如果为None则使用配置文件中的设置
    end_date : str, default None
        回测结束日期，格式YYYY-MM-DD，如果为None则使用配置文件中的设置
    """
    logger.info(f"开始回测，配置文件: {config_file}")
    
    try:
        # 创建回测引擎
        engine = BacktestEngine(config_file)
        
        # 如果提供了日期参数，覆盖配置文件中的设置
        if start_date:
            engine.start_date = start_date
        if end_date:
            engine.end_date = end_date
        
        # 加载配置中的ETF列表
        etf_list_file = engine.config['data_source'].get('etf_list_file', 'data/raw/etf_list.csv')
        if os.path.exists(etf_list_file):
            etf_list = pd.read_csv(etf_list_file)
            symbols = etf_list['symbol'].tolist()
        else:
            # 如果ETF列表文件不存在，使用默认的ETF
            symbols = ['159915']  # 创业板ETF
            logger.warning(f"ETF列表文件 {etf_list_file} 不存在，使用默认ETF: {symbols}")
        
        # 对每个ETF运行回测
        for symbol in symbols:
            logger.info(f"开始回测 {symbol}")
            engine.run_backtest(symbol)
            engine.save_results(symbol)
        
        logger.info("回测完成")
        
    except Exception as e:
        logger.exception(f"回测过程中出现错误: {e}")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行回测
    run_backtest('config/config.yaml')