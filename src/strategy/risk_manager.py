import pandas as pd
import numpy as np
import logging
import talib

logger = logging.getLogger(__name__)

class RiskManager:
    """
    止损系统
    - 初始止损：入场价-2倍ATR
    - 盈利保护：最高点回撤3%止盈
    - 波动率自适应止损调整
    - 策略失效评估
    - 因子优化（主力资金流入、突破新高等）
    - 单日最大亏损熔断
    """

    def __init__(self, config=None):
        self.config = config or {}
        
        # 止损参数
        self.atr_multiplier = self.config.get('atr_multiplier', 2.0)
        self.trailing_drawdown = self.config.get('trailing_drawdown', 0.03)
        self.min_stop_pct = self.config.get('min_stop_pct', 0.01)
        self.max_stop_pct = self.config.get('max_stop_pct', 0.1)
        
        # 策略失效评估参数
        self.failure_max_drawdown = self.config.get('failure_max_drawdown', 0.2)
        self.failure_max_days = self.config.get('failure_max_days', 20)
        
        # 单日最大亏损熔断参数
        self.daily_max_loss_pct = self.config.get('daily_max_loss_pct', 0.05)
        self.portfolio_max_loss_pct = self.config.get('portfolio_max_loss_pct', 0.1)
        
        # 因子优化参数
        self.capital_inflow_days = self.config.get('capital_inflow_days', 5)
        self.new_high_lookback = self.config.get('new_high_lookback', 20)
        self.new_high_breakout_pct = self.config.get('new_high_breakout_pct', 0.01)

    def calculate_initial_stop(self, entry_price, atr):
        """
        计算初始止损价
        """
        stop_price = entry_price - self.atr_multiplier * atr
        logger.info(f"初始止损价: {stop_price:.2f} (入场价: {entry_price:.2f} - {self.atr_multiplier} * ATR: {atr:.2f})")
        return stop_price

    def update_trailing_stop(self, highest_price, current_price):
        """
        根据最高价动态调整止损价
        """
        trailing_stop = highest_price * (1 - self.trailing_drawdown)
        logger.info(f"动态止盈止损价: {trailing_stop:.2f} (最高价: {highest_price:.2f} - 回撤比例: {self.trailing_drawdown*100:.1f}%)")
        return trailing_stop

    def adjust_stop_by_volatility(self, stop_price, current_price, atr, entry_price):
        """
        根据波动率自适应调整止损价
        """
        stop_pct = abs(current_price - stop_price) / current_price
        atr_pct = atr / current_price

        # 限制止损幅度在合理范围
        target_stop_pct = max(self.min_stop_pct, min(self.max_stop_pct, atr_pct * self.atr_multiplier))
        adjusted_stop = entry_price * (1 - target_stop_pct)

        logger.info(f"波动率自适应止损调整: {adjusted_stop:.2f} (目标止损比例: {target_stop_pct:.2%})")
        return adjusted_stop

    def check_stop_loss(self, entry_price, current_price, atr, highest_price):
        """
        综合判断是否止损或止盈
        返回：
            -1: 触发止损
             1: 触发止盈
             0: 持仓继续
        """
        initial_stop = self.calculate_initial_stop(entry_price, atr)
        trailing_stop = self.update_trailing_stop(highest_price, current_price)
        adaptive_stop = self.adjust_stop_by_volatility(initial_stop, current_price, atr, entry_price)

        stop_level = max(initial_stop, adaptive_stop, trailing_stop)

        if current_price <= stop_level:
            logger.info(f"当前价{current_price:.2f} <= 止损价{stop_level:.2f}，触发止损")
            return -1
        elif current_price >= highest_price * (1 - self.trailing_drawdown):
            logger.info(f"当前价{current_price:.2f}接近最高价，持仓继续")
            return 0
        else:
            return 0
            
    def check_daily_circuit_breaker(self, daily_return, portfolio_return):
        """
        检查单日最大亏损熔断
        输入：
            daily_return: 单日收益率
            portfolio_return: 组合总收益率
        返回：
            True: 触发熔断
            False: 未触发熔断
        """
        # 单日亏损超过阈值
        if daily_return < -self.daily_max_loss_pct:
            logger.warning(f"单日亏损{daily_return:.2%}超过阈值{self.daily_max_loss_pct:.2%}，触发熔断")
            return True
        
        # 组合总亏损超过阈值
        if portfolio_return < -self.portfolio_max_loss_pct:
            logger.warning(f"组合总亏损{portfolio_return:.2%}超过阈值{self.portfolio_max_loss_pct:.2%}，触发熔断")
            return True
            
        return False

    def evaluate_strategy_failure(self, equity_curve):
        """
        评估策略是否失效
        输入：
            equity_curve: pd.Series，策略净值曲线
        返回：
            True 表示策略疑似失效
            False 表示策略正常
        """
        if len(equity_curve) < self.failure_max_days:
            return False

        recent_curve = equity_curve[-self.failure_max_days:]
        max_drawdown = (recent_curve.cummax() - recent_curve).max() / recent_curve.cummax().max()

        if max_drawdown > self.failure_max_drawdown:
            logger.warning(f"策略疑似失效，最近{self.failure_max_days}天最大回撤达到{max_drawdown:.2%}")
            return True
        else:
            return False
            
    def optimize_with_factors(self, df, level2_data=None):
        """
        使用因子优化止损策略
        输入：
            df: 价格数据DataFrame
            level2_data: Level2数据（可选）
        返回：
            优化后的止损系数
        """
        # 初始化优化系数
        optimization_factor = 1.0
        
        # 1. 主力资金流入天数统计
        if level2_data is not None and 'net_inflow' in level2_data.columns:
            # 计算过去N天主力资金净流入天数
            inflow_days = (level2_data['net_inflow'] > 0).rolling(self.capital_inflow_days).sum().iloc[-1]
            inflow_ratio = inflow_days / self.capital_inflow_days
            
            # 主力资金持续流入，降低止损敏感度
            if inflow_ratio > 0.7:
                optimization_factor *= 0.8
                logger.info(f"主力资金持续流入({inflow_ratio:.2%})，降低止损敏感度")
            # 主力资金持续流出，提高止损敏感度
            elif inflow_ratio < 0.3:
                optimization_factor *= 1.2
                logger.info(f"主力资金持续流出({inflow_ratio:.2%})，提高止损敏感度")
        
        # 2. 突破新高判定
        if len(df) > self.new_high_lookback:
            # 计算近期最高价
            recent_high = df['high'].rolling(self.new_high_lookback).max().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # 突破新高，降低止损敏感度
            if current_price > recent_high * (1 + self.new_high_breakout_pct):
                optimization_factor *= 0.85
                logger.info(f"价格突破{self.new_high_lookback}日新高，降低止损敏感度")
        
        # 3. 波动率环境评估
        if 'ATR' in df.columns and 'close' in df.columns:
            # 计算当前ATR占比
            current_atr_pct = df['ATR'].iloc[-1] / df['close'].iloc[-1]
            # 计算历史ATR占比分位数
            atr_pct_series = df['ATR'] / df['close']
            atr_percentile = pd.Series(atr_pct_series).rank(pct=True).iloc[-1]
            
            # 高波动环境，提高止损敏感度
            if atr_percentile > 0.8:
                optimization_factor *= 1.15
                logger.info(f"当前处于高波动环境(分位数:{atr_percentile:.2f})，提高止损敏感度")
            # 低波动环境，降低止损敏感度
            elif atr_percentile < 0.2:
                optimization_factor *= 0.9
                logger.info(f"当前处于低波动环境(分位数:{atr_percentile:.2f})，降低止损敏感度")
        
        logger.info(f"因子优化后的止损系数: {optimization_factor:.2f}")
        return optimization_factor
        
    def control_industry_diversification(self, holdings):
        """
        行业分散度控制（不超过3个行业）
        输入：
            holdings: 当前持仓字典，格式为 {symbol: weight}
        返回：
            调整后的持仓字典
        """
        if not holdings:
            return holdings
            
        # 获取每个持仓的行业信息
        # 这里需要根据实际情况获取行业信息，示例中简化处理
        industry_map = self._get_industry_map(list(holdings.keys()))
        
        # 计算每个行业的总权重
        industry_weights = {}
        for symbol, weight in holdings.items():
            industry = industry_map.get(symbol, 'Unknown')
            industry_weights[industry] = industry_weights.get(industry, 0) + weight
        
        # 如果行业数量超过3个，保留权重最高的3个行业
        if len(industry_weights) > 3:
            top_industries = sorted(industry_weights.items(), key=lambda x: x[1], reverse=True)[:3]
            top_industry_names = [ind[0] for ind in top_industries]
            
            # 调整持仓，移除非前3行业的持仓
            adjusted_holdings = {}
            for symbol, weight in holdings.items():
                industry = industry_map.get(symbol, 'Unknown')
                if industry in top_industry_names:
                    adjusted_holdings[symbol] = weight
            
            # 重新归一化权重
            total_weight = sum(adjusted_holdings.values())
            if total_weight > 0:
                for symbol in adjusted_holdings:
                    adjusted_holdings[symbol] /= total_weight
            
            logger.info(f"行业分散度控制: 从{len(industry_weights)}个行业减少到3个行业")
            return adjusted_holdings
        
        return holdings
    
    def _get_industry_map(self, symbols):
        """
        获取股票行业映射
        这里需要根据实际情况实现，可能需要从外部数据源获取
        """
        # 示例实现，实际应用中需要替换为真实数据
        industry_map = {}
        for symbol in symbols:
            # 这里应该是从数据库或API获取行业信息
            # 示例中简单地根据股票代码前两位分配行业
            if symbol.startswith('51'):
                industry_map[symbol] = 'Technology'
            elif symbol.startswith('52'):
                industry_map[symbol] = 'Finance'
            elif symbol.startswith('53'):
                industry_map[symbol] = 'Consumer'
            elif symbol.startswith('54'):
                industry_map[symbol] = 'Healthcare'
            elif symbol.startswith('55'):
                industry_map[symbol] = 'Energy'
            else:
                industry_map[symbol] = 'Other'
        return industry_map