import pandas as pd
import numpy as np
import logging
import talib

from .trend_strategy import TrendStrategy
from .swing_strategy import SwingStrategy

logger = logging.getLogger(__name__)

class FusionStrategy:
    """
    策略融合机制
    - 根据市场状态动态调整趋势策略与波段策略的权重
    - 实现趋势与波段信号的协同确认
    - 支持市场环境自适应切换
    - 市场状态识别与分类（趋势/震荡/剧烈波动）
    """

    def __init__(self, config=None):
        self.config = config or {}

        # 初始化趋势策略和波段策略
        self.trend_strategy = TrendStrategy(config.get('trend', {}))
        self.swing_strategy = SwingStrategy(config.get('swing', {}))

        # 默认权重
        self.trend_weight = self.config.get('trend_weight', 0.6)
        self.swing_weight = self.config.get('swing_weight', 0.4)

        # 市场状态阈值
        self.adx_threshold = self.config.get('adx_threshold', 25)
        self.bandwidth_threshold = self.config.get('bandwidth_threshold', 0.05)
        self.atr_threshold = self.config.get('atr_threshold', 0.02)
        self.ma_slope_threshold = self.config.get('ma_slope_threshold', 0.01)
        
        # 协同确认增强系数
        self.confirmation_multiplier = self.config.get('confirmation_multiplier', 1.5)
        
        # 自适应阈值调整参数
        self.adaptive_threshold = self.config.get('adaptive_threshold', True)
        self.threshold_lookback = self.config.get('threshold_lookback', 20)

    def calculate_indicators(self, df):
        """
        计算市场状态识别所需的技术指标
        """
        if 'ADX' not in df.columns:
            df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        if 'bollinger_bandwidth' not in df.columns:
            upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
            df['bollinger_bandwidth'] = (upper - lower) / middle
        
        if 'ATR_pct' not in df.columns:
            df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            df['ATR_pct'] = df['ATR'] / df['close']
        
        # 计算移动平均线斜率
        if 'MA50' not in df.columns:
            df['MA50'] = talib.SMA(df['close'], timeperiod=50)
        
        if 'MA50_slope' not in df.columns:
            df['MA50_slope'] = df['MA50'].diff(5) / df['MA50'].shift(5)
        
        # 计算长期200日移动平均线（MA200），用于宏观趋势过滤
        if 'MA200' not in df.columns:
            df['MA200'] = talib.SMA(df['close'], timeperiod=200)
        
        return df
    
    def update_thresholds(self, df):
        """
        根据历史数据动态调整阈值
        """
        if not self.adaptive_threshold or len(df) < self.threshold_lookback:
            return
        
        # 使用过去N天的数据动态调整阈值
        lookback_data = df.iloc[-self.threshold_lookback:]
        
        # ADX阈值调整：使用过去N天ADX的中位数作为基准
        adx_median = lookback_data['ADX'].median()
        self.adx_threshold = max(20, min(30, adx_median * 0.9))
        
        # 布林带宽度阈值调整
        bw_median = lookback_data['bollinger_bandwidth'].median()
        self.bandwidth_threshold = max(0.03, min(0.08, bw_median * 0.8))
        
        # ATR百分比阈值调整
        atr_median = lookback_data['ATR_pct'].median()
        self.atr_threshold = max(0.01, min(0.04, atr_median * 1.5))
        
        logger.info(f"动态调整阈值: ADX={self.adx_threshold:.2f}, 布林带宽度={self.bandwidth_threshold:.4f}, ATR%={self.atr_threshold:.4f}")

    def identify_market_state(self, df):
        """
        识别市场状态：趋势、震荡、剧烈波动
        输入DataFrame需包含价格数据，函数会自动计算所需指标
        返回：
            'trend', 'range', 'volatile' 三种状态之一
        """
        # 确保所有必要的指标都已计算
        df = self.calculate_indicators(df)
        
        # 动态调整阈值
        self.update_thresholds(df)
        
        # 获取最新指标值
        adx = df['ADX'].iloc[-1]
        bandwidth = df['bollinger_bandwidth'].iloc[-1]
        atr_pct = df['ATR_pct'].iloc[-1]
        ma_slope = df['MA50_slope'].iloc[-1]
        
        # 趋势市场判断：ADX高 + 布林带宽度大 + MA斜率明显
        if (adx > self.adx_threshold and
            bandwidth > self.bandwidth_threshold and
            abs(ma_slope) > self.ma_slope_threshold):
            market_state = 'trend'
            # 进一步区分上升趋势和下降趋势
            trend_direction = 'up' if ma_slope > 0 else 'down'
            market_state = f"{market_state}_{trend_direction}"
        # 剧烈波动市场：ATR百分比高
        elif atr_pct > self.atr_threshold:
            market_state = 'volatile'
        # 震荡市场：其他情况
        else:
            market_state = 'range'
        
        logger.info(f"市场状态识别: ADX={adx:.2f}, 布林带宽度={bandwidth:.4f}, ATR%={atr_pct:.4f}, MA斜率={ma_slope:.4f} -> 状态: {market_state}")
        return market_state

    def adaptive_weights(self, market_state):
        """
        根据市场状态调整趋势和波段策略的权重
        """
        # 根据市场状态细分调整权重
        if market_state == 'trend_up':
            # 上升趋势市场：趋势策略权重高
            trend_w = 0.85
            swing_w = 0.15
            logger.info("上升趋势市场：大幅提高趋势策略权重")
        elif market_state == 'trend_down':
            # 下降趋势市场：趋势策略权重高，但略低于上升趋势
            trend_w = 0.75
            swing_w = 0.25
            logger.info("下降趋势市场：提高趋势策略权重")
        elif market_state == 'range':
            # 震荡市场：波段策略权重高
            trend_w = 0.25
            swing_w = 0.75
            logger.info("震荡市场：大幅提高波段策略权重")
        elif market_state == 'volatile':
            # 剧烈波动市场：均衡配置，略偏向波段策略
            trend_w = 0.4
            swing_w = 0.6
            logger.info("剧烈波动市场：均衡配置策略权重，略偏向波段策略")
        else:
            # 默认配置
            trend_w = self.trend_weight
            swing_w = self.swing_weight
            logger.info("使用默认策略权重配置")

        logger.info(f"根据市场状态调整权重: 趋势策略={trend_w:.2f}, 波段策略={swing_w:.2f}")
        return trend_w, swing_w

    def calculate_weekly_data(self, df):
        """
        从日线数据计算周线数据
        
        Parameters
        ----------
        df : pandas.DataFrame
            包含日线价格数据的DataFrame
            
        Returns
        -------
        pandas.DataFrame
            周线数据DataFrame，包含date, close, high, low等列
        """
        if df.empty:
            logger.warning("输入的日线数据为空，无法计算周线数据")
            return pd.DataFrame()
            
        # 确保日期列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
            
        # 创建周开始日期列，用于分组
        df['week_start'] = df['date'].apply(lambda x: x - pd.Timedelta(days=x.weekday()))
        
        # 按周分组并聚合数据
        weekly_df = df.groupby('week_start').agg({
            'open': 'first',          # 周一开盘价
            'high': 'max',            # 周内最高价
            'low': 'min',             # 周内最低价
            'close': 'last',          # 周五收盘价
            'volume': 'sum'           # 周成交量总和
        }).reset_index()
        
        # 将week_start列重命名为date
        weekly_df.rename(columns={'week_start': 'date'}, inplace=True)
        
        logger.info(f"成功计算周线数据，共 {len(weekly_df)} 条记录")
        return weekly_df

    def combine_strategies(self, df):
        """
        融合趋势策略与波段策略信号
        输入：
            df: 包含价格数据的DataFrame
        返回：
            融合后的DataFrame，含最终信号
        """
        # 确保所有必要的指标都已计算
        df = self.calculate_indicators(df)
        
        # 计算市场状态
        market_state = self.identify_market_state(df)
        trend_w, swing_w = self.adaptive_weights(market_state)

        # 计算周线数据，用于波段策略
        weekly_df = self.calculate_weekly_data(df)

        # 运行趋势策略
        trend_df = self.trend_strategy.combine_signals(df)
        

        # 运行波段策略
        swing_df = self.swing_strategy.combine_signals(df, weekly_df)

        # 融合信号
        combined_df = df.copy()
        combined_df['trend_signal'] = trend_df['final_signal']
        combined_df['swing_signal'] = swing_df['final_signal']
        combined_df['market_state'] = market_state

        # 信号协同确认：当两个策略信号一致时，增强信号强度
        combined_df['signal_agreement'] = (
            (combined_df['trend_signal'] > 0) & (combined_df['swing_signal'] > 0) |
            (combined_df['trend_signal'] < 0) & (combined_df['swing_signal'] < 0)
        )
        
        # 基础加权融合
        # combined_df['fusion_score'] = trend_w * combined_df['trend_signal'] + swing_w * combined_df['swing_signal']
        
        # 协同确认增强
        # combined_df.loc[combined_df['signal_agreement'], 'fusion_score'] *= self.confirmation_multiplier
        
        # 生成最终信号改为仅使用趋势策略
        combined_df['fusion_signal'] = trend_df['final_signal']
        
        # === 以下阈值及分级信号逻辑暂时停用 ===
        # 强信号阈值
        # buy_threshold = 0.5
        # sell_threshold = -0.5
        
        # 根据市场状态调整信号阈值
        # if 'trend' in market_state:
        #     buy_threshold = 0.4 if 'up' in market_state else 0.6
        #     sell_threshold = -0.6 if 'up' in market_state else -0.4
        # elif market_state == 'range':
        #     buy_threshold = 0.6
        #     sell_threshold = -0.4
        
        # 应用调整后的阈值
        # combined_df.loc[combined_df['fusion_score'] > buy_threshold, 'fusion_signal'] = 1
        # combined_df.loc[combined_df['fusion_score'] < sell_threshold, 'fusion_signal'] = -1
        
        # 试探性信号（0.5强度）
        # combined_df.loc[(combined_df['fusion_score'] <= buy_threshold) &
        #                 (combined_df['fusion_score'] > buy_threshold * 0.6), 'fusion_signal'] = 0.5
        # combined_df.loc[(combined_df['fusion_score'] >= sell_threshold) &
        #                 (combined_df['fusion_score'] < sell_threshold * 0.6), 'fusion_signal'] = -0.5

        # ====================  MA200 趋势过滤  ====================
        # 只有当收盘价位于 MA200 之上时，才允许任何买入信号生效。
        # 当价格低于 MA200 时，所有买入及试探性买入信号被强制设为 0。
        below_ma200 = combined_df['close'] <= combined_df['MA200']
        combined_df.loc[below_ma200 & (combined_df['fusion_signal'] > 0), 'fusion_signal'] = 0
        combined_df.loc[below_ma200 & (combined_df['fusion_signal'] == 0.5), 'fusion_signal'] = 0

        # 记录信号统计
        buy_signals = (combined_df['fusion_signal'] > 0).sum()
        sell_signals = (combined_df['fusion_signal'] < 0).sum()
        agreement_signals = combined_df['signal_agreement'].sum()
        
        logger.info(f"融合策略信号生成完成，买入信号数: {buy_signals}, 卖出信号数: {sell_signals}")
        logger.info(f"其中完全买入信号: {(combined_df['fusion_signal'] == 1).sum()}个, "
                   f"试探性买入信号: {(combined_df['fusion_signal'] == 0.5).sum()}个, "
                   f"完全卖出信号: {(combined_df['fusion_signal'] == -1).sum()}个, "
                   f"减仓信号: {(combined_df['fusion_signal'] == -0.5).sum()}个")
        logger.info(f"趋势与波段策略协同确认信号数: {agreement_signals}")

        return combined_df
        
    def get_market_analysis(self, df):
        """
        获取市场状态分析报告
        """
        df = self.calculate_indicators(df)
        market_state = self.identify_market_state(df)
        
        # 获取最新指标值
        adx = df['ADX'].iloc[-1]
        bandwidth = df['bollinger_bandwidth'].iloc[-1]
        atr_pct = df['ATR_pct'].iloc[-1]
        ma_slope = df['MA50_slope'].iloc[-1]
        
        analysis = {
            'market_state': market_state,
            'indicators': {
                'adx': adx,
                'bollinger_bandwidth': bandwidth,
                'atr_pct': atr_pct,
                'ma_slope': ma_slope
            },
            'thresholds': {
                'adx_threshold': self.adx_threshold,
                'bandwidth_threshold': self.bandwidth_threshold,
                'atr_threshold': self.atr_threshold,
                'ma_slope_threshold': self.ma_slope_threshold
            },
            'weights': {
                'trend_weight': self.trend_weight,
                'swing_weight': self.swing_weight
            }
        }
        
        return analysis