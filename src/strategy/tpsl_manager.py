# -*- coding: utf-8 -*-
"""
@author: Kilo Code
@description:
This module provides a Trailing Stop Loss / Take Profit (TPSL) Manager
for trading strategies. It evaluates various exit conditions based on
market data and position details.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Tuple, Dict, Any

class TPSLManager:
    """
    Manages and checks various stop-loss and take-profit conditions for a given position.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the TPSLManager with a configuration dictionary.

        Args:
            config (Dict[str, Any]): A dictionary containing the settings for
                                      various stop-loss and take-profit rules.
                                      Example:
                                      {
                                          "stop_loss": {
                                              "percentage": {"active": True, "value": 8.0},
                                              "ma_support": {"active": True, "ma_period": 20},
                                              "entry_low": {"active": True},
                                              "trend_break": {"active": True, "short_ma": 20, "long_ma": 60}
                                          },
                                          "take_profit": {}
                                      }
        """
        self.config = config.get("stop_loss", {})

    def check_exit_conditions(self, position: Any, ohlcv_df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Checks if any of the active exit conditions are met for the given position.

        The rules are checked in a specific order of precedence. Once a condition
        is met, the function returns immediately.

        Args:
            position (Any): An object or dictionary containing position details,
                            such as `entry_price` and `entry_candle_low`.
            ohlcv_df (pd.DataFrame): A DataFrame with OHLCV data, indexed by timestamp.
                                     It should contain at least 'open', 'high', 'low', 'close' columns.

        Returns:
            Tuple[bool, str]: A tuple containing:
                              - should_exit (bool): True if an exit condition is met, False otherwise.
                              - reason (str): A string describing the triggered rule.
        """
        if ohlcv_df.empty:
            return False, ""

        # --- Technical Stop-Loss Rules ---

        # Rule A1: MA Support Breach
        ma_support_config = self.config.get("ma_support", {})
        if ma_support_config.get("active", False):
            ma_period = ma_support_config.get("ma_period")
            if ma_period:
                ma = ta.sma(ohlcv_df['close'], length=ma_period)
                if not ma.empty:
                    latest_close = ohlcv_df['close'].iloc[-1]
                    ma_value = ma.iloc[-1]
                    if latest_close < ma_value:
                        reason = f"STOP_LOSS: MA Support Breach (Close < MA_{ma_period})"
                        return True, reason

        # Rule A2: Entry Pattern Low Breach
        entry_low_config = self.config.get("entry_low", {})
        if entry_low_config.get("active", False):
            latest_low = ohlcv_df['low'].iloc[-1]
            if latest_low < position.entry_candle_low:
                reason = "STOP_LOSS: Entry Pattern Low Breach"
                return True, reason

        # Rule A3: Trend Structure Reversal (Death Cross)
        trend_break_config = self.config.get("trend_break", {})
        if trend_break_config.get("active", False):
            short_ma_period = trend_break_config.get("short_ma")
            long_ma_period = trend_break_config.get("long_ma")
            if short_ma_period and long_ma_period and len(ohlcv_df) > long_ma_period:
                short_ma = ta.sma(ohlcv_df['close'], length=short_ma_period)
                long_ma = ta.sma(ohlcv_df['close'], length=long_ma_period)
                if short_ma.iloc[-2] > long_ma.iloc[-2] and short_ma.iloc[-1] < long_ma.iloc[-1]:
                    reason = f"STOP_LOSS: Trend Reversal (MA{short_ma_period} crossed below MA{long_ma_period})"
                    return True, reason

        # --- Absolute Stop-Loss Rules ---

        # Rule B1: Percentage Loss
        percentage_config = self.config.get("percentage", {})
        if percentage_config.get("active", False):
            loss_percentage = percentage_config.get("value")
            if loss_percentage is not None:
                current_low = ohlcv_df['low'].iloc[-1]
                pnl_percentage = (current_low / position.entry_price - 1) * 100
                if pnl_percentage <= -abs(loss_percentage):
                    reason = f"STOP_LOSS: Percentage {-abs(loss_percentage)}%"
                    return True, reason

        return False, ""