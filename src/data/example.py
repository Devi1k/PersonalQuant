#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据模块使用示例
展示如何使用数据模块获取和处理数据
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data import (
    get_etf_history,
    get_industry_etfs,
    get_technical_indicators,
    get_industry_fund_flow,
    get_market_sentiment
)

def example_get_etf_list():
    """获取ETF列表示例"""
    print("获取行业ETF列表示例:")
    
    # 获取行业ETF列表
    industry_etfs = get_industry_etfs()
    
    if industry_etfs.empty:
        print("未获取到行业ETF列表")
        return
    
    print(f"成功获取行业ETF列表，共 {len(industry_etfs)} 个ETF")
    
    # 打印前10个ETF
    print("\n前10个行业ETF:")
    print(industry_etfs.head(10)[["code", "name", "price", "pct_change"]])

def example_get_etf_history():
    """获取ETF历史数据示例"""
    print("\n获取ETF历史数据示例:")
    
    # 设置ETF代码和日期范围
    code = "159915"  # 创业板ETF
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"获取ETF {code} 从 {start_date} 到 {end_date} 的历史数据")
    
    # 获取ETF历史数据
    df = get_etf_history(code, start_date, end_date)
    
    if df.empty:
        print(f"未获取到ETF {code} 的历史数据")
        return
    
    print(f"成功获取ETF {code} 历史数据，共 {len(df)} 条记录")
    
    # 打印最近5条记录
    print("\n最近5条记录:")
    recent_data = df.sort_values("date", ascending=False).head(5)
    print(recent_data[["date", "open", "high", "low", "close", "volume"]])
    
    # 绘制收盘价走势图
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["close"])
    plt.title(f"ETF {code} 收盘价走势")
    plt.xlabel("日期")
    plt.ylabel("收盘价")
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(f"data/processed/etf_{code}_close.png")
    print(f"\n收盘价走势图已保存至 data/processed/etf_{code}_close.png")

def example_get_technical_indicators():
    """获取技术指标示例"""
    print("\n获取技术指标示例:")
    
    # 设置ETF代码和日期范围
    code = "159915"  # 创业板ETF
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"获取ETF {code} 的MACD指标")
    
    # 获取MACD指标
    macd = get_technical_indicators(code, "macd", start_date=start_date, end_date=end_date)
    
    if macd.empty:
        print(f"未获取到ETF {code} 的MACD指标")
        return
    
    print(f"成功获取ETF {code} MACD指标，共 {len(macd)} 条记录")
    
    # 打印最近5条记录
    print("\n最近5条MACD记录:")
    recent_macd = macd.sort_values("date", ascending=False).head(5)
    print(recent_macd[["date", "macd", "macd_signal", "macd_hist"]])
    
    # 绘制MACD图表
    plt.figure(figsize=(12, 8))
    
    # 绘制MACD线和信号线
    plt.subplot(2, 1, 1)
    plt.plot(macd["date"], macd["macd"], label="MACD")
    plt.plot(macd["date"], macd["macd_signal"], label="Signal")
    plt.title(f"ETF {code} MACD指标")
    plt.legend()
    plt.grid(True)
    
    # 绘制MACD柱状图
    plt.subplot(2, 1, 2)
    plt.bar(macd["date"], macd["macd_hist"], label="Histogram")
    plt.title(f"ETF {code} MACD柱状图")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(f"data/processed/etf_{code}_macd.png")
    print(f"\nMACD指标图已保存至 data/processed/etf_{code}_macd.png")

def example_get_fund_flow():
    """获取行业资金流向示例"""
    print("\n获取行业资金流向示例:")
    
    # 获取行业资金流向数据
    df = get_industry_fund_flow()
    
    if df.empty:
        print("未获取到行业资金流向数据")
        return
    
    print(f"成功获取行业资金流向数据，共 {len(df)} 条记录")
    
    # 打印前5条记录
    print("\n行业资金流向前5条记录:")
    print(df.head(5))

def example_get_market_sentiment():
    """获取市场情绪指标示例"""
    print("\n获取市场情绪指标示例:")
    
    # 获取市场情绪指标数据
    sentiment = get_market_sentiment()
    
    if not sentiment:
        print("未获取到市场情绪指标数据")
        return
    
    print(f"成功获取市场情绪指标数据，共 {len(sentiment)} 个指标")
    
    # 打印所有指标
    print("\n市场情绪指标:")
    for key, value in sentiment.items():
        print(f"{key}: {value}")

def main():
    """主函数"""
    print("=" * 50)
    print("数据模块使用示例")
    print("=" * 50)
    
    # 确保目录存在
    os.makedirs("data/processed", exist_ok=True)
    
    # 运行示例
    example_get_etf_list()
    example_get_etf_history()
    example_get_technical_indicators()
    example_get_fund_flow()
    example_get_market_sentiment()
    
    print("\n示例运行完成")

if __name__ == "__main__":
    main()