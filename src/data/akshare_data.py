#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AKShare数据获取模块
用于从AKShare获取ETF、指数和股票数据
"""

import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta
import logging
from pathlib import Path

# --- 新增导入 ---
from ..utils.db_utils import get_db_engine, upsert_df_to_sql, load_config

# --- 移除 os 导入 (如果不再需要) ---
# import os

# 设置日志
logger = logging.getLogger(__name__)


class AKShareData:
    """AKShare数据获取类"""

    def __init__(self, raw_data_dir="data/raw", processed_data_dir="data/processed"):
        """
        初始化AKShare数据获取类

        Parameters
        ----------
        raw_data_dir : str
            原始数据存储目录 (保留用于可能的备份或日志)
        processed_data_dir : str
            处理后数据存储目录 (可能不再需要)
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)

        # 确保目录存在
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        # --- 初始化数据库引擎 ---
        self.config = load_config()
        self.engine = get_db_engine(self.config)
        # --- ---

        logger.info(
            f"AKShare数据获取模块初始化完成，原始数据目录: {raw_data_dir}, 处理后数据目录: {processed_data_dir}"
        )
        if not self.engine:
            logger.warning("数据库引擎初始化失败，数据将无法保存到数据库。")

    def get_etf_list(self, save=True):
        """
        获取所有ETF基金列表并写入数据库
        """
        logger.info("开始获取ETF基金列表...")
        if not self.engine and save:
            logger.error("数据库未连接，无法保存ETF列表。")
            return pd.DataFrame()

        try:
            etf_df = ak.fund_etf_category_sina(symbol="ETF基金")

            # --- 数据库列名映射 ---
            column_mapping = {
                "代码": "code",
                "名称": "name",
                "最新价": "latest_price",
                "涨跌额": "price_change",
                "涨跌幅": "pct_change",
                "买入": "buy_price",
                "卖出": "sell_price",
                "昨收": "pre_close",
                "今开": "open_price",
                "最高": "high_price",
                "最低": "low_price",
                "成交量": "volume",  # 注意单位是手
                "成交额": "amount",
            }
            etf_df = etf_df.rename(columns=column_mapping)

            # 处理百分比字符串
            if "pct_change" in etf_df.columns:
                # 转换为数值，处理可能的 '--' 或空值
                etf_df["pct_change"] = (
                    pd.to_numeric(
                        etf_df["pct_change"].astype(str).str.replace("%", ""),
                        errors="coerce",
                    )
                    / 100
                )

            # 处理成交量单位 (手 -> 股，假设1手=100股，如果接口单位是股则不需要)
            # if "volume" in etf_df.columns:
            #     etf_df["volume"] = pd.to_numeric(etf_df["volume"], errors='coerce') * 100

            # 添加获取日期作为更新日期
            etf_df["update_date"] = datetime.now().date()  # 使用 date() 获取日期部分

            # --- 选择数据库表需要的列 ---
            db_columns = [
                "code",
                "name",
                "latest_price",
                "price_change",
                "pct_change",
                "buy_price",
                "sell_price",
                "pre_close",
                "open_price",
                "high_price",
                "low_price",
                "volume",
                "amount",
                "update_date",
            ]
            # 过滤掉 DataFrame 中不存在的列
            db_columns_present = [col for col in db_columns if col in etf_df.columns]
            etf_df_to_save = etf_df[db_columns_present].copy()

            # 转换数据类型以匹配数据库
            for col in [
                "latest_price",
                "price_change",
                "pct_change",
                "pre_close",
                "open_price",
                "high_price",
                "low_price",
                "amount",
            ]:
                if col in etf_df_to_save.columns:
                    etf_df_to_save[col] = pd.to_numeric(
                        etf_df_to_save[col], errors="coerce"
                    )
            if "volume" in etf_df_to_save.columns:
                etf_df_to_save["volume"] = pd.to_numeric(
                    etf_df_to_save["volume"], errors="coerce"
                ).astype(
                    "Int64"
                )  # 使用可空整数

            # 移除包含 NaN 主键的行 (code 不应为空)
            etf_df_to_save.dropna(subset=["code"], inplace=True)

            if save and self.engine:
                # --- 写入数据库 ---
                success = upsert_df_to_sql(
                    etf_df_to_save, "etf_list", self.engine, unique_columns=["code"]
                )
                if success:
                    logger.info(
                        f"ETF基金列表已成功写入数据库 etf_list，共 {len(etf_df_to_save)} 条记录"
                    )
                else:
                    logger.error("ETF基金列表写入数据库失败。")
                # --- (可选) 保留CSV备份 ---
                # file_path = self.raw_data_dir / f"etf_list_{datetime.now().strftime('%Y%m%d')}.csv"
                # etf_df.to_csv(file_path, index=False, encoding="utf-8-sig")
                # logger.info(f"ETF基金列表已备份至 {file_path}")

            return etf_df  # 返回原始获取的完整 DataFrame

        except Exception as e:
            logger.error(f"获取或处理ETF基金列表失败: {e}", exc_info=True)
            return pd.DataFrame()

    def get_industry_etfs(self, save=True):  # save 参数保留接口一致性，但实际不再使用
        """
        获取所有ETF列表 (数据来自 get_etf_list)
        """
        logger.info("开始获取所有ETF列表 (通过 get_etf_list)...")
        try:
            # 直接调用 get_etf_list，它会处理数据库写入
            all_etfs = self.get_etf_list(save=save)  # 传递 save 参数

            # --- 移除冗余的 CSV 保存 ---
            # if save:
            #     file_path = self.raw_data_dir / f"industry_etfs_{datetime.now().strftime('%Y%m%d')}.csv"
            #     all_etfs.to_csv(file_path, index=False, encoding="utf-8-sig")
            #     logger.info(f"ETF列表已保存至 {file_path}，共 {len(all_etfs)} 条记录")

            return all_etfs

        except Exception as e:
            logger.error(f"获取ETF列表失败: {e}", exc_info=True)
            return pd.DataFrame()

    def get_etf_history(
        self, code, start_date=None, end_date=None, fields=None, adjust="", save=True
    ):
        """
        获取ETF历史行情数据并写入数据库
        """
        if not self.engine and save:
            logger.error(f"数据库未连接，无法保存ETF {code} 的历史数据。")
            return pd.DataFrame()

        # 设置默认日期
        if start_date is None:
            start_date_req = (datetime.now() - timedelta(days=365)).strftime(
                "%Y%m%d"
            )  # AKShare 需要 YYYYMMDD
        else:
            start_date_req = start_date.replace("-", "")
        if end_date is None:
            end_date_req = datetime.now().strftime("%Y%m%d")  # AKShare 需要 YYYYMMDD
        else:
            end_date_req = end_date.replace("-", "")

        logger.info(
            f"开始获取ETF {code} 从 {start_date_req} 到 {end_date_req} 的历史数据..."
        )

        try:
            # 使用AKShare获取ETF历史数据 (fund_etf_hist_em 更常用)
            # df = ak.fund_etf_hist_sina(symbol=code) # Sina 接口可能不稳定或数据不全
            code = code[2:]
            df = ak.fund_etf_hist_em(
                symbol=code,
                period="daily",
                start_date=start_date_req,
                end_date=end_date_req,
                adjust=adjust,
            )

            if df.empty:
                logger.warning(f"未能获取到 ETF {code} 的历史数据。")
                return pd.DataFrame()

            # --- 数据库列名映射 ---
            column_mapping = {
                "日期": "trade_date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",  # 注意单位可能是手或股，需确认
                # "成交额": "amount", # etf_indicators 表没有 amount
                # "涨跌幅": "pct_change", # etf_indicators 表没有 pct_change, 但有 daily_return
                # "涨跌额": "change",
                # "换手率": "turnover_rate"
            }
            df = df.rename(columns=column_mapping)

            # 转换日期格式为 date 对象
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

            # 添加 ETF 代码列
            df["etf_code"] = code

            # --- 计算日收益率 (如果需要且表中没有) ---
            # df['daily_return'] = df['close'].pct_change() # etf_indicators 有 daily_return，但通常在 processor 中计算

            # --- 选择数据库表需要的列 ---
            db_columns = [
                "etf_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
            # 过滤掉 DataFrame 中不存在的列
            db_columns_present = [col for col in db_columns if col in df.columns]
            df_to_save = df[db_columns_present].copy()

            # 转换数据类型
            for col in ["open", "high", "low", "close"]:
                if col in df_to_save.columns:
                    df_to_save[col] = pd.to_numeric(df_to_save[col], errors="coerce")
            if "volume" in df_to_save.columns:
                # AKShare 的成交量单位通常是 手，数据库需要 BIGINT (股)
                df_to_save["volume"] = (
                    pd.to_numeric(df_to_save["volume"], errors="coerce") * 100
                )
                df_to_save["volume"] = df_to_save["volume"].astype("Int64")

            # 移除包含 NaN 主键的行
            df_to_save.dropna(subset=["etf_code", "trade_date"], inplace=True)

            # 不再在这里保存数据，而是在 calculate_technical_indicators 方法计算完成后保存
            # if save and self.engine:
            #     # --- 写入数据库 ---
            #     success = upsert_df_to_sql(
            #         df_to_save,
            #         "etf_indicators",
            #         self.engine,
            #         unique_columns=["etf_code", "trade_date"],
            #     )
            #     if success:
            #         logger.info(
            #             f"ETF {code} 历史数据已成功写入数据库 etf_indicators，共 {len(df_to_save)} 条记录"
            #         )
            #     else:
            #         logger.error(f"ETF {code} 历史数据写入数据库失败。")
            #     # --- (可选) 保留CSV备份 ---
            #     # file_path = self.raw_data_dir / f"etf_{code}_{start_date_req}_{end_date_req}.csv"
            #     # df.to_csv(file_path, index=False)
            #     # logger.info(f"ETF {code} 历史数据已备份至 {file_path}")

            return df_to_save  # 返回处理后的 DataFrame 而不是原始的 df

        except Exception as e:
            logger.error(f"获取或处理ETF {code} 历史数据失败: {e}", exc_info=True)
            return pd.DataFrame()

    def get_index_history(
        self, code, start_date=None, end_date=None, fields=None, save=True
    ):
        """
        获取指数历史行情数据 (暂不写入数据库)
        """
        # 设置默认日期
        if start_date is None:
            start_date_req = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
        else:
            start_date_req = start_date.replace("-", "")
        if end_date is None:
            end_date_req = datetime.now().strftime("%Y%m%d")
        else:
            end_date_req = end_date.replace("-", "")

        logger.info(
            f"开始获取指数 {code} 从 {start_date_req} 到 {end_date_req} 的历史数据..."
        )

        try:
            # 使用AKShare获取指数历史数据
            # 需要根据指数代码前缀判断市场，如 sh000001, sz399001
            # AKShare 指数接口通常不需要前缀，如 "000300"
            df = ak.stock_zh_index_daily(symbol=code)

            if df.empty:
                logger.warning(f"未能获取到指数 {code} 的历史数据。")
                return pd.DataFrame()

            # 转换日期格式
            df["date"] = pd.to_datetime(df["date"]).dt.date  # 使用 date 对象

            # 筛选日期范围
            start_date_obj = pd.to_datetime(start_date_req).date()
            end_date_obj = pd.to_datetime(end_date_req).date()
            df = df[(df["date"] >= start_date_obj) & (df["date"] <= end_date_obj)]

            # 按日期排序
            df = df.sort_values("date")

            # 筛选字段
            if fields is not None:
                df = df[["date"] + [f for f in fields if f in df.columns]]

            if save:
                # --- 暂不写入数据库 ---
                logger.warning(
                    f"指数 {code} 历史数据获取成功，但目前未配置写入数据库。"
                )
                # --- (可选) 保留CSV备份 ---
                # file_path = self.raw_data_dir / f"index_{code}_{start_date_req}_{end_date_req}.csv"
                # df.to_csv(file_path, index=False)
                # logger.info(f"指数 {code} 历史数据已备份至 {file_path}，共 {len(df)} 条记录")

            return df

        except Exception as e:
            logger.error(f"获取指数 {code} 历史数据失败: {e}", exc_info=True)
            return pd.DataFrame()

    def get_fund_flow(self, date=None, save=True):
        """
        获取行业资金流向数据并写入数据库
        """
        if not self.engine and save:
            logger.error("数据库未连接，无法保存行业资金流向数据。")
            return pd.DataFrame()

        if date is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
            date_obj = datetime.now().date()
        else:
            date_str = date
            date_obj = pd.to_datetime(date).date()

        logger.info(f"开始获取 {date_str} 的行业资金流向数据...")

        try:
            # 使用AKShare获取行业资金流向数据
            df = ak.stock_sector_fund_flow_rank(
                indicator="今日", sector_type="行业资金流"
            )  # 确保获取的是行业数据

            if df.empty:
                logger.warning(f"未能获取到 {date_str} 的行业资金流向数据。")
                return pd.DataFrame()

            # --- 数据库列名映射 ---
            column_mapping = {
                "名称": "industry_name",
                "今日涨跌幅": "price_change_pct",
                "今日主力净流入-净额": "main_net_inflow",
                "今日主力净流入-净占比": "main_net_inflow_pct",
                "今日超大单净流入-净额": "super_large_net_inflow",
                "今日超大单净流入-净占比": "super_large_net_inflow_pct",
                "今日大单净流入-净额": "large_net_inflow",
                "今日大单净流入-净占比": "large_net_inflow_pct",
                "今日中单净流入-净额": "medium_net_inflow",
                "今日中单净流入-净占比": "medium_net_inflow_pct",
                "今日小单净流入-净额": "small_net_inflow",
                "今日小单净流入-净占比": "small_net_inflow_pct",
                "今日主力净流入最大股": "top_stock",
            }
            df = df.rename(columns=column_mapping)

            # 添加日期列
            df["trade_date"] = date_obj

            # 处理百分比和金额单位
            pct_cols = [col for col in df.columns if "pct" in col]
            for col in pct_cols:
                df[col] = (
                    pd.to_numeric(df[col], errors="coerce") / 100
                )  # AKShare 返回的是 xx.xx

            amount_cols = [
                col for col in df.columns if "inflow" in col and "pct" not in col
            ]
            for col in amount_cols:
                # AKShare 返回单位是 万，数据库需要 元
                df[col] = pd.to_numeric(df[col], errors="coerce") * 10000

            # --- 选择数据库表需要的列 ---
            db_columns = [
                "industry_name",
                "trade_date",
                "price_change_pct",
                "main_net_inflow",
                "main_net_inflow_pct",
                "super_large_net_inflow",
                "super_large_net_inflow_pct",
                "large_net_inflow",
                "large_net_inflow_pct",
                "medium_net_inflow",
                "medium_net_inflow_pct",
                "small_net_inflow",
                "small_net_inflow_pct",
                "top_stock",
            ]
            db_columns_present = [col for col in db_columns if col in df.columns]
            df_to_save = df[db_columns_present].copy()

            # 移除包含 NaN 主键的行
            df_to_save.dropna(subset=["industry_name", "trade_date"], inplace=True)

            if save and self.engine:
                # --- 写入数据库 ---
                success = upsert_df_to_sql(
                    df_to_save,
                    "industry_fund_flow",
                    self.engine,
                    unique_columns=["industry_name", "trade_date"],
                )
                if success:
                    logger.info(
                        f"行业资金流向数据已成功写入数据库 industry_fund_flow，共 {len(df_to_save)} 条记录"
                    )
                else:
                    logger.error("行业资金流向数据写入数据库失败。")
                # --- (可选) 保留CSV备份 ---
                # file_path = self.raw_data_dir / f"industry_fund_flow_{date_str.replace('-', '')}.csv"
                # df.to_csv(file_path, index=False, encoding="utf-8-sig")
                # logger.info(f"行业资金流向数据已备份至 {file_path}")

            return df  # 返回原始获取的 DataFrame

        except Exception as e:
            logger.error(f"获取或处理行业资金流向数据失败: {e}", exc_info=True)
            return pd.DataFrame()

    def get_market_sentiment(self, date=None, save=True):
        """
        获取市场情绪指标数据并写入数据库
        """
        if not self.engine and save:
            logger.error("数据库未连接，无法保存市场情绪数据。")
            return {}

        if date is None:
            date_str_req = datetime.now().strftime("%Y%m%d")  # AKShare 需要 YYYYMMDD
            date_obj = datetime.now().date()
        else:
            date_str_req = date.replace("-", "")
            date_obj = pd.to_datetime(date).date()

        logger.info(f"开始获取 {date_str_req} 的市场情绪指标数据...")

        sentiment_data = {"trade_date": date_obj}  # 使用 date 对象

        try:
            # # 1. 获取北向资金数据 (接口不稳定或已更改，暂时注释)
            # try:
            #     north_df = ak.stock_hsgt_hist_em(symbol="北向资金")
            #     north_df["日期"] = pd.to_datetime(north_df["日期"]).dt.date
            #     north_today = north_df[north_df["日期"] == date_obj]
            #     if not north_today.empty:
            #         sentiment_data["north_fund"] = north_today.iloc[0]["当日成交净买额"] # 单位：亿元
            #         logger.info(f"获取到北向资金净流入: {sentiment_data['north_fund']}亿元")
            #     else:
            #         logger.warning(f"未找到{date_str_req}的北向资金数据")
            # except Exception as e:
            #     logger.warning(f"获取北向资金数据失败: {e}")

            # 2. 获取融资融券数据 
            try:
                # 尝试获取沪市
                margin_sh_df = ak.stock_margin_sse(start_date=date_str_req, end_date=date_str_req)
                # 尝试获取深市
                margin_sz_df = ak.stock_margin_szse(date=date_str_req) # 注意接口参数差异
                margin_balance = 0
                if not margin_sh_df.empty:
                    margin_balance += margin_sh_df['融资余额'].iloc[0] if '融资余额' in margin_sh_df.columns else 0
                if not margin_sz_df.empty:
                     margin_balance += margin_sz_df['融资余额'].iloc[0] if '融资余额' in margin_sz_df.columns else 0
                if margin_balance > 0:
                    sentiment_data["margin_balance"] = margin_balance / 100000000 # 转为亿元
                    logger.info(f"获取到融资融券余额: {sentiment_data['margin_balance']}亿元")
                else:
                     logger.warning(f"未找到{date_str_req}的融资融券数据")
            except Exception as e:
                logger.warning(f"获取融资融券数据失败: {e}")

            # 3. 获取涨跌停数据
            try:
                # 涨停
                limit_up_df = ak.stock_zt_pool_em(date=date_str_req)
                sentiment_data["up_limit_count"] = (
                    len(limit_up_df) if not limit_up_df.empty else 0
                )

                # 跌停
                limit_down_df = ak.stock_zt_pool_dtgc_em(date=date_str_req)
                sentiment_data["down_limit_count"] = (
                    len(limit_down_df) if not limit_down_df.empty else 0
                )

                # 炸板
                # failed_limit_up_df = ak.stock_zt_pool_strong_em(date=date_str_req)
                # sentiment_data["failed_limit_up_count"] = len(failed_limit_up_df) if not failed_limit_up_df.empty else 0

                logger.info(
                    f"获取到涨停数: {sentiment_data.get('up_limit_count', 0)}, 跌停数: {sentiment_data.get('down_limit_count', 0)}"
                )

            except Exception as e:
                logger.warning(f"获取涨跌停数据失败: {e}")
                sentiment_data["up_limit_count"] = 0
                sentiment_data["down_limit_count"] = 0

            # 4. 计算涨跌比 (使用 stock_board_industry_summary_ths)
            logger.info("Fetching industry summary data (ths) for ratio calculation...")
            advances = 0
            declines = 0
            ratio = None # Default value
            try:
                industry_summary_df = ak.stock_board_industry_summary_ths()

                if not industry_summary_df.empty and '上涨家数' in industry_summary_df.columns and '下跌家数' in industry_summary_df.columns:
                    # 确保列是数值类型，非数值转为0
                    industry_summary_df['上涨家数'] = pd.to_numeric(industry_summary_df['上涨家数'], errors='coerce').fillna(0)
                    industry_summary_df['下跌家数'] = pd.to_numeric(industry_summary_df['下跌家数'], errors='coerce').fillna(0)
                    
                    advances = int(industry_summary_df['上涨家数'].sum())
                    declines = int(industry_summary_df['下跌家数'].sum())
                    
                    if advances > 0 or declines > 0: # Avoid division by zero
                        total = advances + declines
                        ratio = round(advances / total, 4)
                        logger.info(f"Calculated up/down ratio from ths industry summary: {ratio} (Advances: {advances}, Declines: {declines})")
                    else:
                         logger.warning("Aggregated advances and declines from ths industry summary are zero, cannot calculate ratio.")
                else:
                    logger.warning("Could not find or process '上涨家数'/'下跌家数' columns in stock_board_industry_summary_ths output. Ratio calculation skipped.")
                    if not industry_summary_df.empty:
                        logger.debug(f"Columns available in ths industry summary: {industry_summary_df.columns}")
                    else:
                        logger.warning("stock_board_industry_summary_ths returned an empty DataFrame.")

            except Exception as e:
                logger.error(f"Error fetching or processing stock_board_industry_summary_ths data for ratio: {e}", exc_info=True)
            
            sentiment_data["up_down_ratio"] = ratio # Assign calculated ratio or None
            # 如果需要，也可以保存原始涨跌家数
            # sentiment_data["advances_count"] = advances
            # sentiment_data["declines_count"] = declines

            if save and self.engine:
                # --- 转换为 DataFrame 并写入数据库 ---
                sentiment_df = pd.DataFrame([sentiment_data])

                # 选择数据库列 (确保包含 up_down_ratio)
                db_columns = [
                    "trade_date",
                    "up_limit_count",
                    "down_limit_count",
                    "up_down_ratio",
                    "margin_balance"
                    # 如有需要，添加其他之前保存的列: "north_fund", "margin_balance", "failed_limit_up_count"
                ]
                db_columns_present = [
                    col for col in db_columns if col in sentiment_df.columns
                ]
                sentiment_df_to_save = sentiment_df[db_columns_present].copy()

                # 转换数据类型
                if "up_limit_count" in sentiment_df_to_save.columns:
                    sentiment_df_to_save["up_limit_count"] = sentiment_df_to_save[
                        "up_limit_count"
                    ].astype("Int64")
                if "down_limit_count" in sentiment_df_to_save.columns:
                    sentiment_df_to_save["down_limit_count"] = sentiment_df_to_save[
                        "down_limit_count"
                    ].astype("Int64")
                if "up_down_ratio" in sentiment_df_to_save.columns:
                    sentiment_df_to_save["up_down_ratio"] = pd.to_numeric(
                        sentiment_df_to_save["up_down_ratio"], errors="coerce"
                    ).round(4) # 增加精度控制

                # 移除 NaN 主键行
                sentiment_df_to_save.dropna(subset=["trade_date"], inplace=True)

                if not sentiment_df_to_save.empty:
                    success = upsert_df_to_sql(
                        sentiment_df_to_save,
                        "market_sentiment",
                        self.engine,
                        unique_columns=["trade_date"],
                    )
                    if success:
                        logger.info(
                            f"市场情绪指标数据已成功写入数据库 market_sentiment"
                        )
                    else:
                        logger.error("市场情绪指标数据写入数据库失败。")
                else:
                    logger.warning("没有有效的市场情绪数据可写入数据库。")

                # --- (可选) 保留CSV备份 ---
                # file_path = self.raw_data_dir / f"market_sentiment_{date_str_req}.csv"
                # pd.DataFrame([sentiment_data]).to_csv(file_path, index=False)
                # logger.info(f"市场情绪指标数据已备份至 {file_path}")

            return sentiment_data  # 返回包含所有指标的字典

        except Exception as e:
            logger.error(f"获取或处理市场情绪指标数据失败: {e}", exc_info=True)
            return {}

    def get_all_data(
        self, start_date=None, end_date=None, etf_codes=None, save=True
    ):  # 添加 save 参数
        """
        获取所有需要的数据并写入数据库
        """
        # 设置默认日期
        start_date_req = start_date  # 保留原始格式用于日志
        end_date_req = end_date
        if start_date is None:
            start_date_req = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date_req = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"开始获取所有数据，是否保存到数据库: {save}")
        result = {}

        # 1. 获取 ETF 列表 (get_etf_list 内部处理保存)
        etf_list = self.get_etf_list(save=save)
        result["etf_list"] = etf_list

        # 2. 获取 ETF 指标数据 (get_etf_indicator 内部处理保存)
        etf_indicator = self.get_etf_indicator(
            etf_codes if etf_codes else etf_list["ts_code"].tolist(),
            start_date,
            end_date,
            save=save,
        )
        result["etf_indicator"] = etf_indicator

        # 3. 获取市场概览数据 (暂未实现)
        # market_overview = self.get_market_overview(save=save)
        # result['market_overview'] = market_overview

        # 4. 获取行业资金流向 (get_industry_fund_flow 内部处理保存)
        fund_flow = self.get_industry_fund_flow(save=save)
        result["fund_flow"] = fund_flow

        # 5. 获取最新的市场情绪指标 (get_market_sentiment 内部处理保存)
        sentiment = self.get_market_sentiment(
            date="2024-04-12", save=True
        )  # 使用一个过去的交易日测试
        result["sentiment"] = sentiment

        logger.info("所有数据获取完成")
        return result

    def get_etf_minute_kline(
        self, code, period=5, start_date=None, end_date=None, save=True
    ):
        """
        获取ETF分钟级别K线数据并写入数据库
        
        Parameters
        ----------
        code : str
            ETF代码，例如 "sh510050"
        period : int
            K线周期，支持 5（5分钟）、15（15分钟）、60（60分钟）
        start_date : str, optional
            开始日期，格式为 "YYYY-MM-DD"，默认为当前日期前7天
        end_date : str, optional
            结束日期，格式为 "YYYY-MM-DD"，默认为当前日期
        save : bool, optional
            是否保存到数据库，默认为 True
            
        Returns
        -------
        pandas.DataFrame
            包含分钟级别K线数据的DataFrame
        """
        if not self.engine and save:
            logger.error(f"数据库未连接，无法保存ETF {code} 的分钟K线数据。")
            return pd.DataFrame()
            
        # 验证周期参数
        valid_periods = [5, 15, 60]
        if period not in valid_periods:
            logger.error(f"不支持的K线周期: {period}，支持的周期为: {valid_periods}")
            return pd.DataFrame()
            
        # 设置默认日期
        if start_date is None:
            start_date_obj = (datetime.now() - timedelta(days=7)).date()
            start_date = start_date_obj.strftime("%Y-%m-%d")
        else:
            start_date_obj = pd.to_datetime(start_date).date()
            
        if end_date is None:
            end_date_obj = datetime.now().date()
            end_date = end_date_obj.strftime("%Y-%m-%d")
        else:
            end_date_obj = pd.to_datetime(end_date).date()
            
        # AKShare接口需要的日期格式转换
        start_date_req = start_date.replace("-", "")
        end_date_req = end_date.replace("-", "")
        
        logger.info(
            f"开始获取ETF {code} 从 {start_date} 到 {end_date} 的 {period} 分钟K线数据..."
        )
        
        try:
            # 使用AKShare获取ETF分钟K线数据
            # 移除前缀，AKShare接口通常不需要sh/sz前缀
            if code.startswith("sh") or code.startswith("sz"):
                code_no_prefix = code[2:]
            else:
                code_no_prefix = code
                
            # 调用AKShare接口获取分钟K线数据
            # 根据period参数选择相应的周期
            period_map = {5: "5", 15: "15", 60: "60"}
            period_str = period_map.get(period, "5")
            
            df = ak.stock_zh_a_hist_min_em(
                symbol=code_no_prefix,
                period=period_str,
                start_date=start_date_req,
                end_date=end_date_req,
            )
            
            if df.empty:
                logger.warning(f"未能获取到ETF {code} 的 {period} 分钟K线数据。")
                return pd.DataFrame()
                
            # 数据库列名映射
            column_mapping = {
                "时间": "datetime",  # 临时列名，后面会拆分为trade_date和trade_time
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "成交额": "amount",
            }
            df = df.rename(columns=column_mapping)
            
            # 将datetime列拆分为trade_date和trade_time
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["trade_date"] = df["datetime"].dt.date
            df["trade_time"] = df["datetime"].dt.time
            
            # 添加ETF代码和周期列
            df["etf_code"] = code_no_prefix
            df["period"] = period
            
            # 选择数据库表需要的列
            db_columns = [
                "etf_code",
                "trade_date",
                "trade_time",
                "period",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
            ]
            
            # 过滤掉DataFrame中不存在的列
            db_columns_present = [col for col in db_columns if col in df.columns]
            df_to_save = df[db_columns_present].copy()
            
            # 转换数据类型
            for col in ["open", "high", "low", "close", "amount"]:
                if col in df_to_save.columns:
                    df_to_save[col] = pd.to_numeric(df_to_save[col], errors="coerce")
                    
            if "volume" in df_to_save.columns:
                # AKShare的成交量单位通常是手，数据库需要股
                df_to_save["volume"] = (
                    pd.to_numeric(df_to_save["volume"], errors="coerce") * 100
                )
                df_to_save["volume"] = df_to_save["volume"].astype("Int64")
                
            # 移除包含NaN主键的行
            df_to_save.dropna(subset=["etf_code", "trade_date", "trade_time"], inplace=True)
            
            # 写入数据库
            if save and self.engine:
                success = upsert_df_to_sql(
                    df_to_save,
                    "minute_kline_data",
                    self.engine,
                    unique_columns=["etf_code", "trade_date", "trade_time", "period"],
                )
                
                if success:
                    logger.info(
                        f"ETF {code} 的 {period} 分钟K线数据已成功写入数据库 minute_kline_data，共 {len(df_to_save)} 条记录"
                    )
                else:
                    logger.error(f"ETF {code} 的 {period} 分钟K线数据写入数据库失败。")
                    
            return df_to_save
            
        except Exception as e:
            logger.error(f"获取或处理ETF {code} 的 {period} 分钟K线数据失败: {e}", exc_info=True)
            return pd.DataFrame()

if __name__ == "__main__":
    from sqlalchemy import create_engine
    from pathlib import Path
    from dotenv import load_dotenv
    import os

    # 加载 .env 文件
    load_dotenv()

    # 从环境变量获取数据库连接信息
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")

    # 创建数据库连接
    DATABASE_URL = (
        f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )
    engine = create_engine(DATABASE_URL)

    # 假设项目根目录是当前文件向上两级
    project_root = Path(__file__).resolve().parents[2]
    raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 AKShareData
    data = AKShareData(engine=engine, raw_data_dir=raw_data_dir)

    # --- 测试各个方法 ---
    logger.info("--- 开始 AKShare 数据获取与数据库写入测试 ---")

    # 测试获取 ETF 列表
    logger.info("--- 测试 get_etf_list ---")
    etf_list_df = data.get_etf_list(save=True)
    if not etf_list_df.empty:
        logger.info(f"获取到 {len(etf_list_df)} 条 ETF 列表数据")
        logger.info(etf_list_df.head())
    else:
        logger.warning("未能获取 ETF 列表数据或写入数据库失败")

    # 测试获取部分 ETF 指标数据
    logger.info("--- 测试 get_etf_indicator ---")
    test_etf_codes = ["510300.SH", "159915.SZ"]  # 示例 ETF 代码
    start = "2024-04-01"
    end = "2024-04-12"
    etf_indicator_df = data.get_etf_indicator(
        test_etf_codes, start_date=start, end_date=end, save=True
    )
    if not etf_indicator_df.empty:
        logger.info(f"获取到 {len(etf_indicator_df)} 条 ETF 指标数据")
        logger.info(etf_indicator_df.head())
    else:
        logger.warning("未能获取 ETF 指标数据或写入数据库失败")

    # 测试获取行业资金流向
    logger.info("--- 测试 get_industry_fund_flow ---")
    fund_flow_df = data.get_industry_fund_flow(save=True)
    if not fund_flow_df.empty:
        logger.info(f"获取到 {len(fund_flow_df)} 条行业资金流向数据")
        logger.info(fund_flow_df.head())
    else:
        logger.warning("未能获取行业资金流向数据或写入数据库失败")

    # 测试获取市场情绪
    logger.info("--- 测试 get_market_sentiment ---")
    sentiment = data.get_market_sentiment(
        date="2024-04-12", save=True
    )  # 使用一个过去的交易日测试
    if sentiment:
        logger.info(f"获取到市场情绪数据: {sentiment}")
    else:
        logger.warning("未能获取市场情绪数据或写入数据库失败")

    logger.info("--- AKShare 数据获取与数据库写入测试完成 ---")
