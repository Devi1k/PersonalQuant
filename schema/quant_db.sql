-- 创建量化交易数据库
CREATE DATABASE IF NOT EXISTS quant_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE quant_db;

-- 创建ETF基础信息表
CREATE TABLE IF NOT EXISTS etf_list (
    id INT NOT NULL AUTO_INCREMENT COMMENT '自增主键',
    code VARCHAR(8) NOT NULL COMMENT 'ETF代码',
    name VARCHAR(50) NOT NULL COMMENT 'ETF名称',
    latest_price DECIMAL(10, 2) COMMENT '最新价',
    price_change DECIMAL(10, 2) COMMENT '涨跌额',
    pct_change DECIMAL(10, 4) COMMENT '涨跌幅(%)',
    buy_price DECIMAL(10, 2) COMMENT '买入价',
    sell_price DECIMAL(10, 2) COMMENT '卖出价',
    pre_close DECIMAL(10, 2) COMMENT '昨收',
    open_price DECIMAL(10, 2) COMMENT '今开',
    high_price DECIMAL(10, 2) COMMENT '最高',
    low_price DECIMAL(10, 2) COMMENT '最低',
    volume BIGINT COMMENT '成交量(手)',
    amount DECIMAL(20, 2) COMMENT '成交额',
    update_date DATE NOT NULL COMMENT '数据更新日期',
    PRIMARY KEY (id),
    UNIQUE KEY uk_code (code),
    INDEX idx_update_date (update_date)
) ENGINE=InnoDB COMMENT='ETF基础信息表';

-- 创建ETF指标数据表
CREATE TABLE IF NOT EXISTS etf_indicators (
    id INT NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    etf_code VARCHAR(8) NOT NULL COMMENT 'ETF代码',
    trade_date DATE NOT NULL COMMENT '交易日期',
    open DECIMAL(10, 2) COMMENT '开盘价',
    high DECIMAL(10, 2) COMMENT '最高价',
    low DECIMAL(10, 2) COMMENT '最低价',
    close DECIMAL(10, 2) COMMENT '收盘价',
    volume BIGINT COMMENT '成交量',
    daily_return DECIMAL(10, 4) COMMENT '日收益率',
    volatility_20d DECIMAL(10, 4) COMMENT '20日波动率',
    ma_5 DECIMAL(10, 2) COMMENT '5日均线',
    ma_10 DECIMAL(10, 2) COMMENT '10日均线',
    ma_20 DECIMAL(10, 2) COMMENT '20日均线',
    ma_60 DECIMAL(10, 2) COMMENT '60日均线',
    ema_12 DECIMAL(10, 2) COMMENT '12日指数移动平均线',
    ema_21 DECIMAL(10, 2) COMMENT '21日指数移动平均线(1个月期)',
    ema_26 DECIMAL(10, 2) COMMENT '26日指数移动平均线',
    ema_144 DECIMAL(10, 2) COMMENT '144日指数移动平均线',
    ema_144_upper DECIMAL(10, 2) COMMENT '144日EMA上通道',
    ema_144_lower DECIMAL(10, 2) COMMENT '144日EMA下通道',
    ema_200 DECIMAL(10, 2) COMMENT '200日指数移动平均线',
    ema_21_200_diff DECIMAL(10, 4) COMMENT '21日与200日EMA差值',
    macd DECIMAL(10, 4) COMMENT 'MACD值',
    macd_signal DECIMAL(10, 4) COMMENT 'MACD信号线',
    macd_hist DECIMAL(10, 4) COMMENT 'MACD柱状图',
    rsi_14 DECIMAL(10, 4) COMMENT '14日RSI',
    bb_middle DECIMAL(10, 2) COMMENT '布林带中轨',
    bb_std DECIMAL(10, 4) COMMENT '布林带标准差',
    bb_upper DECIMAL(10, 2) COMMENT '布林带上轨',
    bb_lower DECIMAL(10, 2) COMMENT '布林带下轨',
    kdj_k DECIMAL(10, 4) COMMENT 'KDJ指标K值',
    kdj_d DECIMAL(10, 4) COMMENT 'KDJ指标D值',
    kdj_j DECIMAL(10, 4) COMMENT 'KDJ指标J值',
    cci_20 DECIMAL(10, 4) COMMENT '20日CCI',
    atr_14 DECIMAL(10, 4) COMMENT '14日ATR',
    obv BIGINT COMMENT 'OBV指标',
    volume_ma_5 BIGINT COMMENT '成交量5日均线',
    vwap_20 DECIMAL(10, 2) COMMENT '20日成交量加权平均价格',
    PRIMARY KEY (id, trade_date),
    UNIQUE KEY uk_etf_date (etf_code, trade_date),
    INDEX idx_trade_date (trade_date)
--  CONSTRAINT fk_etf_indicators_etf_list FOREIGN KEY (etf_code) REFERENCES etf_list (code) ON DELETE CASCADE ON UPDATE CASCADE -- Removed due to partitioning incompatibility
) ENGINE=InnoDB COMMENT='ETF技术指标表'
PARTITION BY RANGE (YEAR(trade_date)) (
    PARTITION p2020 VALUES LESS THAN (2021),
    PARTITION p2021 VALUES LESS THAN (2022),
    PARTITION p2022 VALUES LESS THAN (2023),
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026),
    PARTITION p_max VALUES LESS THAN MAXVALUE
);

-- 创建市场情绪数据表
CREATE TABLE IF NOT EXISTS market_sentiment (
    id INT NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    trade_date DATE NOT NULL COMMENT '交易日期',
    up_limit_count INT NOT NULL COMMENT '涨停数量',
    down_limit_count INT NOT NULL COMMENT '跌停数量',
    up_down_ratio DECIMAL(10, 4) COMMENT '涨跌比',
    margin_balance DECIMAL(20, 2) COMMENT '融资融券余额',
    PRIMARY KEY (id, trade_date),
    UNIQUE KEY uk_trade_date (trade_date)
) ENGINE=InnoDB COMMENT='市场情绪数据表'
PARTITION BY RANGE (YEAR(trade_date)) (
    PARTITION p2020 VALUES LESS THAN (2021),
    PARTITION p2021 VALUES LESS THAN (2022),
    PARTITION p2022 VALUES LESS THAN (2023),
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026),
    PARTITION p_max VALUES LESS THAN MAXVALUE
);

-- 创建行业资金流向表
CREATE TABLE IF NOT EXISTS industry_fund_flow (
    id INT NOT NULL AUTO_INCREMENT COMMENT '序号',
    industry_name VARCHAR(50) NOT NULL COMMENT '行业名称',
    trade_date DATE NOT NULL COMMENT '交易日期',
    price_change_pct DECIMAL(10, 4) COMMENT '今日涨跌幅',
    main_net_inflow DECIMAL(20, 2) COMMENT '今日主力净流入-净额',
    main_net_inflow_pct DECIMAL(10, 4) COMMENT '今日主力净流入-净占比',
    super_large_net_inflow DECIMAL(20, 2) COMMENT '今日超大单净流入-净额',
    super_large_net_inflow_pct DECIMAL(10, 4) COMMENT '今日超大单净流入-净占比',
    large_net_inflow DECIMAL(20, 2) COMMENT '今日大单净流入-净额',
    large_net_inflow_pct DECIMAL(10, 4) COMMENT '今日大单净流入-净占比',
    medium_net_inflow DECIMAL(20, 2) COMMENT '今日中单净流入-净额',
    medium_net_inflow_pct DECIMAL(10, 4) COMMENT '今日中单净流入-净占比',
    small_net_inflow DECIMAL(20, 2) COMMENT '今日小单净流入-净额',
    small_net_inflow_pct DECIMAL(10, 4) COMMENT '今日小单净流入-净占比',
    top_stock VARCHAR(50) COMMENT '今日主力净流入最大股',
    PRIMARY KEY (id, trade_date),
    UNIQUE KEY uk_industry_date (industry_name, trade_date),
    INDEX idx_trade_date (trade_date),
    INDEX idx_industry_name (industry_name)
) ENGINE=InnoDB COMMENT='行业资金流向表'
PARTITION BY RANGE (YEAR(trade_date)) (
    PARTITION p2020 VALUES LESS THAN (2021),
    PARTITION p2021 VALUES LESS THAN (2022),
    PARTITION p2022 VALUES LESS THAN (2023),
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026),
    PARTITION p_max VALUES LESS THAN MAXVALUE
);

-- 创建分钟级别K线数据表
CREATE TABLE IF NOT EXISTS minute_kline_data (
    id INT NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    etf_code VARCHAR(8) NOT NULL COMMENT 'ETF代码',
    trade_date DATE NOT NULL COMMENT '交易日期',
    trade_time TIME NOT NULL COMMENT '交易时间',
    period INT NOT NULL COMMENT 'K线周期(分钟)',
    open DECIMAL(10, 2) COMMENT '开盘价',
    high DECIMAL(10, 2) COMMENT '最高价',
    low DECIMAL(10, 2) COMMENT '最低价',
    close DECIMAL(10, 2) COMMENT '收盘价',
    volume BIGINT COMMENT '成交量',
    amount DECIMAL(20, 2) COMMENT '成交额',
    PRIMARY KEY (id, trade_date),
    UNIQUE KEY uk_etf_datetime_period (etf_code, trade_date, trade_time, period),
    INDEX idx_etf_code (etf_code),
    INDEX idx_trade_date (trade_date),
    INDEX idx_period (period)
) ENGINE=InnoDB COMMENT='分钟级别K线数据表'
PARTITION BY RANGE (YEAR(trade_date)) (
    PARTITION p2020 VALUES LESS THAN (2021),
    PARTITION p2021 VALUES LESS THAN (2022),
    PARTITION p2022 VALUES LESS THAN (2023),
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026),
    PARTITION p_max VALUES LESS THAN MAXVALUE
);

-- 创建分钟级别技术指标表
CREATE TABLE IF NOT EXISTS minute_indicators (
    id INT NOT NULL AUTO_INCREMENT COMMENT '自增ID',
    etf_code VARCHAR(8) NOT NULL COMMENT 'ETF代码',
    trade_date DATE NOT NULL COMMENT '交易日期',
    trade_time TIME NOT NULL COMMENT '交易时间',
    period INT NOT NULL COMMENT 'K线周期(分钟)',
    ma_5 DECIMAL(10, 2) COMMENT '5周期均线',
    ma_10 DECIMAL(10, 2) COMMENT '10周期均线',
    ma_20 DECIMAL(10, 2) COMMENT '20周期均线',
    ema_12 DECIMAL(10, 2) COMMENT '12周期指数移动平均线',
    ema_26 DECIMAL(10, 2) COMMENT '26周期指数移动平均线',
    macd DECIMAL(10, 4) COMMENT 'MACD值',
    macd_signal DECIMAL(10, 4) COMMENT 'MACD信号线',
    macd_hist DECIMAL(10, 4) COMMENT 'MACD柱状图',
    rsi_14 DECIMAL(10, 4) COMMENT '14周期RSI',
    bb_middle DECIMAL(10, 2) COMMENT '布林带中轨',
    bb_upper DECIMAL(10, 2) COMMENT '布林带上轨',
    bb_lower DECIMAL(10, 2) COMMENT '布林带下轨',
    volume_ma_5 BIGINT COMMENT '成交量5周期均线',
    PRIMARY KEY (id, trade_date),
    UNIQUE KEY uk_etf_datetime_period (etf_code, trade_date, trade_time, period),
    INDEX idx_etf_code (etf_code),
    INDEX idx_trade_date (trade_date),
    INDEX idx_period (period)
) ENGINE=InnoDB COMMENT='分钟级别技术指标表'
PARTITION BY RANGE (YEAR(trade_date)) (
    PARTITION p2020 VALUES LESS THAN (2021),
    PARTITION p2021 VALUES LESS THAN (2022),
    PARTITION p2022 VALUES LESS THAN (2023),
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026),
    PARTITION p_max VALUES LESS THAN MAXVALUE
);

-- 创建视图: 最新分钟级别K线数据视图
CREATE OR REPLACE VIEW v_latest_minute_kline AS
SELECT k.*, l.name as etf_name
FROM minute_kline_data k
JOIN (
    SELECT etf_code, period, MAX(trade_date) as latest_date
    FROM minute_kline_data
    GROUP BY etf_code, period
) latest ON k.etf_code = latest.etf_code AND k.period = latest.period AND k.trade_date = latest.latest_date
JOIN etf_list l ON k.etf_code = l.code;

-- 创建视图: 最新ETF指标视图
CREATE OR REPLACE VIEW v_latest_etf_indicators AS
SELECT e.*, l.name as etf_name
FROM etf_indicators e
JOIN (
    SELECT etf_code, MAX(trade_date) as latest_date
    FROM etf_indicators
    GROUP BY etf_code
) latest ON e.etf_code = latest.etf_code AND e.trade_date = latest.latest_date
JOIN etf_list l ON e.etf_code = l.code;

-- 创建视图: 市场情绪与资金流向综合视图
CREATE OR REPLACE VIEW v_market_overview AS
SELECT 
    ms.trade_date,
    ms.up_limit_count,
    ms.down_limit_count,
    ms.up_down_ratio,
    ms.margin_balance,
    COUNT(DISTINCT iff.industry_name) AS industry_count,
    SUM(CASE WHEN iff.main_net_inflow > 0 THEN 1 ELSE 0 END) AS inflow_industry_count,
    SUM(CASE WHEN iff.main_net_inflow < 0 THEN 1 ELSE 0 END) AS outflow_industry_count,
    SUM(iff.main_net_inflow) AS total_main_net_inflow
FROM market_sentiment ms
LEFT JOIN industry_fund_flow iff ON ms.trade_date = iff.trade_date
GROUP BY ms.trade_date, ms.up_limit_count, ms.down_limit_count, ms.up_down_ratio, ms.margin_balance;

-- 添加数据库用户和权限
-- CREATE USER 'quant_user'@'localhost' IDENTIFIED BY 'password';
-- GRANT SELECT, INSERT, UPDATE, DELETE ON quant_db.* TO 'quant_user'@'localhost';
-- FLUSH PRIVILEGES;

-- 注释: 根据实际需求调整以下参数
-- 
-- 优化InnoDB性能:
-- SET GLOBAL innodb_buffer_pool_size = 1G;  -- 根据服务器内存调整
-- SET GLOBAL innodb_flush_log_at_trx_commit = 2;  -- 提高写入性能
-- 
-- 优化查询缓存:
-- SET GLOBAL query_cache_size = 64M;
-- SET GLOBAL query_cache_type = 1;

-- 创建数据导入存储过程
DELIMITER //

CREATE PROCEDURE sp_import_etf_list(IN p_file_path VARCHAR(255), IN p_update_date DATE)
BEGIN
    -- 导入ETF列表数据的存储过程
    -- 参数:
    -- p_file_path: CSV文件路径
    -- p_update_date: 数据更新日期
    
    -- 实际实现需要根据MySQL服务器配置和权限调整
    -- 这里仅提供存储过程框架
    
    -- 示例:
    -- LOAD DATA INFILE p_file_path
    -- INTO TABLE etf_list
    -- FIELDS TERMINATED BY ','
    -- LINES TERMINATED BY '\n'
    -- IGNORE 1 LINES
    -- (code, name, latest_price, price_change, pct_change, buy_price, sell_price, 
    -- pre_close, open_price, high_price, low_price, volume, amount)
    -- SET update_date = p_update_date;
    
    SELECT CONCAT('导入ETF列表数据: ', p_file_path, ' 日期: ', p_update_date) AS message;
END //

CREATE PROCEDURE sp_import_etf_indicators(IN p_file_path VARCHAR(255), IN p_etf_code VARCHAR(6))
BEGIN
    -- 导入ETF指标数据的存储过程
    -- 参数:
    -- p_file_path: CSV文件路径
    -- p_etf_code: ETF代码
    
    SELECT CONCAT('导入ETF指标数据: ', p_file_path, ' ETF代码: ', p_etf_code) AS message;
END //

DELIMITER ;