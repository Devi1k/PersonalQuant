# 量化交易数据库设计与使用说明

## 1. 数据库概述

本数据库设计用于存储和管理量化交易相关的数据，包括ETF基础信息、技术指标、市场情绪和行业资金流向等数据。数据库采用MySQL实现，支持按年分区存储，适合大规模历史数据存储和高效查询。

## 2. 数据库结构

### 2.1 核心表

| 表名 | 说明 | 分区策略 | 主键 |
|------|------|----------|------|
| etf_list | ETF基础信息表 | 无 | code |
| etf_indicators | ETF技术指标表 | 按年分区 | etf_code, trade_date |
| market_sentiment | 市场情绪数据表 | 按年分区 | trade_date |
| industry_fund_flow | 行业资金流向表 | 按年分区 | id (自增) |

### 2.2 视图

| 视图名 | 说明 |
|--------|------|
| v_latest_etf_indicators | 最新ETF指标视图 |
| v_market_overview | 市场情绪与资金流向综合视图 |

## 3. 安装与初始化

### 3.1 环境要求

- MySQL 5.7+ 或 MariaDB 10.3+
- Python 3.8+
- 必要的Python包（见requirements.txt）

### 3.2 安装步骤

1. 安装依赖包：

```bash
pip install -r requirements.txt
```

2. 配置数据库连接：

编辑 `src/data/data_import.py` 文件中的 `DB_CONFIG` 配置：

```python
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'your_password',  # 修改为实际密码
    'database': 'quant_db',
    'charset': 'utf8mb4'
}
```

3. 初始化数据库：

```bash
python src/data/data_import.py --init
```

## 4. 数据导入

### 4.1 导入ETF列表

```bash
python src/data/data_import.py --etf-list data/raw/etf_list_20250401.csv --date 20250401
```

### 4.2 导入ETF指标数据

```bash
python src/data/data_import.py --etf-indicators data/processed/etf_159915_with_indicators_20231231.csv --etf-code 159915
```

### 4.3 导入市场情绪数据

```bash
python src/data/data_import.py --market-sentiment data/processed/market_sentiment_20250401.csv
```

### 4.4 导入行业资金流向数据

```bash
python src/data/data_import.py --industry-flow data/processed/industry_fund_flow_20250401.csv
```

### 4.5 批量导入

可以编写Shell脚本或批处理文件进行批量导入，例如：

```bash
#!/bin/bash
# 批量导入示例

# 导入ETF列表
python src/data/data_import.py --etf-list data/raw/etf_list_20250401.csv --date 20250401

# 导入多个ETF指标数据
for etf_code in 159915 159919 159922 159937
do
    python src/data/data_import.py --etf-indicators data/processed/etf_${etf_code}_with_indicators_20231231.csv --etf-code ${etf_code}
done

# 导入市场情绪和行业资金流向
python src/data/data_import.py --market-sentiment data/processed/market_sentiment_20250401.csv
python src/data/data_import.py --industry-flow data/processed/industry_fund_flow_20250401.csv
```

## 5. 常用查询示例

### 5.1 查询最新ETF指标

```sql
SELECT * FROM v_latest_etf_indicators WHERE etf_code = '159915';
```

### 5.2 查询特定日期的市场情绪

```sql
SELECT * FROM market_sentiment 
WHERE trade_date BETWEEN '2025-03-01' AND '2025-04-01'
ORDER BY trade_date;
```

### 5.3 查询行业资金流向排名

```sql
SELECT industry_name, main_net_inflow, main_net_inflow_pct
FROM industry_fund_flow
WHERE trade_date = '2025-04-01'
ORDER BY main_net_inflow DESC
LIMIT 10;
```

### 5.4 查询ETF历史走势

```sql
SELECT trade_date, open, high, low, close, volume
FROM etf_indicators
WHERE etf_code = '159915'
  AND trade_date BETWEEN '2025-01-01' AND '2025-04-01'
ORDER BY trade_date;
```

## 6. 性能优化建议

1. **索引优化**：
   - 已为常用查询字段创建索引
   - 对于特定查询模式，可添加复合索引

2. **分区管理**：
   - 定期维护分区，删除过旧数据或归档
   - 可使用存储过程自动管理分区

3. **查询优化**：
   - 使用EXPLAIN分析查询计划
   - 避免SELECT *，只查询需要的列
   - 使用适当的WHERE条件限制结果集大小

4. **服务器配置**：
   - 调整innodb_buffer_pool_size（建议为系统内存的50-80%）
   - 优化query_cache_size提高查询性能

## 7. 数据库维护

### 7.1 备份

```bash
# 使用mysqldump备份整个数据库
mysqldump -u root -p quant_db > quant_db_backup_$(date +%Y%m%d).sql

# 备份特定表
mysqldump -u root -p quant_db etf_list > etf_list_backup_$(date +%Y%m%d).sql
```

### 7.2 恢复

```bash
# 恢复整个数据库
mysql -u root -p quant_db < quant_db_backup_20250401.sql

# 恢复特定表
mysql -u root -p quant_db < etf_list_backup_20250401.sql
```

## 8. 故障排除

1. **连接问题**：
   - 检查数据库服务是否运行
   - 验证用户名和密码
   - 确认主机和端口配置

2. **导入错误**：
   - 检查CSV文件格式是否正确
   - 确认字段名称与映射一致
   - 查看日志文件获取详细错误信息

3. **性能问题**：
   - 使用EXPLAIN分析慢查询
   - 检查索引使用情况
   - 考虑优化表结构或查询语句

## 9. 扩展与升级

数据库设计预留了扩展空间，可以通过以下方式进行升级：

1. 添加新的技术指标列
2. 创建新的视图满足特定分析需求
3. 添加新的表存储其他类型的数据
4. 实现存储过程自动化数据处理流程

## 10. 联系与支持

如有问题或建议，请联系项目维护者。

## 11. 数据库Schema

### 11.1 etf_list

**用途:** 存储市场上所有 ETF 的基本信息。

| 列名            | 类型           | 注释/说明       | 约束/索引       |
|-----------------|----------------|-----------------|-----------------|
| id              | INT            | 自增主键        | PRIMARY KEY     |
| code            | VARCHAR(8)     | ETF代码         | UNIQUE KEY `uk_code` |
| name            | VARCHAR(50)    | ETF名称         |                 |
| latest_price    | DECIMAL(10, 2) | 最新价          |                 |
| price_change    | DECIMAL(10, 2) | 涨跌额          |                 |
| pct_change      | DECIMAL(10, 4) | 涨跌幅(%)       |                 |
| buy_price       | DECIMAL(10, 2) | 买入价          |                 |
| sell_price      | DECIMAL(10, 2) | 卖出价          |                 |
| pre_close       | DECIMAL(10, 2) | 昨收            |                 |
| open_price      | DECIMAL(10, 2) | 今开            |                 |
| high_price      | DECIMAL(10, 2) | 最高            |                 |
| low_price       | DECIMAL(10, 2) | 最低            |                 |
| volume          | BIGINT         | 成交量(手)      |                 |
| amount          | DECIMAL(20, 2) | 成交额          |                 |
| update_date     | DATE           | 数据更新日期    | INDEX `idx_update_date` |

**引擎:** InnoDB

### 11.2 etf_indicators

**用途:** 存储 ETF 的每日技术分析指标。

| 列名                  | 类型           | 注释/说明                     | 约束/索引                  |
|-----------------------|----------------|-------------------------------|----------------------------|
| id                    | INT            | 自增ID                        | PRIMARY KEY (Part of)      |
| etf_code              | VARCHAR(8)     | ETF代码                       | UNIQUE KEY `uk_etf_date` (Part of) |
| trade_date            | DATE           | 交易日期                      | PRIMARY KEY (Part of), UNIQUE KEY `uk_etf_date` (Part of), INDEX `idx_trade_date`, PARTITION KEY |
| open                  | DECIMAL(10, 2) | 开盘价                        |                            |
| high                  | DECIMAL(10, 2) | 最高价                        |                            |
| low                   | DECIMAL(10, 2) | 最低价                        |                            |
| close                 | DECIMAL(10, 2) | 收盘价                        |                            |
| volume                | BIGINT         | 成交量                        |                            |
| daily_return          | DECIMAL(10, 4) | 日收益率                      |                            |
| volatility_20d        | DECIMAL(10, 4) | 20日波动率                    |                            |
| ma_5                  | DECIMAL(10, 2) | 5日均线                       |                            |
| ma_10                 | DECIMAL(10, 2) | 10日均线                      |                            |
| ma_20                 | DECIMAL(10, 2) | 20日均线                      |                            |
| ma_60                 | DECIMAL(10, 2) | 60日均线                      |                            |
| ema_12                | DECIMAL(10, 2) | 12日指数移动平均线            |                            |
| ema_21                | DECIMAL(10, 2) | 21日指数移动平均线(1个月期)   |                            |
| ema_26                | DECIMAL(10, 2) | 26日指数移动平均线            |                            |
| ema_144               | DECIMAL(10, 2) | 144日指数移动平均线           |                            |
| ema_144_upper         | DECIMAL(10, 2) | 144日EMA上通道              |                            |
| ema_144_lower         | DECIMAL(10, 2) | 144日EMA下通道              |                            |
| ema_200               | DECIMAL(10, 2) | 200日指数移动平均线           |                            |
| ema_21_200_diff       | DECIMAL(10, 4) | 21日与200日EMA差值          |                            |
| macd                  | DECIMAL(10, 4) | MACD值                        |                            |
| macd_signal           | DECIMAL(10, 4) | MACD信号线                    |                            |
| macd_hist             | DECIMAL(10, 4) | MACD柱状图                    |                            |
| rsi_14                | DECIMAL(10, 4) | 14日RSI                     |                            |
| bb_middle             | DECIMAL(10, 2) | 布林带中轨                    |                            |
| bb_std                | DECIMAL(10, 4) | 布林带标准差                  |                            |
| bb_upper              | DECIMAL(10, 2) | 布林带上轨                    |                            |
| bb_lower              | DECIMAL(10, 2) | 布林带下轨                    |                            |
| kdj_k                 | DECIMAL(10, 4) | KDJ指标K值                    |                            |
| kdj_d                 | DECIMAL(10, 4) | KDJ指标D值                    |                            |
| kdj_j                 | DECIMAL(10, 4) | KDJ指标J值                    |                            |
| cci_20                | DECIMAL(10, 4) | 20日CCI                     |                            |
| atr_14                | DECIMAL(10, 4) | 14日ATR                     |                            |
| obv                   | BIGINT         | OBV指标                       |                            |
| volume_ma_5           | BIGINT         | 成交量5日均线                 |                            |
| vwap_20               | DECIMAL(10, 2) | 20日成交量加权平均价格        |                            |

**主键:** `(id, trade_date)`
**唯一键:** `uk_etf_date (etf_code, trade_date)`
**索引:** `idx_trade_date (trade_date)`
**引擎:** InnoDB
**分区:** 按 `trade_date` 的年份进行 `RANGE` 分区 (p2020, p2021, ..., p_max)。
**外键:** 原计划有指向 `etf_list(code)` 的外键，但由于分区表限制已移除。

### 11.3 market_sentiment

**用途:** 存储每日的市场整体情绪指标。

| 列名              | 类型           | 注释/说明       | 约束/索引                      |
|-------------------|----------------|-----------------|--------------------------------|
| id                | INT            | 自增ID          | PRIMARY KEY (Part of)          |
| trade_date        | DATE           | 交易日期        | PRIMARY KEY (Part of), UNIQUE KEY `uk_trade_date`, PARTITION KEY |
| up_limit_count    | INT            | 涨停数量        |                                |
| down_limit_count  | INT            | 跌停数量        |                                |
| up_down_ratio     | DECIMAL(10, 4) | 涨跌比          |                                |
| margin_balance    | DECIMAL(20, 2) | 融资融券余额    |                                |

**主键:** `(id, trade_date)`
**唯一键:** `uk_trade_date (trade_date)`
**引擎:** InnoDB
**分区:** 按 `trade_date` 的年份进行 `RANGE` 分区 (p2020, p2021, ..., p_max)。

### 11.4 industry_fund_flow

**用途:** 存储每日各个行业的资金流入流出情况。

| 列名                        | 类型           | 注释/说明               | 约束/索引                                  |
|-----------------------------|----------------|-------------------------|--------------------------------------------|
| id                          | INT            | 序号                    | PRIMARY KEY (Part of)                      |
| industry_name               | VARCHAR(50)    | 行业名称                | UNIQUE KEY `uk_industry_date` (Part of), INDEX `idx_industry_name` |
| trade_date                  | DATE           | 交易日期                | PRIMARY KEY (Part of), UNIQUE KEY `uk_industry_date` (Part of), INDEX `idx_trade_date`, PARTITION KEY |
| price_change_pct            | DECIMAL(10, 4) | 今日涨跌幅              |                                            |
| main_net_inflow             | DECIMAL(20, 2) | 今日主力净流入-净额     |                                            |
| main_net_inflow_pct         | DECIMAL(10, 4) | 今日主力净流入-净占比   |                                            |
| super_large_net_inflow      | DECIMAL(20, 2) | 今日超大单净流入-净额   |                                            |
| super_large_net_inflow_pct  | DECIMAL(10, 4) | 今日超大单净流入-净占比 |                                            |
| large_net_inflow            | DECIMAL(20, 2) | 今日大单净流入-净额     |                                            |
| large_net_inflow_pct        | DECIMAL(10, 4) | 今日大单净流入-净占比   |                                            |
| medium_net_inflow           | DECIMAL(20, 2) | 今日中单净流入-净额     |                                            |
| medium_net_inflow_pct       | DECIMAL(10, 4) | 今日中单净流入-净占比   |                                            |
| small_net_inflow            | DECIMAL(20, 2) | 今日小单净流入-净额     |                                            |
| small_net_inflow_pct        | DECIMAL(10, 4) | 今日小单净流入-净占比   |                                            |
| top_stock                   | VARCHAR(50)    | 今日主力净流入最大股    |                                            |

**主键:** `(id, trade_date)`
**唯一键:** `uk_industry_date (industry_name, trade_date)`
**索引:** `idx_trade_date (trade_date)`, `idx_industry_name (industry_name)`
**引擎:** InnoDB
**分区:** 按 `trade_date` 的年份进行 `RANGE` 分区 (p2020, p2021, ..., p_max)。