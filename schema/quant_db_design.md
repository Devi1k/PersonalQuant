# 量化交易数据库设计方案

## 1. 数据库架构图
```mermaid
erDiagram
    etf_list ||--o{ etf_indicators : "代码"
    market_sentiment ||--o{ industry_fund_flow : "日期"

    etf_list {
        varchar(6) code PK "ETF代码"
        varchar(50) name "名称"
        decimal(10,2) price "最新价"
        date update_date "更新日期"
    }

    etf_indicators {
        date trade_date PK "交易日"
        varchar(6) etf_code PK "ETF代码"
        decimal(10,2) open
        decimal(10,2) high
        decimal(10,2) low
        decimal(10,2) close
        bigint volume "成交量(手)"
        decimal(5,2) daily_return "日收益率(%)"
        decimal(15,4) macd
        decimal(15,4) macd_signal
        decimal(15,4) rsi_14
        index idx_etf_date (etf_code, trade_date)
    }
    PARTITION BY RANGE (YEAR(trade_date)) (...)
```

## 2. 核心表结构

### 2.1 ETF基础信息表 (etf_list)
| 字段名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| code | varchar(6) | ETF代码 | PK |
| name | varchar(50) | ETF名称 | NOT NULL |
| price | decimal(10,2) | 最新价格 |  |
| update_date | date | 数据更新日期 |  |

### 2.2 ETF指标表 (etf_indicators)
```sql
CREATE TABLE etf_indicators (
  trade_date DATE NOT NULL,
  etf_code VARCHAR(6) NOT NULL,
  open DECIMAL(10,2),
  high DECIMAL(10,2),
  low DECIMAL(10,2),
  close DECIMAL(10,2),
  volume BIGINT,
  -- 常规技术指标
  ma_5 DECIMAL(10,2),
  ma_10 DECIMAL(10,2),
  ma_20 DECIMAL(10,2),
  ema_12 DECIMAL(10,2),
  ema_26 DECIMAL(10,2),
  macd DECIMAL(10,4),
  rsi_14 DECIMAL(10,4),
  -- 资金流量指标
  mfi_3 DECIMAL(10,4) COMMENT '3日资金流量指标',
  mfi_5 DECIMAL(10,4) COMMENT '5日资金流量指标',
  mfi_7 DECIMAL(10,4) COMMENT '7日资金流量指标',
  mfi_14 DECIMAL(10,4) COMMENT '14日资金流量指标',
  -- 其他技术指标字段...
  PRIMARY KEY (trade_date, etf_code),
  INDEX idx_etf (etf_code)
) ENGINE=InnoDB
PARTITION BY RANGE (YEAR(trade_date)) (
  PARTITION p2023 VALUES LESS THAN (2024),
  PARTITION p2024 VALUES LESS THAN (2025),
  PARTITION p_max VALUES LESS THAN MAXVALUE
);
```

#### ETF指标表主要技术指标

| 指标类别 | 指标名称 | 说明 |
|--------|------|------|
| 趋势指标 | MA (移动平均线) | 5日、10日、20日、60日均线 |
| 趋势指标 | EMA (指数移动平均线) | 12日、21日、26日、144日、200日 |
| 趋势指标 | MACD | MACD值、信号线、柱状图 |
| 震荡指标 | RSI | 14日相对强弱指标 |
| 震荡指标 | KDJ | K值、D值、J值 |
| 震荡指标 | CCI | 20日顺势指标 |
| 波动指标 | BB (布林带) | 中轨、上轨、下轨、标准差 |
| 波动指标 | ATR | 14日真实波动幅度均值 |
| 成交量指标 | OBV | 能量潮指标 |
| 成交量指标 | Volume MA | 成交量移动平均线 |
| 资金流向指标 | MFI | 3日、5日、7日、14日资金流量指标 |

## 3. 分区策略
- 按年分区：每个自然年一个分区
- 历史分区：2023及之前
- 当前分区：2024
- 未来分区：p_max

### 2.3 板块指数与统计信息表 (sector_data)

该表合并了板块/概念指数数据和板块统计信息，用于记录ETF所属板块/概念的指数数据及每个板块内的涨跌停家数统计和成交情况。

| 字段名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| id | INT | 自增ID | PK |
| sector_id | VARCHAR(20) | 板块ID | NOT NULL |
| sector_name | VARCHAR(50) | 板块名称 | NOT NULL |
| trade_date | DATE | 交易日期 | PK |
| index_value | DECIMAL(10,2) | 指数值 | |
| change_pct_1d | DECIMAL(10,4) | 1日涨跌幅 | |
| change_pct_5d | DECIMAL(10,4) | 5日涨跌幅 | |
| change_pct_10d | DECIMAL(10,4) | 10日涨跌幅 | |
| up_limit_count | INT | 涨停家数 | |
| down_limit_count | INT | 跌停家数 | |
| up_down_ratio | DECIMAL(10,4) | 涨跌比 | |
| volume | BIGINT | 成交量 | |
| amount | DECIMAL(20,2) | 成交额 | |

```sql
CREATE TABLE sector_data (
  id INT NOT NULL AUTO_INCREMENT COMMENT '自增ID',
  sector_id VARCHAR(20) NOT NULL COMMENT '板块ID',
  sector_name VARCHAR(50) NOT NULL COMMENT '板块名称',
  trade_date DATE NOT NULL COMMENT '交易日期',
  -- 其他字段...
  PRIMARY KEY (id, trade_date),
  UNIQUE KEY uk_sector_date (sector_id, trade_date)
) ENGINE=InnoDB
PARTITION BY RANGE (YEAR(trade_date)) (...)
```

## 4. 下一步建议
1. 确认设计方案
2. 切换到Code模式生成SQL文件
3. 执行数据库初始化