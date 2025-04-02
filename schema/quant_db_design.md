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

## 3. 分区策略
- 按年分区：每个自然年一个分区
- 历史分区：2023及之前
- 当前分区：2024
- 未来分区：p_max

## 4. 下一步建议
1. 确认设计方案
2. 切换到Code模式生成SQL文件
3. 执行数据库初始化