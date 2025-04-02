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