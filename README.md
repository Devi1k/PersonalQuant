# 量化交易系统 (Quantitative Trading System)

本项目是一个专注于中国市场行业ETF的量化交易系统，基于趋势识别、行业轮动和风险控制构建的自动化交易框架。

## 系统概述

本系统旨在通过技术分析和行业轮动策略，在中国ETF市场中捕捉行业轮动机会，实现超额收益。系统通过多维度的技术指标和行业强弱对比，优先选择强势行业的ETF进行交易，同时配备了严格的风险控制机制。

## 目标

- 构建一个完整的ETF行业轮动量化交易系统
- 实现自动化的数据获取、信号生成和交易执行流程
- 通过回测验证策略在不同市场环境下的表现
- 建立实时交易系统与预警机制

## 系统架构设计

### 1. 数据模块 (`src/data`)
- [x] 使用 AKShare 接入行业ETF历史数据、指数数据等 (`akshare_data.py`, `fetch_data.py`)
- [x] 数据清洗、处理与转换 (`data_processor.py`)
- [x] 数据存储至 MySQL 数据库 (`data_import.py`, `utils/db_utils.py`)
- [x] 提供统一数据接口 (`data_interface.py`)

### 2. 策略模块 (`src/strategy`)
- [x] 趋势策略实现 (`trend_strategy.py`)
- [x] 波段策略实现 (`swing_strategy.py`)
- [x] 策略融合机制 (`fusion_strategy.py`)
- [x] 风险管理模块 (`risk_manager.py`)
- [ ] 行业强弱对比系统 (待完善)
- [ ] 仓位控制模型 (待完善)

### 3. 回测模块 (`src/backtesting`)
- [x] 基于 Backtrader 的回测引擎 (`backtest_engine.py`)
- [x] 交易记录与绩效统计
- [x] 回测结果可视化 (依赖 Backtrader)

### 4. 辅助工具 (`src/utils`)
- [x] 数据库连接与操作工具 (`db_utils.py`)
- [x] 数据库写入测试脚本 (`db_write_test.py`)

## 实施路线图

### 第一阶段：基础建设
- [x] 选择数据源 (AKShare)
- [x] 搭建Python数据管道
  - [x] 历史数据下载模块
  - [x] 数据清洗与规范化
  - [x] 数据库存储实现
- [x] 构建基础回测框架
  - [x] 选择回测引擎 (Backtrader)
  - [x] 实现交易记录与绩效统计模块

### 第二阶段：策略开发
- [x] 趋势策略开发 (`trend_strategy.py`)
- [x] 波段策略开发 (`swing_strategy.py`)
- [x] 策略融合机制 (`fusion_strategy.py`)
- [x] 风险管理/止损系统 (`risk_manager.py`)
- [ ] 行业轮动模型 (待开发)

### 第三阶段：回测优化
- [ ] 测试周期选择与执行
- [ ] 关键参数敏感性分析
- [ ] 极端行情压力测试

### 第四阶段：实盘对接
- [ ] 开发定时任务
- [ ] 构建信号输出模块
- [ ] 开发异常监控

## 技术栈选择

- **编程语言**: Python
- **数据源**: AKshare
- **回测框架**: Backtrader
- **数据存储**: MySQL
- **可视化**: Matplotlib/Plotly

## 项目结构

```
PersonalQuant/
├── src/                  # 源代码
│   ├── data/             # 数据获取、处理与存储
│   │   ├── akshare_data.py
│   │   ├── data_import.py
│   │   ├── data_interface.py
│   │   ├── data_processor.py
│   │   └── fetch_data.py
│   ├── strategy/         # 交易策略与风险管理
│   │   ├── trend_strategy.py
│   │   ├── swing_strategy.py
│   │   ├── fusion_strategy.py
│   │   └── risk_manager.py
│   ├── backtesting/      # 回测引擎
│   │   └── backtest_engine.py
│   └── utils/            # 辅助工具
│       └── db_utils.py
│       └── db_write_test.py
├── data/                 # 数据存储 (示例，可按需调整)
│   ├── raw/
│   └── processed/
├── tests/                # 单元测试 (建议添加)
├── config/               # 配置文件 (建议添加)
├── schema/               # 数据库 Schema 定义
│   └── README.md
├── .gitignore
└── README.md             # 本文档
```

## 开发进度跟踪

(根据实际情况更新)

## 注意事项

- 确保数据质量和完整性
- 关注回测与实盘结果的差异
- 考虑交易成本和滑点因素
- 定期进行策略评估和调整
- 控制单日最大亏损和总风险敞口

## 资源链接

- [Backtrader文档](https://www.backtrader.com/docu/)