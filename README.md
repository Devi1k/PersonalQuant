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

### 1. 数据模块
- [x] 接入行业ETF历史数据（日线/分钟线）
- [x] 获取大盘指数数据（沪深300/中证500等）
- [x] 集成实时行情API（用于14:50前最后信号确认）

### 2. 策略模块
- [x] 趋势识别（均线系统/MACD/布林带）
- [x] 动量指标计算（RSI/ATR波动率）
- [ ] 行业强弱对比系统

### 3. 交易逻辑
- [x] 趋势策略
- [x] 波段策略
- [ ] 策略融合机制
  - [ ] 趋势与波段信号协同确认
  - [ ] 市场环境自适应切换
- [ ] 仓位控制模型（凯利公式改进版）

### 4. 风控模块
- [ ] 动态止损机制（追踪式/波动率自适应）
- [ ] 单日最大亏损熔断
- [ ] 行业分散度控制（不超过3个行业）

## 实施路线图

### 第一阶段：基础建设（1-2周）
- [x] 选择数据源（Tushare Pro/聚宽）
  - [x] 获取行业ETF清单
  - [x] 申请API密钥
  - [x] 测试API连接
- [x] 搭建Python数据管道
  - [x] 历史数据下载模块
  - [x] 实时数据订阅模块
  - [x] 数据清洗与规范化
- [x] 构建基础回测框架
  - [x] 选择/自建回测引擎（Backtrader/向量化回测）
  - [x] 实现交易记录与绩效统计模块
  - [x] 回测可视化功能

### 第二阶段：策略开发（2-3周）
- [x] 趋势策略开发:以趋势跟踪为主（60-80%仓位），多周期策略优化入场（20-30%），反转策略作为风险对冲（5-10%）。
  - [x] 布林带（Bollinger Bands）突破信号识别
  - [x] MACD长短周期移动平均线收敛与发散（1个月期与200日EMA）
  - [x] 多维周期组合策略（5分钟/15分钟/60分钟）
  - [x] 对冲型反转策略（基于144日EMA均线通道的突破交易）
  - [x] 量价齐升确认（成交量>5日均量*1.2）
  - [x] K线收盘价交易执行（减少滑点冲击）
- [x] 波段策略开发
  - [x] 多周期动量确认系统
    - [x] 短期动量：5分钟K线RSI（14周期）<30且KDJ-J线<20（超卖）+ 布林带下轨价格回归至中轨上方
    - [x] 中期过滤：15分钟K线需满足MACD柱线连续3根递增（防止短期假信号）
    - [x] 执行逻辑：观察60分钟K线是否形成双底形态（第二个底不创新低，且右底成交量>左底1.3倍）
  - [x] 通道回归均值策略
    - [x] 布林带（20日，2倍标准差）上轨做空/下轨做多，但仅在价格触碰通道后3根K线内回归中轨时生效
    - [x] 过滤条件：结合日线级别的KDJ（9,3,3）金叉/死叉提高胜率
    - [x] 风险控制：单次止损为ATR（14日）的1.5倍，若价格突破通道后20分钟未回归则强制止损
  - [x] 形态驱动反转交易
    - [x] 形态库识别：头肩底（右肩成交量>头部50%，颈线突破时成交量>5日均量2倍）
    - [x] 形态库识别：上升三角形（第3次回踩支撑线缩量至均量70%，突破时放量）
    - [x] 形态库识别：旗形整理（旗杆涨幅>15%，回调幅度<38.2%斐波那契位）
    - [x] 增强信号：形态完成后RSI出现背离（价格新低但RSI未新低，或新高但RSI未新高）
  - [x] 对冲型仓位平衡
    - [x] 对冲触发：当144日EMA通道上轨被突破且同时出现周线级别TD序列13计数
    - [x] 对冲方式：做空对应股指期货或买入认沽期权（Delta=0.3-0.4）
    - [x] 仓位权重：主策略持仓市值的15%-25%（动态调整波动率锥）
  - [x] 量价微观验证
    - [x] 入场校验：价格突破时需满足成交量>当日VWAP的120%，并结合Level2数据监控主力资金流向
    - [x] 出场优化：采用收盘价成交的同时，设定14:55的尾盘集中成交算法（TWAP策略降低滑点）
- [x] 策略融合机制
  - [x] 市场状态识别模块：使用技术指标: 例如 ADX (Average Directional Index) 可以衡量趋势强度，布林带宽度 (Bollinger Band Width) 或 ATR (Average True Range) 的变化可以反映波动性。当 ADX 高于某个阈值（如 25）时，可能表示趋势市场；低于该阈值则可能表示震荡市场。或者观察移动平均线的排列和斜率。
  趋势与波段信号权重分配
  - [x] 市场环境分类（震荡/单边/剧烈波动）
  - [x] 自适应策略切换阈值设定
- [ ] 行业轮动模型
  - [ ] 计算各行业ETF的5日收益率排名
  - [ ] 结合行业资金流向数据（L2数据优先）
  - [ ] 行业景气度评分系统
- [x] 止损系统
  - [x] 初始止损：入场价-2倍ATR
  - [x] 盈利保护：最高点回撤3%平仓
  - [x] 波动率自适应止损调整
  - [x] 策略失效标准评估（核心逻辑崩溃而非短期回撤）
  - [x] 因子优化（主力资金流入天数统计、突破新高判定等）

### 第三阶段：回测优化（1周）
- [ ] 测试周期选择
  - [ ] 包含牛熊周期（2018-2024）
  - [ ] 不同市场环境下的表现分析
  - [ ] 指数基准对比（如沪深300）
- [ ] 关键参数敏感性分析
  - [ ] 持仓周期（1-5日测试）
  - [ ] 仓位权重（30%-70%动态调整）
  - [ ] 止损参数优化
- [ ] 极端行情压力测试
  - [ ] 2015股灾场景模拟
  - [ ] 2020疫情行情回测
  - [ ] 流动性风险评估

### 第四阶段：实盘对接（1周）
- [ ] 开发定时任务
  - [ ] 每天14:30启动策略计算
  - [ ] 数据更新与信号生成流程
  - [ ] 任务状态监控
- [ ] 构建信号输出模块
  - [ ] CSV结构化输出
  - [ ] 微信API通知集成
  - [ ] 信号详情与解释
- [ ] 开发异常监控
  - [ ] 网络中断检测
  - [ ] 数据异常报警
  - [ ] 系统恢复机制

## 技术栈选择

- **编程语言**: Python
- **数据源**: AKshare
- **回测框架**: Backtrader
- **数据存储**: MySQL
- **可视化**: Matplotlib/Plotly

## 项目结构

```
PersonalQuant/
├── data/                 # 数据存储
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后的数据
├── src/                  # 源代码
│   ├── data/             # 数据获取与处理
│   ├── features/         # 特征工程
│   ├── models/           # 策略模型
│   ├── backtesting/      # 回测框架
│   └── trading/          # 交易执行
├── notebooks/            # Jupyter笔记本
├── tests/                # 测试代码
├── config/               # 配置文件
└── docs/                 # 文档
```

## 开发进度跟踪

- [ ] 第一阶段: 基础建设 (计划完成日期: _________)
- [ ] 第二阶段: 策略开发 (计划完成日期: _________)
- [ ] 第三阶段: 回测优化 (计划完成日期: _________)
- [ ] 第四阶段: 实盘对接 (计划完成日期: _________)

## 注意事项

- 确保数据质量和完整性
- 关注回测与实盘结果的差异
- 考虑交易成本和滑点因素
- 定期进行策略评估和调整
- 控制单日最大亏损和总风险敞口

## 资源链接

- [Backtrader文档](https://www.backtrader.com/docu/)