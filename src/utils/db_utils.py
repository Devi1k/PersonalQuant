import yaml
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import MetaData, Table 
from loguru import logger
from pathlib import Path
import warnings
from datetime import datetime, date

# 忽略 pandas 关于 SQLAlchemy 2.0 的警告
warnings.filterwarnings("ignore", category=UserWarning, module="pandas.io.sql")

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"

def load_config(config_path=CONFIG_PATH):
    """加载 YAML 配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if not config:
            logger.error(f"配置文件为空或加载失败: {config_path}")
            return None
        return config
    except FileNotFoundError:
        logger.error(f"配置文件未找到: {config_path}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"解析配置文件时出错: {e}")
        return None

def get_db_engine(config=None):
    """根据配置创建 SQLAlchemy 数据库引擎"""
    if config is None:
        config = load_config()

    if not config or 'database' not in config:
        logger.error("数据库配置未找到或不完整。")
        return None

    db_config = config['database']
    db_type = db_config.get('type')
    username = db_config.get('username')
    password = db_config.get('password')
    host = db_config.get('host')
    port = db_config.get('port')
    database = db_config.get('database')

    if not all([db_type, username, password, host, port, database]):
        logger.error("数据库连接信息不完整 (type, username, password, host, port, database)。")
        return None

    if password == "YOUR_PASSWORD":
         logger.warning("数据库密码未在 config.yaml 中设置，请将其替换为您的真实密码。")
         # 可以选择在这里返回 None 或抛出异常，阻止继续执行
         # return None

    try:
        if db_type == 'mysql':
            # 使用 pymysql 驱动
            connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}?charset=utf8mb4"
            engine = create_engine(connection_string, pool_pre_ping=True)
            # 测试连接
            with engine.connect() as connection:
                logger.info(f"成功连接到 MySQL 数据库: {database}@{host}:{port}")
            return engine
        elif db_type == 'postgresql':
            # 使用 psycopg2 驱动 (需要 pip install psycopg2-binary)
            connection_string = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
            engine = create_engine(connection_string, pool_pre_ping=True)
            with engine.connect() as connection:
                 logger.info(f"成功连接到 PostgreSQL 数据库: {database}@{host}:{port}")
            return engine
        elif db_type == 'sqlite':
            # SQLite 连接字符串是文件路径
            db_path = Path(database) # 假设 database 字段存的是相对或绝对路径
            db_path.parent.mkdir(parents=True, exist_ok=True) # 确保目录存在
            connection_string = f"sqlite:///{db_path.resolve()}"
            engine = create_engine(connection_string)
            with engine.connect() as connection:
                 logger.info(f"成功连接到 SQLite 数据库: {db_path.resolve()}")
            return engine
        else:
            logger.error(f"不支持的数据库类型: {db_type}")
            return None
    except ImportError as e:
        logger.error(f"缺少数据库驱动: {e}. 请确保已安装相应的库 (如 pymysql, psycopg2-binary)。")
        return None
    except SQLAlchemyError as e:
        logger.error(f"数据库连接失败: {e}")
        return None
    except Exception as e:
        logger.error(f"创建数据库引擎时发生未知错误: {e}")
        return None


def df_to_sql(df: pd.DataFrame, table_name: str, engine, if_exists: str = 'append', index: bool = False, chunksize: int = 1000):
    """
    将 DataFrame 写入 SQL 数据库表。

    Args:
        df (pd.DataFrame): 要写入的数据。
        table_name (str): 目标数据库表名。
        engine: SQLAlchemy 数据库引擎。
        if_exists (str): 如果表已存在时的行为 ('fail', 'replace', 'append')。默认为 'append'。
                         注意：'replace' 会先 DROP 表再 CREATE，请谨慎使用。
        index (bool): 是否将 DataFrame 索引写入数据库。默认为 False。
        chunksize (int): 每次写入的行数。默认为 1000。

    Returns:
        bool: 写入成功返回 True，失败返回 False。
    """
    if df.empty:
        logger.info(f"DataFrame 为空，无需写入数据库表 {table_name}。")
        return True

    if engine is None:
        logger.error("数据库引擎无效，无法写入数据。")
        return False

    try:
        # 在写入前将数字和日期转换为字符串
        df_copy = df.copy()
        
        # 将数字类型转换为字符串
        for col in df_copy.select_dtypes(include=['int', 'float', 'int64', 'float64']).columns:
            df_copy[col] = df_copy[col].astype(str)
        
        # 将日期类型转换为字符串 (ISO 格式)
        for col in df_copy.select_dtypes(include=['datetime', 'datetime64', 'datetime64[ns]']).columns:
            df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 检查并转换 Python date 对象
        for col in df_copy.columns:
            # 检查列的第一个非空值是否为 datetime.date 类型
            sample_vals = df_copy[col].dropna()
            if len(sample_vals) > 0 and isinstance(sample_vals.iloc[0], date) and not isinstance(sample_vals.iloc[0], datetime):
                df_copy[col] = df_copy[col].astype(str)
        
        # 尝试写入数据
        df_copy.to_sql(name=table_name, con=engine, if_exists=if_exists, index=index, chunksize=chunksize, method='multi')
        logger.info(f"成功将 {len(df_copy)} 条记录写入数据库表 {table_name} (策略: {if_exists})。")
        return True
    except SQLAlchemyError as e:
        logger.error(f"写入数据库表 {table_name} 时出错: {e}")
        # 这里可以根据具体错误类型进行更细致的处理，例如处理重复键冲突
        # 对于 MySQL 的 ON DUPLICATE KEY UPDATE，需要更复杂的逻辑，可能需要原生 SQL 或 ORM 层面的支持
        # Pandas 的 to_sql 本身不直接支持 ON DUPLICATE KEY UPDATE
        # 一个简化的处理方式是先删除可能冲突的数据，但这依赖于主键或唯一键信息
        return False
    except Exception as e:
        logger.error(f"写入数据库时发生未知错误: {e}")
        return False

# --- 针对特定表的写入函数 (处理重复键更新) ---

def upsert_df_to_sql(df: pd.DataFrame, table_name: str, engine, unique_columns: list):
    """通用函数，用于将 DataFrame 数据 upsert 到 SQL 表中。

    Args:
        df (pd.DataFrame): 需要插入或更新的数据。
        table_name (str): 目标数据库表名。
        engine: SQLAlchemy 数据库引擎。
        unique_columns (list): 用于判断记录唯一性的列名列表。
                         如果记录已存在（基于 unique_columns），则更新其他列；
                         否则，插入新记录。

    Returns:
        bool: 操作是否成功。
    """
    if df.empty:
        logger.info(f"输入 DataFrame 为空，无需写入表 {table_name}。")
        return True

    df_copy = df.copy()

    # --- 数据预处理 ---
    # 处理 NaTType: MySQL 不直接支持 NaT，通常转换为 None 或特定日期字符串
    # 这里选择转换为 None，让数据库处理（假设列允许 NULL 或有默认值）
    # 注意：确保数据库列类型兼容 None (NULL)
    for col in df_copy.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
        df_copy[col] = df_copy[col].apply(lambda x: None if pd.isna(x) else x)

    # 处理所有NaN/inf值，转换为None (MySQL中的NULL)
    # 这解决了pymysql.err.ProgrammingError: nan can not be used with MySQL的问题
    import numpy as np
    df_copy = df_copy.replace([np.inf, -np.inf], np.nan)
    # 确保所有 NaN 值被转换为 None，先转换为 object 类型
    df_copy = df_copy.astype(object)
    df_copy = df_copy.where(pd.notnull(df_copy), None)

    # --- 构建 SQL 语句 ---
    cols = df_copy.columns.tolist()
    cols_str = ", ".join([f"`{c}`" for c in cols])
    # 使用命名占位符 :col_name
    placeholders = ", ".join([f":{c}" for c in cols])

    sql = f"INSERT INTO `{table_name}` ({cols_str}) VALUES ({placeholders})"

    # 构建 ON DUPLICATE KEY UPDATE 部分
    update_statements = []
    for col in cols:
        # unique_columns 不应在 UPDATE 部分更新
        if col not in unique_columns:
            # 对于 UPDATE 部分，我们引用 VALUES(col_name)，而不是占位符
            update_statements.append(f"`{col}`=VALUES(`{col}`)")

    if update_statements:
        sql += " ON DUPLICATE KEY UPDATE " + ", ".join(update_statements)
    else:
        # 如果所有列都是 unique_columns，则无需 UPDATE，但这种情况很少见
        # MySQL 需要至少一个更新字段，否则会报错。可以添加一个无关紧要的更新或采取其他策略。
        # 例如: 添加一个 update_timestamp 列并更新它
        # 或者简单地忽略重复：INSERT IGNORE INTO ... (但这会丢失更新)
        # 这里假设总有非 unique 列需要更新，或者表结构保证了这一点。
        # 如果可能没有非 unique 列，需要调整逻辑。
        logger.warning(f"表 {table_name} 的所有列都在 unique_columns 中，ON DUPLICATE KEY UPDATE 部分为空。请检查逻辑。")
        # 考虑添加一个虚拟更新，如 `id`=LAST_INSERT_ID(`id`) 如果有自增主键 id
        # 或者如果业务允许，改为 INSERT IGNORE
        # sql = f"INSERT IGNORE INTO `{table_name}` ({cols_str}) VALUES ({placeholders})"

    # --- 准备数据 ---
    # 将 DataFrame 转换为字典列表
    data_dicts = df_copy.to_dict(orient='records')

    # --- 执行数据库操作 ---
    conn = None
    trans = None
    try:
        conn = engine.connect()
        trans = conn.begin()

        # 一次性传递整个字典列表给 execute
        if data_dicts: # 确保列表不为空
            # 使用 text() 包装 SQL，并传递字典列表作为参数
            conn.execute(text(sql), data_dicts)

        trans.commit()
        logger.info(f"成功 upsert {len(df_copy)} 条记录到数据库表 {table_name}。")
        return True
    except SQLAlchemyError as e:
        logger.error(f"Upsert 数据到表 {table_name} 时出错: {e}")
        if trans:
            trans.rollback()
        return False
    except Exception as e: # 捕获其他可能的错误
        logger.error(f"Upsert 数据到表 {table_name} 时发生意外错误: {e}")
        if trans:
            trans.rollback()
        return False
    finally:
        if conn:
            conn.close()


def query_etf_history(engine, etf_code, start_date=None, end_date=None, fields=None):
    """
    从数据库中查询ETF历史数据

    Args:
        engine: SQLAlchemy 数据库引擎
        etf_code (str): ETF代码，如 "510300"
        start_date (str, optional): 开始日期，格式 "YYYY-MM-DD"
        end_date (str, optional): 结束日期，格式 "YYYY-MM-DD"
        fields (list, optional): 需要查询的字段列表，默认为全部字段

    Returns:
        pandas.DataFrame: 查询结果
    """
    if not engine:
        logger.error("数据库引擎为空，无法查询ETF历史数据")
        return pd.DataFrame()

    try:
        # 构建基本查询
        if fields and isinstance(fields, list):
            # 确保必要的字段总是被包含
            if 'etf_code' not in fields:
                fields.append('etf_code')
            if 'trade_date' not in fields:
                fields.append('trade_date')
            columns_str = ", ".join(fields)
        else:
            columns_str = "*"  # 查询所有字段

        # 构建查询条件
        conditions = [f"etf_code = '{etf_code}'"]
        if start_date:
            conditions.append(f"trade_date >= '{start_date}'")
        if end_date:
            conditions.append(f"trade_date <= '{end_date}'")

        where_clause = " AND ".join(conditions)

        # 构建完整的SQL查询
        query = f"""
        SELECT {columns_str}
        FROM etf_indicators
        WHERE {where_clause}
        ORDER BY trade_date
        """

        # 执行查询
        logger.info(f"查询ETF {etf_code} 的历史数据，时间范围: {start_date or '最早'} 至 {end_date or '最新'}")
        df = pd.read_sql(query, engine)

        if df.empty:
            logger.warning(f"未找到ETF {etf_code} 的历史数据")
            return pd.DataFrame()

        # 转换日期列为日期类型
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])

        logger.info(f"成功查询到ETF {etf_code} 的历史数据，共 {len(df)} 条记录")
        return df

    except Exception as e:
        logger.error(f"查询ETF {etf_code} 的历史数据时出错: {e}")
        return pd.DataFrame()

def query_etf_list(engine):
    """
    从数据库中查询 ETF List 表的所有内容
    
    Args:
        engine: SQLAlchemy 数据库引擎
        
    Returns:
        pandas.DataFrame: 包含所有 ETF 基础信息的 DataFrame，如果查询失败则返回空 DataFrame
    """
    if engine is None:
        logger.error("数据库引擎无效，无法查询 ETF 列表。")
        return pd.DataFrame()
    
    try:
        # 构建查询
        query = "SELECT * FROM etf_list"
        
        # 执行查询
        df = pd.read_sql(query, engine)
        logger.info(f"成功从数据库查询到 {len(df)} 条 ETF 记录。")
        return df
    except SQLAlchemyError as e:
        logger.error(f"查询 ETF 列表时发生错误: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"查询 ETF 列表时发生未知错误: {e}")
        return pd.DataFrame()

def query_minute_kline_data(engine, etf_code, period, start_date=None, end_date=None):
    """
    从数据库中查询分钟级别K线数据和指标
    
    Args:
        engine: SQLAlchemy 数据库引擎
        etf_code (str): ETF代码，如 "510050"
        period (int): K线周期，支持 5（5分钟）、15（15分钟）、60（60分钟）
        start_date (str, optional): 开始日期，格式 "YYYY-MM-DD"
        end_date (str, optional): 结束日期，格式 "YYYY-MM-DD"
        
    Returns:
        pandas.DataFrame: 包含分钟级别K线数据和指标的DataFrame，如果查询失败则返回空DataFrame
    """
    if engine is None:
        logger.error("数据库引擎无效，无法查询分钟级别K线数据。")
        return pd.DataFrame()
    
    try:
        # 构建基础查询
        query = """
        SELECT 
            CONCAT(k.trade_date, ' ', k.trade_time) AS date,
            k.close,
            k.open,
            k.high,
            k.low,
            k.volume,
            i.rsi_14,
            i.bb_lower,
            i.bb_middle,
            i.bb_upper
        FROM 
            minute_kline_data k
        LEFT JOIN 
            minute_indicators i 
        ON 
            k.etf_code = i.etf_code AND 
            k.trade_date = i.trade_date AND 
            k.trade_time = i.trade_time AND 
            k.period = i.period
        WHERE 
            k.etf_code = :etf_code AND 
            k.period = :period
        """
        
        # 添加日期过滤条件
        params = {"etf_code": etf_code, "period": period}
        
        if start_date:
            query += " AND k.trade_date >= :start_date"
            params["start_date"] = start_date
            
        if end_date:
            query += " AND k.trade_date <= :end_date"
            params["end_date"] = end_date
        
        # 添加排序
        query += " ORDER BY k.trade_date, k.trade_time"
        
        # 执行查询
        df = pd.read_sql(text(query), engine, params=params)
        
        # 转换日期列为datetime类型
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        logger.info(f"成功查询到ETF {etf_code} 的 {period} 分钟K线数据，共 {len(df)} 条记录。")
        return df
    except SQLAlchemyError as e:
        logger.error(f"查询分钟级别K线数据时发生错误: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"查询分钟级别K线数据时发生未知错误: {e}")
        return pd.DataFrame()


# --- 测试代码 (可选) ---
if __name__ == '__main__':
    # 测试代码
    logger.add("logs/db_utils_test.log", rotation="5 MB") # 添加日志文件输出

    test_config = load_config()
    if test_config:
        logger.info("配置文件加载成功。")
        test_engine = get_db_engine(test_config)
        if test_engine:
            logger.info("数据库引擎创建成功。")

            # --- 测试 df_to_sql (append) ---
            # logger.info("--- 测试 df_to_sql (append) ---")
            # test_data_append = {'col1': [1, 2], 'col2': ['A', 'B']}
            # test_df_append = pd.DataFrame(test_data_append)
            # # 假设存在一个测试表 test_append_table (col1 INT PRIMARY KEY, col2 VARCHAR(10))
            # # 请手动创建此表用于测试: CREATE TABLE test_append_table (col1 INT PRIMARY KEY, col2 VARCHAR(10));
            # # df_to_sql(test_df_append, 'test_append_table', test_engine, if_exists='append', index=False)

            # # --- 测试 upsert_df_to_sql ---
            # logger.info("--- 测试 upsert_df_to_sql ---")
            # test_data_upsert = {
            #     'etf_code': ['510300', '159915', '510300'], # 重复的 etf_code
            #     'trade_date': ['2024-01-01', '2024-01-01', '2024-01-02'],
            #     'close': [3.50, 1.20, 3.55],
            #     'volume': [10000, 5000, 12000]
            # }
            # test_df_upsert = pd.DataFrame(test_data_upsert)
            # 假设 etf_indicators 表存在，并且 (etf_code, trade_date) 是唯一键或主键
            # upsert_df_to_sql(test_df_upsert, 'etf_indicators', test_engine, unique_columns=['etf_code', 'trade_date'])

            # --- 测试 query_etf_history ---
            logger.info("--- 测试 query_etf_history ---")
            # 测试查询所有字段
            logger.info("测试查询所有字段:")
            test_result_all = query_etf_history(
                engine=test_engine,
                etf_code='510300',
                start_date='2024-01-01',
                end_date='2024-01-31'
            )
            if not test_result_all.empty:
                logger.info(f"查询成功，获取到 {len(test_result_all)} 条记录")
                logger.info(f"字段列表: {list(test_result_all.columns)}")
                logger.info(f"前3条记录:\n{test_result_all.head(3)}")
            else:
                logger.warning("查询结果为空，请确认数据库中是否有相关数据")
            
            # 测试查询特定字段
            logger.info("测试查询特定字段:")
            test_fields = ['etf_code', 'trade_date', 'open', 'high', 'low', 'close', 'volume']
            test_result_fields = query_etf_history(
                engine=test_engine,
                etf_code='510300',
                start_date='2024-01-01',
                end_date='2024-01-31',
                fields=test_fields
            )
            if not test_result_fields.empty:
                logger.info(f"查询成功，获取到 {len(test_result_fields)} 条记录")
                logger.info(f"字段列表: {list(test_result_fields.columns)}")
                logger.info(f"前3条记录:\n{test_result_fields.head(3)}")
            else:
                logger.warning("查询结果为空，请确认数据库中是否有相关数据")
            
            # 测试不同的ETF代码
            logger.info("测试查询不同的ETF代码:")
            test_result_other = query_etf_history(
                engine=test_engine,
                etf_code='159915',
                start_date='2024-01-01',
                end_date='2024-01-31'
            )
            if not test_result_other.empty:
                logger.info(f"查询成功，获取到 {len(test_result_other)} 条记录")
                logger.info(f"前3条记录:\n{test_result_other.head(3)}")
            else:
                logger.warning("查询结果为空，请确认数据库中是否有相关数据")

            logger.info("测试完成。请检查数据库和日志文件 'logs/db_utils_test.log'。")
        else:
            logger.error("数据库引擎创建失败，测试中止。")
    else:
        logger.error("配置文件加载失败，测试中止。")
