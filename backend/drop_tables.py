import os
from dotenv import load_dotenv
import mysql.connector

# 加载环境变量
load_dotenv()

# 获取数据库配置
db_config = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'database': os.getenv('DB_NAME')
}

# 连接数据库
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# 禁用外键约束检查
cursor.execute('SET FOREIGN_KEY_CHECKS = 0')

# 删除所有表
tables = [
    'administrators', 'doctors', 'mri_seq_items', 'mri_sequences',
    'patients', 'pred_records', 'pred_doctor', 'pred_mri_item',
    'sequence_item', 'patient_sequence', 'alembic_version'
]

for table in tables:
    cursor.execute(f'DROP TABLE IF EXISTS {table}')

# 启用外键约束检查
cursor.execute('SET FOREIGN_KEY_CHECKS = 1')

conn.commit()
conn.close()

print("All tables have been dropped.") 