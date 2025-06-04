"""
SEME项目配置模块

本模块定义了SEME项目的所有配置项，采用类的方式组织配置，支持通过环境变量覆盖默认值。
配置内容包括：数据库连接、JWT认证、文件上传限制、邮件服务、验证码等核心功能的参数设置。

配置加载流程：
1. 首先加载 .env 文件中的环境变量
2. 如果环境变量不存在，则使用默认值
3. 所有配置项都可以通过环境变量覆盖

使用示例：
    from config import Config
    app.config.from_object(Config)

环境变量示例：
    DB_USER=myuser
    DB_PASSWORD=mypassword
    JWT_SECRET_KEY=my-secret-key
    MAIL_USERNAME=sender@example.com

注意事项：
- 生产环境必须修改所有安全相关的默认值
- 敏感信息应通过环境变量设置，不要硬编码
- 数据库连接池大小要根据实际负载调整
- 文件上传限制要根据实际需求设置
"""

import os
from datetime import timedelta
from dotenv import load_dotenv

# 获取后端根目录的绝对路径
basedir = os.path.dirname(os.path.abspath(__file__))

# 加载环境变量配置文件
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    """
    应用配置类
    
    包含所有的应用配置项，通过类属性的方式组织。
    所有配置项都支持通过环境变量覆盖默认值。
    
    配置分类：
    1. 基础安全配置
    2. 数据库连接配置
    3. JWT认证配置
    4. 文件上传配置
    5. 验证码配置
    6. 邮件服务配置
    
    每个配置项都有详细的说明和默认值。
    """
    
    # ============================
    # 基础安全配置
    # ============================
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    
    # ============================
    # 数据库连接配置
    # ============================
    DB_USER = os.environ.get('DB_USER', 'root')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', 'your_password')
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_PORT = os.environ.get('DB_PORT', '3306')
    DB_NAME = os.environ.get('DB_NAME', 'db')
    
    # SQLAlchemy配置
    SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False  # 关闭SQLAlchemy的事件系统，提升性能
    SQLALCHEMY_POOL_SIZE = 10              # 连接池大小
    SQLALCHEMY_MAX_OVERFLOW = 20           # 连接池溢出大小
    
    # ============================
    # JWT认证配置
    # ============================
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-secret-string'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)     # 访问令牌1小时过期
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)    # 刷新令牌30天过期
    
    # ============================
    # 文件上传配置
    # ============================
    UPLOAD_FOLDER = os.path.join(basedir, 'uploads')  # 文件上传根目录
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024            # 最大文件大小限制：16MB
    
    # ============================
    # 验证码配置
    # ============================
    VERIFICATION_CODE_EXPIRE = 900         # 验证码有效期：15分钟
    VERIFICATION_CODE_RESEND_INTERVAL = 60 # 重发间隔：1分钟
    
    # ============================
    # 邮件服务配置
    # ============================
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.qq.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', '587'))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')   # 发件人邮箱
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')   # 邮箱授权码（QQ邮箱需要使用授权码）
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER')

# config.py - 配置文件
"""
系统配置文件，包含:
- 数据库连接配置
- JWT 令牌配置
- 文件上传配置
- 验证码配置
- 邮件服务器配置
所有配置都可通过环境变量覆盖
"""

# wsgi.py - WSGI入口文件
"""
Web应用程序入口文件:
- 配置日志系统
- 创建应用实例
- 注册CLI命令
- 启动应用服务器
"""

# app.py - 应用工厂
"""
Flask应用工厂模式实现:
- 初始化Flask扩展
- 配置数据库
- 配置JWT
- 配置CORS跨域
- 注册蓝图
- 加载预测模型
"""