import os
from datetime import timedelta
from dotenv import load_dotenv

# 使用相对路径
basedir = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(basedir, '.env'))

class Config:
    # 基础配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    
    # 数据库配置
    DB_USER = os.environ.get('DB_USER', 'root')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', 'your_password')
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_PORT = os.environ.get('DB_PORT', '3306')
    DB_NAME = os.environ.get('DB_NAME', 'db')
    
    SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_POOL_SIZE = 10
    SQLALCHEMY_MAX_OVERFLOW = 20
    
    # JWT配置
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-secret-string'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    
    # 文件上传配置
    UPLOAD_FOLDER = os.path.join(basedir, 'uploads')  # 使用绝对路径
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max-limit
    
    # 邮件验证码配置
    VERIFICATION_CODE_EXPIRE = 900  # 15分钟过期
    VERIFICATION_CODE_RESEND_INTERVAL = 60  # 1分钟后可重新发送
    
    # 邮件服务器配置
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.qq.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', '587'))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')  # 对于QQ邮箱，这里需要使用授权码
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