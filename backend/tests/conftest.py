"""
测试配置和公共测试工具模块 (Test Configuration and Utilities)

本模块为整个测试套件提供基础设施，包括：
1. 测试环境配置
2. 测试固件(Fixtures)
3. 测试数据工厂
4. 测试辅助函数

主要组件：
1. 配置管理
   - 测试专用配置类
   - 环境变量设置
   - 资源路径配置

2. 数据库管理
   - 内存数据库配置
   - 测试数据初始化
   - 数据清理机制

3. 认证机制
   - JWT令牌生成
   - 测试用户创建
   - 权限管理

4. 文件系统
   - 临时目录管理
   - 文件上传模拟
   - 资源清理

测试设计理念：
1. 隔离性
   - 独立的测试环境
   - 测试间无干扰
   - 状态自动重置

2. 可重复性
   - 确定性结果
   - 自动化设置
   - 统一的配置

3. 易用性
   - 共享固件
   - 工厂模式
   - 辅助功能

作者: Your Name
创建日期: 2025-06-16
"""

import os
import pytest
from app import create_app, db
from config import Config
import tempfile
from tests.factories.patient import PatientFactory
from app.models import Doctor, Administrator
from datetime import timedelta
from werkzeug.security import generate_password_hash

class TestConfig(Config):
    """
    测试专用配置类
    
    继承自基础配置类(Config)，添加和覆盖测试所需的特定配置。
    
    配置项：
    1. 基础设置
       - TESTING: 启用测试模式
       - SECRET_KEY: 测试密钥
       - WTF_CSRF_ENABLED: 禁用CSRF保护
    
    2. 数据库配置
       - SQLALCHEMY_DATABASE_URI: 内存SQLite数据库
       - 自动提交设置
       - 事务管理
    
    3. 认证配置
       - JWT_SECRET_KEY: JWT测试密钥
       - JWT_ACCESS_TOKEN_EXPIRES: 令牌过期时间
    
    4. 文件存储
       - UPLOAD_FOLDER: 上传文件目录
       - 临时文件配置
    
    使用说明：
    >>> app = create_app(TestConfig)
    >>> app.config['TESTING']
    True
    """
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'  # 使用内存数据库
    WTF_CSRF_ENABLED = False
    SECRET_KEY = 'test-key'
    JWT_SECRET_KEY = 'test-jwt-key'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    UPLOAD_FOLDER = 'uploads'

@pytest.fixture
def app():
    """
    创建测试应用的Fixture
    
    功能：
    1. 环境设置
       - 创建临时上传目录
       - 配置应用实例
       - 初始化数据库
    
    2. 数据准备
       - 创建数据库表
       - 添加测试用户
       - 设置初始状态
    
    3. 资源清理
       - 删除数据库表
       - 清理临时文件
       - 释放资源
    
    生命周期：
    1. 设置阶段
       - 创建临时目录
       - 初始化应用
       - 准备数据库
    
    2. 测试执行
       - yield应用实例
       - 测试代码运行
    
    3. 清理阶段
       - 删除数据库
       - 删除临时文件
       - 清理会话
    
    使用示例：
    def test_something(app):
        with app.app_context():
            # 执行测试逻辑
            assert app.config['TESTING']
    """
    # 创建临时上传目录
    uploads_dir = tempfile.mkdtemp()
    TestConfig.UPLOAD_FOLDER = uploads_dir
    
    # 创建应用实例
    app = create_app(TestConfig)
    
    # 创建必要的数据库表和测试数据
    with app.app_context():
        # 创建所有表
        db.create_all()
        
        # 创建测试医生用户
        doctor = Doctor(
            doctor_id='doctor_1',
            name='Test Doctor',
            email='test@example.com',
            department='Test Department',
            password_hash=generate_password_hash('test_password')
        )
        db.session.add(doctor)

        # 创建测试管理员用户
        admin = Administrator(
            admin_id='admin_1',
            name='Test Admin',
            email='admin@example.com',
            password_hash=generate_password_hash('admin_password')
        )
        db.session.add(admin)
        
        # 提交更改
        db.session.commit()
    
    yield app
    
    # 清理资源
    with app.app_context():
        db.session.remove()
        db.drop_all()
    os.rmdir(uploads_dir)

@pytest.fixture
def client(app):
    """
    测试客户端Fixture
    
    功能：
    - 提供测试HTTP客户端
    - 继承应用配置
    - 支持请求测试
    
    用法示例：
    def test_home_page(client):
        response = client.get('/')
        assert response.status_code == 200
    """
    return app.test_client()

@pytest.fixture
def auth_headers(app):
    """
    认证头部Fixture
    
    功能：
    1. 生成JWT令牌
       - 使用测试用户ID
       - 设置过期时间
       - 包含必要声明
    
    2. 构造请求头
       - Bearer认证格式
       - 标准JWT结构
    
    用法示例：
    def test_protected_route(client, auth_headers):
        response = client.get('/api/protected', headers=auth_headers)
        assert response.status_code == 200
    """
    from flask_jwt_extended import create_access_token
    with app.app_context():
        # 创建测试令牌
        access_token = create_access_token('doctor_1')
        return {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }

@pytest.fixture
def test_db(app):
    """
    测试数据库会话Fixture
    
    功能：
    - 提供数据库会话
    - 自动管理上下文
    - 支持事务操作
    
    用法示例：
    def test_database_operation(test_db):
        user = User(username='test')
        test_db.session.add(user)
        test_db.session.commit()
    """
    with app.app_context():
        yield db

@pytest.fixture
def existing_patient(test_db):
    """
    测试患者数据Fixture
    
    功能：
    1. 创建测试患者
       - 使用工厂模式
       - 自动生成数据
       - 提交到数据库
    
    2. 数据特点
       - 真实模拟
       - 完整字段
       - 关联数据
    
    用法示例：
    def test_patient_detail(client, existing_patient):
        response = client.get(f'/api/patients/{existing_patient.id}')
        assert response.status_code == 200
    """
    # 使用工厂创建测试患者
    patient = PatientFactory()
    test_db.session.commit()
    return patient

@pytest.fixture
def upload_file():
    """
    文件上传测试Fixture
    
    功能：
    1. 创建测试文件
       - 支持多种文件类型
       - 可控的文件大小
       - 临时文件管理
    
    2. 文件操作
       - 读写测试
       - MIME类型设置
       - 文件清理
    
    用法示例：
    def test_file_upload(client, upload_file):
        response = client.post('/api/upload', 
                             data={'file': upload_file},
                             content_type='multipart/form-data')
        assert response.status_code == 200
    """
    content = b'test file content'
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(content)
        f.seek(0)
        yield f
    
    # 清理临时文件
    os.unlink(f.name)
