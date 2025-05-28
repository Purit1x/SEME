"""
测试配置文件，包含共享的fixtures
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
    """测试配置类"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'  # 使用内存数据库
    WTF_CSRF_ENABLED = False
    SECRET_KEY = 'test-key'
    JWT_SECRET_KEY = 'test-jwt-key'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    UPLOAD_FOLDER = 'uploads'

@pytest.fixture
def app():
    """创建测试应用"""
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
        db.session.commit()
    
    yield app
    
    # 清理
    with app.app_context():
        db.session.remove()
        db.drop_all()
    os.rmdir(uploads_dir)

@pytest.fixture
def client(app):
    """测试客户端"""
    return app.test_client()

@pytest.fixture
def auth_headers(app):
    """认证头部"""
    from flask_jwt_extended import create_access_token
    with app.app_context():
        # 创建一个测试用的token，使用doctor_1作为identity
        access_token = create_access_token('doctor_1')
        return {
            'Authorization': f'Bearer {access_token}'
        }

@pytest.fixture
def test_db(app):
    """测试数据库会话"""
    # 使用已存在的应用上下文
    with app.app_context():
        yield db

@pytest.fixture
def existing_patient(test_db):
    """创建测试患者"""
    patient = PatientFactory()
    test_db.session.commit()
    return patient
