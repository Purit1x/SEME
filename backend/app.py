from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from config import Config
from app.prediction.pred_utils.predict import load_model
import os

# 初始化扩展
db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # 初始化扩展
    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    
    # 配置CORS
    CORS(app, resources={
        r"/api/*": {"origins": "*"},
        r"/uploads/*": {"origins": "*"}  # 允许访问上传的文件
    })
    
    # 确保上传文件夹存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # 注册蓝图
    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    
    from app.patient import bp as patient_bp
    app.register_blueprint(patient_bp, url_prefix='/api/patients')
    
    from app.mri import bp as mri_bp
    app.register_blueprint(mri_bp, url_prefix='/api/mri')
    
    from app.prediction import bp as prediction_bp
    app.register_blueprint(prediction_bp, url_prefix='/api/predictions')
    
    # 加载模型
    load_model()
    
    return app