from flask import Flask, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from flask_mail import Mail
from config import Config
import os

db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()
mail = Mail()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # 初始化扩展
    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    mail.init_app(app)
    
    # 配置CORS
    CORS(app, resources={
        r"/api/*": {"origins": "*"},
        r"/uploads/*": {"origins": "*", "methods": ["GET"]}  # 允许访问上传的文件
    })
    
    # 确保上传文件夹存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # 添加静态文件服务路由
    @app.route('/uploads/<path:filename>')
    def serve_file(filename):
        """提供上传文件的访问服务"""
        try:
            app.logger.info(f"请求文件: {filename}")
            app.logger.info(f"上传文件夹: {app.config['UPLOAD_FOLDER']}")
            full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            app.logger.info(f"完整路径: {full_path}")
            app.logger.info(f"文件是否存在: {os.path.exists(full_path)}")
            
            if not os.path.exists(full_path):
                app.logger.error(f"文件不存在: {full_path}")
                return "File not found", 404
                
            response = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response
            
        except Exception as e:
            app.logger.error(f"服务文件时出错: {str(e)}")
            return str(e), 500
    
    # 注册蓝图
    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    
    from app.patient import bp as patient_bp
    app.register_blueprint(patient_bp, url_prefix='/api/patients')
    
    from app.mri import bp as mri_bp
    app.register_blueprint(mri_bp, url_prefix='/api/mri')
    
    from app.prediction import bp as prediction_bp
    app.register_blueprint(prediction_bp, url_prefix='/api/predictions')
    
    @app.route('/test')
    def test():
        return 'Hello, World!'
    
    return app 