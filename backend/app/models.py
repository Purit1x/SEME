from datetime import datetime, timedelta
from app import db
from werkzeug.security import generate_password_hash, check_password_hash
import random

class Doctor(db.Model):
    __tablename__ = 'doctors'
    
    doctor_id = db.Column(db.String(64), primary_key=True)
    name = db.Column(db.String(64), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    department = db.Column(db.String(64), nullable=False)
    password_hash = db.Column(db.String(256))
    login_attempts = db.Column(db.Integer, default=0)  # 登录尝试次数
    locked_until = db.Column(db.DateTime)  # 账户锁定截止时间
    
    def set_password(self, password):
        """设置密码"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """验证密码"""
        return check_password_hash(self.password_hash, password)
    
    def is_locked(self):
        """检查账户是否被锁定"""
        if self.locked_until is None:
            return False
        return datetime.utcnow() < self.locked_until
    
    def increment_login_attempts(self):
        """增加登录尝试次数，如果超过限制则锁定账户"""
        self.login_attempts += 1
        if self.login_attempts >= 5:  # 5次失败后锁定账户
            self.locked_until = datetime.utcnow() + timedelta(minutes=30)  # 锁定30分钟
    
    def reset_login_attempts(self):
        """重置登录尝试次数"""
        self.login_attempts = 0
        self.locked_until = None

class Administrator(db.Model):
    __tablename__ = 'administrators'
    
    admin_id = db.Column(db.Integer, primary_key=True)
    password_hash = db.Column(db.String(256), nullable=False)
    
    def set_password(self, password):
        """设置密码"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """验证密码"""
        return check_password_hash(self.password_hash, password)

class MRISequence(db.Model):
    __tablename__ = 'mri_sequences'
    
    seq_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    seq_name = db.Column(db.String(255), nullable=False)  # 序列名称
    seq_dir = db.Column(db.String(255), nullable=False)   # 序列文件夹路径
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.patient_id'), nullable=False)  # 关联患者
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)  # 创建时间
    
    # 关联关系
    patient = db.relationship('Patient', backref=db.backref('sequences', lazy=True))
    items = db.relationship('MRISeqItem', backref='sequence', lazy=True)

class MRISeqItem(db.Model):
    __tablename__ = 'mri_seq_items'
    
    item_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    item_name = db.Column(db.String(255), nullable=False)  # 图像文件名
    file_path = db.Column(db.String(255), nullable=False)  # 图像文件路径
    seq_id = db.Column(db.Integer, db.ForeignKey('mri_sequences.seq_id'), nullable=False)
    uploaded_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)  # 上传时间
    item_type = db.Column(db.String(10), nullable=False)  # 'rgb' 或 'depth'

class Patient(db.Model):
    __tablename__ = 'patients'
    
    patient_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_name = db.Column(db.String(50), nullable=False)
    sex = db.Column(db.String(10), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    id_number = db.Column(db.String(18), unique=True, nullable=False)
    photo_path = db.Column(db.String(255))  # 患者照片路径

class PredRecord(db.Model):
    __tablename__ = 'pred_records'
    
    pred_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    pred_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    doctor_id = db.Column(db.String(64), db.ForeignKey('doctors.doctor_id'), nullable=False)
    item_id = db.Column(db.Integer, db.ForeignKey('mri_seq_items.item_id'), nullable=False)
    crop_x = db.Column(db.Integer, nullable=False)  # 裁切起始点x坐标
    crop_y = db.Column(db.Integer, nullable=False)  # 裁切起始点y坐标
    needle_positions = db.Column(db.JSON, nullable=False)  # 布针位置数组
    processed_rgb_path = db.Column(db.String(255), nullable=False)  # 处理后的RGB图像路径
    processed_depth_path = db.Column(db.String(255), nullable=False)  # 处理后的深度图像路径
    result_path = db.Column(db.String(255), nullable=False)  # 预测结果图像路径
    
    # 关联关系
    doctor = db.relationship('Doctor', backref=db.backref('predictions', lazy=True))
    mri_item = db.relationship('MRISeqItem', backref=db.backref('predictions', lazy=True))
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'pred_id': self.pred_id,
            'pred_time': self.pred_time.isoformat(),
            'doctor_id': self.doctor_id,
            'item_id': self.item_id,
            'crop_x': self.crop_x,
            'crop_y': self.crop_y,
            'needle_positions': self.needle_positions,
            'processed_rgb_path': self.processed_rgb_path,
            'processed_depth_path': self.processed_depth_path,
            'result_path': self.result_path
        }