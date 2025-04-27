from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import create_access_token, create_refresh_token, jwt_required, get_jwt_identity
from app.models import Doctor, Administrator
from app import db
from app.utils.email import send_verification_code
from datetime import datetime, timedelta
import re
import secrets
import random

bp = Blueprint('auth', __name__)

def validate_password(password):
    """验证密码强度"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number"
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character"
    return True, None

def generate_verification_code():
    """生成6位数字验证码"""
    return ''.join(str(random.randint(0, 9)) for _ in range(6))

# 存储验证码信息
verification_codes = {}

# 存储密码修改验证码信息
password_change_codes = {}

@bp.route('/doctor/register', methods=['POST'])
def doctor_register():
    """医生注册第一步：提交基本信息"""
    data = request.get_json()
    
    # 验证必要字段
    required_fields = ['doctor_id', 'name', 'password', 'email', 'department']
    if not all(k in data for k in required_fields):
        return jsonify({
            'success': False,
            'message': '缺少必要字段'
        }), 400
    
    # 验证工号是否已存在
    if Doctor.query.filter_by(doctor_id=data['doctor_id']).first():
        return jsonify({
            'success': False,
            'message': 'Doctor ID already exists'
        }), 400
    
    # 验证邮箱是否已存在
    if Doctor.query.filter_by(email=data['email']).first():
        return jsonify({
            'success': False,
            'message': 'Email already exists'
        }), 400
    
    # 验证邮箱格式
    if not re.match(r"[^@]+@[^@]+\.[^@]+", data['email']):
        return jsonify({
            'success': False,
            'message': 'Invalid email format'
        }), 400
    
    # 验证密码强度
    is_valid, error_msg = validate_password(data['password'])
    if not is_valid:
        return jsonify({
            'success': False,
            'message': error_msg
        }), 400
    
    # 生成验证码
    verification_id = secrets.token_urlsafe(32)
    verification_code = generate_verification_code()  # 使用随机生成的验证码
    verification_codes[verification_id] = {
        'code': verification_code,
        'doctor_data': data,
        'created_at': datetime.utcnow()
    }
    
    try:
        # 在测试环境中跳过实际的邮件发送
        if current_app.config.get('TESTING'):
            return jsonify({
                'success': True,
                'message': 'Verification code sent successfully',
                'verification_id': verification_id
            })
        
        # 发送验证码邮件
        if not send_verification_code(data['email'], verification_code):
            return jsonify({
                'success': False,
                'message': 'Failed to send verification code'
            }), 500
        
        return jsonify({
            'success': True,
            'message': 'Verification code sent successfully',
            'verification_id': verification_id
        })
    except Exception as e:
        current_app.logger.error(f"Error in doctor_register: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@bp.route('/doctor/verify', methods=['POST'])
def verify_doctor():
    """医生注册第二步：验证邮箱"""
    data = request.get_json()
    
    if not all(k in data for k in ['verification_id', 'code']):
        return jsonify({
            'success': False,
            'message': 'Missing required fields'
        }), 400
    
    verification_info = verification_codes.get(data['verification_id'])
    if not verification_info:
        return jsonify({
            'success': False,
            'message': 'Invalid verification ID'
        }), 400
    
    # 验证码过期检查（30分钟）
    if datetime.utcnow() - verification_info['created_at'] > timedelta(minutes=30):
        verification_codes.pop(data['verification_id'])
        return jsonify({
            'success': False,
            'message': 'Verification code expired'
        }), 400
    
    if verification_info['code'] != data['code']:
        return jsonify({
            'success': False,
            'message': 'Invalid verification code'
        }), 400
    
    try:
        # 创建新医生账户
        doctor_data = verification_info['doctor_data']
        doctor = Doctor(
            doctor_id=doctor_data['doctor_id'],
            name=doctor_data['name'],
            email=doctor_data['email'],
            department=doctor_data['department']
        )
        doctor.set_password(doctor_data['password'])
        
        db.session.add(doctor)
        db.session.commit()
        
        # 清理验证码信息
        verification_codes.pop(data['verification_id'])
        
        # 生成登录令牌
        access_token = create_access_token(identity=doctor.doctor_id)
        refresh_token = create_refresh_token(identity=doctor.doctor_id)
        
        return jsonify({
            'success': True,
            'message': 'Registration completed successfully',
            'access_token': access_token,
            'refresh_token': refresh_token
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@bp.route('/doctor/resend-code', methods=['POST'])
def resend_verification_code():
    """重新发送验证码"""
    data = request.get_json()
    
    if 'verification_id' not in data:
        return jsonify({
            'success': False,
            'message': 'Missing verification ID'
        }), 400
    
    verification_info = verification_codes.get(data['verification_id'])
    if not verification_info:
        return jsonify({
            'success': False,
            'message': 'Invalid verification ID'
        }), 400
    
    # 在测试环境中跳过等待时间查
    if not current_app.config.get('TESTING'):
        # 检查是否在60秒内重发
        if datetime.utcnow() - verification_info['created_at'] < timedelta(seconds=60):
            return jsonify({
                'success': False,
                'message': 'Please wait before requesting a new code'
            }), 400
    
    try:
        # 生成新验证码
        new_code = generate_verification_code()  # 使用随机生成的验证码
        verification_info['code'] = new_code
        verification_info['created_at'] = datetime.utcnow()
        
        # 在测试环境中跳过实际的邮件发送
        if current_app.config.get('TESTING'):
            return jsonify({
                'success': True,
                'message': 'Verification code resent successfully'
            })
        
        # 发送新验证码
        if not send_verification_code(verification_info['doctor_data']['email'], new_code):
            return jsonify({
                'success': False,
                'message': 'Failed to send verification code'
            }), 500
        
        return jsonify({
            'success': True,
            'message': 'Verification code resent successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@bp.route('/doctor/login', methods=['POST'])
def doctor_login():
    """医生登录接口"""
    data = request.get_json()
    
    # 验证必要字段
    if not all(k in data for k in ['login_id', 'password']):
        return jsonify({
            'success': False,
            'message': '缺少必要字段'
        }), 400
    
    # 通过邮箱或工号查找医生
    doctor = Doctor.query.filter(
        (Doctor.email == data['login_id']) | 
        (Doctor.doctor_id == data['login_id'])
    ).first()
    
    if not doctor:
        return jsonify({
            'success': False,
            'message': '用户不存在'
        }), 401
    
    # 检查账户是否被锁定
    if doctor.is_locked():
        return jsonify({
            'success': False,
            'message': f'账户已被锁定，请在 {doctor.locked_until} 后重试'
        }), 401
    
    # 验证密码
    if not doctor.check_password(data['password']):
        doctor.increment_login_attempts()
        db.session.commit()
        
        if doctor.is_locked():
            return jsonify({
                'success': False,
                'message': '登录失败次数过多，账户已被锁定30分钟'
            }), 401
        
        remaining_attempts = 5 - doctor.login_attempts
        return jsonify({
            'success': False,
            'message': f'密码错误，还剩 {remaining_attempts} 次尝试机会'
        }), 401
    
    # 登录成功，重置登录尝试次数
    doctor.reset_login_attempts()
    db.session.commit()
    
    # 生成访问令牌
    access_token = create_access_token(identity=doctor.doctor_id)
    refresh_token = create_refresh_token(identity=doctor.doctor_id)
    
    return jsonify({
        'success': True,
        'access_token': access_token,
        'refresh_token': refresh_token,
        'doctor': {
            'id': doctor.doctor_id,
            'name': doctor.name,
            'department': doctor.department,
            'email': doctor.email
        }
    })

@bp.route('/admin/login', methods=['POST'])
def admin_login():
    """管理员登录接口"""
    data = request.get_json()
    
    if not all(k in data for k in ['admin_id', 'password']):
        return jsonify({
            'success': False,
            'message': '缺少必要字段'
        }), 400
    
    admin = Administrator.query.get(data['admin_id'])
    
    if not admin:
        return jsonify({
            'success': False,
            'message': '管理员ID不存在'
        }), 401
    
    if not admin.check_password(data['password']):
        return jsonify({
            'success': False,
            'message': '密码错误'
        }), 401
    
    # 生成访问令牌
    access_token = create_access_token(identity=f"admin_{admin.admin_id}")
    refresh_token = create_refresh_token(identity=f"admin_{admin.admin_id}")
    
    return jsonify({
        'success': True,
        'access_token': access_token,
        'refresh_token': refresh_token,
        'admin': {
            'id': admin.admin_id
        }
    })

@bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    """刷新访问令牌"""
    current_user = get_jwt_identity()
    access_token = create_access_token(identity=current_user)
    return jsonify({
        'success': True,
        'access_token': access_token
    })

@bp.route('/doctor/change-password/send-code', methods=['POST'])
@jwt_required()
def send_password_change_code():
    """发送密码修改验证码"""
    current_user_id = get_jwt_identity()
    doctor = Doctor.query.get(current_user_id)
    
    if not doctor:
        return jsonify({
            'success': False,
            'message': '用户不存在'
        }), 404
    
    # 检查是否在1分钟内重复发送
    verification_info = password_change_codes.get(current_user_id)
    if verification_info and not current_app.config.get('TESTING'):
        time_diff = datetime.utcnow() - verification_info['created_at']
        if time_diff < timedelta(minutes=1):
            return jsonify({
                'success': False,
                'message': '请等待1分钟后再重新发送验证码'
            }), 400
    
    # 生成新的验证码
    verification_code = generate_verification_code()
    password_change_codes[current_user_id] = {
        'code': verification_code,
        'created_at': datetime.utcnow()
    }
    
    try:
        # 在测试环境中跳过实际的邮件发送
        if current_app.config.get('TESTING'):
            return jsonify({
                'success': True,
                'message': '验证码发送成功'
            })
        
        # 发送验证码邮件
        if not send_verification_code(doctor.email, verification_code):
            return jsonify({
                'success': False,
                'message': '验证码发送失败'
            }), 500
        
        return jsonify({
            'success': True,
            'message': '验证码发送成功'
        })
    except Exception as e:
        current_app.logger.error(f"Error in send_password_change_code: {str(e)}")
        return jsonify({
            'success': False,
            'message': '验证码发送失败'
        }), 500

@bp.route('/doctor/change-password', methods=['POST'])
@jwt_required()
def change_password():
    """修改密码"""
    current_user_id = get_jwt_identity()
    doctor = Doctor.query.get(current_user_id)
    
    if not doctor:
        return jsonify({
            'success': False,
            'message': '用户不存在'
        }), 404
    
    data = request.get_json()
    required_fields = ['current_password', 'new_password', 'confirm_password', 'verification_code']
    if not all(k in data for k in required_fields):
        return jsonify({
            'success': False,
            'message': '缺少必要字段'
        }), 400
    
    # 验证当前密码
    if not doctor.check_password(data['current_password']):
        return jsonify({
            'success': False,
            'message': '当前密码错误'
        }), 400
    
    # 验证新密码一致性
    if data['new_password'] != data['confirm_password']:
        return jsonify({
            'success': False,
            'message': '两次输入的新密码不一致'
        }), 400
    
    # 验证新密码强度
    is_valid, error_msg = validate_password(data['new_password'])
    if not is_valid:
        return jsonify({
            'success': False,
            'message': error_msg
        }), 400
    
    # 验证验证码
    verification_info = password_change_codes.get(current_user_id)
    if not verification_info:
        return jsonify({
            'success': False,
            'message': '请先获取验证码'
        }), 400
    
    # 验证码过期检查（10分钟）
    if datetime.utcnow() - verification_info['created_at'] > timedelta(minutes=10):
        password_change_codes.pop(current_user_id)
        return jsonify({
            'success': False,
            'message': '验证码已过期，请重新获取'
        }), 400
    
    if verification_info['code'] != data['verification_code']:
        return jsonify({
            'success': False,
            'message': '验证码错误'
        }), 400
    
    try:
        # 更新密码
        doctor.set_password(data['new_password'])
        db.session.commit()
        
        # 清理验证码
        password_change_codes.pop(current_user_id)
        
        return jsonify({
            'success': True,
            'message': '密码修改成功'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': '密码修改失败'
        }), 500

@bp.route('/doctors', methods=['GET'])
@jwt_required()
def get_doctors():
    """获取医生列表"""
    try:
        # 获取所有医生信息
        doctors = Doctor.query.all()
        
        # 构造返回数据
        doctors_data = []
        for doctor in doctors:
            doctor_info = {
                'doctor_id': doctor.doctor_id,
                'name': doctor.name,
                'email': doctor.email,
                'department': doctor.department,
                'locked_until': doctor.locked_until.isoformat() if doctor.locked_until else None
            }
            doctors_data.append(doctor_info)
        
        return jsonify({
            'success': True,
            'doctors': doctors_data
        })
        
    except Exception as e:
        current_app.logger.error(f"获取医生列表失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': '获取医生列表失败'
        }), 500

@bp.route('/doctor/<doctor_id>/ban', methods=['POST'])
@jwt_required()
def ban_doctor(doctor_id):
    """封禁医生账号"""
    try:
        # 验证当前用户是否为管理员
        current_user = get_jwt_identity()
        if not current_user.startswith('admin_'):
            return jsonify({
                'success': False,
                'message': '无权限执行此操作'
            }), 403
        
        # 查找医生
        doctor = Doctor.query.get(doctor_id)
        if not doctor:
            return jsonify({
                'success': False,
                'message': '医生不存在'
            }), 404
        
        # 设置封禁时间为180天后
        doctor.locked_until = datetime.utcnow() + timedelta(days=180)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': '医生账号已被封禁',
            'locked_until': doctor.locked_until.isoformat()
        })
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"封禁医生失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': '封禁医生失败'
        }), 500

@bp.route('/doctor/<doctor_id>/unban', methods=['POST'])
@jwt_required()
def unban_doctor(doctor_id):
    """解除医生账号封禁"""
    try:
        # 验证当前用户是否为管理员
        current_user = get_jwt_identity()
        if not current_user.startswith('admin_'):
            return jsonify({
                'success': False,
                'message': '无权限执行此操作'
            }), 403
        
        # 查找医生
        doctor = Doctor.query.get(doctor_id)
        if not doctor:
            return jsonify({
                'success': False,
                'message': '医生不存在'
            }), 404
        
        # 检查是否被封禁
        if not doctor.is_locked():
            return jsonify({
                'success': False,
                'message': '该医生账号未被封禁'
            }), 400
        
        # 解除封禁
        doctor.locked_until = None
        doctor.login_attempts = 0
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': '已解除医生账号封禁'
        })
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"解除医生封禁失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': '解除封禁失败'
        }), 500