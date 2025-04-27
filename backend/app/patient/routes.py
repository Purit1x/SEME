import os
from flask import request, jsonify, current_app, url_for, send_from_directory
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from app.models import Patient
from app import db
from app.patient import bp
import re

def allowed_file(filename):
    """检查文件类型是否允许"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_id_number(id_number):
    """验证身份证号码格式"""
    # 身份证号码为18位
    if not re.match(r'^\d{17}[\dXx]$', id_number):
        return False
    return True

def validate_patient_info(data):
    """验证患者信息"""
    errors = []
    
    # 验证必填字段
    required_fields = ['patient_name', 'sex', 'age', 'id_number']
    for field in required_fields:
        if field not in data:
            errors.append(f'缺少必填字段: {field}')
    
    if errors:
        return False, errors
    
    # 验证姓名（2-50个字符）
    if not 2 <= len(data['patient_name']) <= 50:
        errors.append('姓名长度必须在2-50个字符之间')
    
    # 验证性别（男/女）
    if data['sex'] not in ['男', '女']:
        errors.append('性别必须是"男"或"女"')
    
    # 验证年龄（0-150岁）
    try:
        age = int(data['age'])
        if not 0 <= age <= 150:
            errors.append('年龄必须在0-150岁之间')
    except ValueError:
        errors.append('年龄必须是数字')
    
    # 验证身份证号
    if not validate_id_number(data['id_number']):
        errors.append('身份证号格式不正确')
    
    return len(errors) == 0, errors

def save_patient_photo(photo_file, patient_id):
    """保存患者照片"""
    # 创建患者照片目录，不包含 uploads 前缀
    photo_dir = os.path.join(f'patient_{patient_id}', 'photo')
    full_photo_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], photo_dir)
    os.makedirs(full_photo_dir, exist_ok=True)
    
    # 如果没有上传照片，使用默认照片
    if not photo_file:
        default_photo = os.path.join(current_app.config['UPLOAD_FOLDER'], 'default.png')
        if os.path.exists(default_photo):
            photo_path = os.path.join(photo_dir, 'photo.png')
            full_photo_path = os.path.join(current_app.config['UPLOAD_FOLDER'], photo_path)
            import shutil
            shutil.copy2(default_photo, full_photo_path)
            return photo_path  # 返回不包含 uploads 前缀的相对路径
        return None
        
    # 验证上传的照片格式
    if not allowed_file(photo_file.filename):
        raise ValueError('不支持的文件类型')
    
    # 保存上传的照片
    filename = secure_filename(photo_file.filename)
    base, ext = os.path.splitext(filename)
    photo_path = os.path.join(photo_dir, f'photo{ext}')
    full_photo_path = os.path.join(current_app.config['UPLOAD_FOLDER'], photo_path)
    photo_file.save(full_photo_path)
    
    return photo_path  # 返回不包含 uploads 前缀的相对路径

def get_file_url(path):
    """获取文件的完整URL"""
    if not path:
        return None
    return url_for('serve_file', filename=path, _external=True)

@bp.route('', methods=['POST'])
@jwt_required()
def create_patient():
    """导入患者信息"""
    try:
        # 验证用户身份
        current_user_id = get_jwt_identity()
        
        # 打印接收到的所有数据
        current_app.logger.info(f"Form data: {request.form}")
        current_app.logger.info(f"Files: {request.files}")
        
        # 获取表单数据并清理字段名中的空格
        data = {k.strip(): v.strip() for k, v in request.form.items()}
        current_app.logger.info(f"Cleaned data: {data}")
        
        # 验证患者信息
        is_valid, errors = validate_patient_info(data)
        if not is_valid:
            return jsonify({
                'success': False,
                'message': '输入信息有误',
                'errors': errors,
                'received_data': data  # 添加接收到的数据到响应中
            }), 400
        
        # 检查身份证号是否已存在
        existing_patient = Patient.query.filter_by(id_number=data['id_number']).first()
        if existing_patient:
            return jsonify({
                'success': False,
                'message': '该身份证号已存在'
            }), 400
        
        # 创建患者记录
        patient = Patient(
            patient_name=data['patient_name'],
            sex=data['sex'],
            age=int(data['age']),
            id_number=data['id_number']
        )
        db.session.add(patient)
        db.session.flush()  # 获取patient_id
        
        # 处理照片
        try:
            photo_file = request.files.get('photo')
            photo_path = save_patient_photo(photo_file, patient.patient_id)
            if photo_path:
                patient.photo_path = photo_path
        except ValueError as e:
            return jsonify({
                'success': False,
                'message': str(e)
            }), 400
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': '患者信息导入成功',
            'patient': {
                'id': patient.patient_id,
                'name': patient.patient_name,
                'sex': patient.sex,
                'age': patient.age,
                'id_number': patient.id_number,
                'photo_path': get_file_url(patient.photo_path)
            }
        })
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error in create_patient: {str(e)}")
        return jsonify({
            'success': False,
            'message': '导入失败，请稍后重试'
        }), 500

@bp.route('', methods=['GET'])
@jwt_required()
def list_patients():
    """获取患者列表"""
    try:
        # 获取分页参数
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        # 获取分页的患者列表
        pagination = Patient.query.order_by(Patient.patient_id.desc()).paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )
        
        patients = [{
            'id': patient.patient_id,
            'name': patient.patient_name,
            'sex': patient.sex,
            'age': patient.age,
            'id_number': patient.id_number,
            'photo_url': '/uploads/' + patient.photo_path.replace('\\', '/') if patient.photo_path else '/uploads/default.png'
        } for patient in pagination.items]
        
        return jsonify({
            'success': True,
            'patients': patients,
            'pagination': {
                'total': pagination.total,
                'pages': pagination.pages,
                'current_page': page,
                'per_page': per_page
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in list_patients: {str(e)}")
        return jsonify({
            'success': False,
            'message': '获取患者列表失败，请稍后重试'
        }), 500 