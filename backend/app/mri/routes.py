import os
from flask import request, jsonify, current_app, url_for, send_from_directory, send_file
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from app.models import Patient, MRISequence, MRISeqItem, Doctor, Administrator
from app import db
from datetime import datetime
from app.mri import bp
import json
from ..auth.auth_utils import token_required

def allowed_file(filename, file_type='rgb'):
    """检查文件类型是否允许
    
    Args:
        filename: 文件名
        file_type: 文件类型，'rgb' 或 'depth'
    """
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_sequence_directory(patient_id, seq_name):
    """创建序列文件夹，包括RGB和深度图像子文件夹"""
    # 构建相对路径（不包含uploads前缀）
    patient_dir = os.path.join(f'patient_{patient_id}', 'sequences')
    seq_dir = os.path.join(patient_dir, secure_filename(seq_name))
    rgb_dir = os.path.join(seq_dir, 'rgb')
    depth_dir = os.path.join(seq_dir, 'depth')
    
    # 获取基础目录
    base_dir = os.path.join(current_app.root_path, '..', 'uploads')
    
    # 确保目录存在
    os.makedirs(os.path.join(base_dir, rgb_dir), exist_ok=True)
    os.makedirs(os.path.join(base_dir, depth_dir), exist_ok=True)
    
    # 返回相对路径
    return seq_dir, rgb_dir, depth_dir

def get_user_type(user_id):
    """获取用户类型"""
    if user_id.startswith('admin_'):
        return 'admin', int(user_id.replace('admin_', ''))
    return 'doctor', user_id

def get_file_url(path):
    """获取文件的完整URL"""
    if not path:
        return None
    # 确保路径使用正斜杠
    normalized_path = path.replace('\\', '/')
    # 直接返回完整URL，不需要添加 uploads 前缀，因为 serve_file 会处理
    return url_for('serve_file', filename=normalized_path, _external=True)

@bp.route('/patients/<int:patient_id>/sequences', methods=['POST'])
@jwt_required()
def create_sequence(patient_id):
    """创建新的MRI序列，包括RGB和深度图像"""
    base_dir = None 
    seq_dir = None
    
    try:
        # 验证用户身份
        current_user_id = get_jwt_identity()
        user_type, user_id = get_user_type(current_user_id)
        
        # 检查患者是否存在
        patient = Patient.query.get(patient_id)
        if not patient:
            return jsonify({
                'success': False,
                'message': '患者不存在',
                'should_create_patient': True,
                'redirect_url': url_for('patient.create_patient')
            }), 404
        
        # 获取序列名称
        seq_name = request.form.get('seq_name')
        if not seq_name:
            return jsonify({
                'success': False,
                'message': '缺少序列名称'
            }), 400
        
        # 检查序列名称是否已存在
        existing_sequence = MRISequence.query.filter_by(
            patient_id=patient_id,
            seq_name=seq_name
        ).first()
        
        if existing_sequence:
            return jsonify({
                'success': False,
                'message': '序列名称已存在，请修改序列名称'
            }), 400
        
        # 检查是否上传了RGB和深度图像文件
        if 'rgb_files[]' not in request.files or 'depth_files[]' not in request.files:
            return jsonify({
                'success': False,
                'message': '请同时上传RGB图像和深度图像'
            }), 400
        
        rgb_files = request.files.getlist('rgb_files[]')
        depth_files = request.files.getlist('depth_files[]')
        
        # 检查文件数量是否匹配
        if len(rgb_files) != len(depth_files):
            return jsonify({
                'success': False,
                'message': 'RGB图像和深度图像数量不匹配'
            }), 400
        
        if not rgb_files or not any(file.filename for file in rgb_files):
            return jsonify({
                'success': False,
                'message': '未选择RGB图像文件'
            }), 400
            
        if not depth_files or not any(file.filename for file in depth_files):
            return jsonify({
                'success': False,
                'message': '未选择深度图像文件'
            }), 400
        
        # 验证所有文件格式
        invalid_rgb_files = [f.filename for f in rgb_files if f.filename and not allowed_file(f.filename, 'rgb')]
        invalid_depth_files = [f.filename for f in depth_files if f.filename and not allowed_file(f.filename, 'depth')]
        
        if invalid_rgb_files or invalid_depth_files:
            return jsonify({
                'success': False,
                'message': '存在不支持的文件类型',
                'invalid_rgb_files': invalid_rgb_files,
                'invalid_depth_files': invalid_depth_files
            }), 400

        # 创建序列目录
        seq_dir, rgb_dir, depth_dir = create_sequence_directory(patient_id, seq_name)
        base_dir = os.path.join(current_app.root_path, '..', 'uploads')

        uploaded_files = []
        saved_files = []  # 跟踪已保存的文件

        # 创建序列记录
        sequence = MRISequence(
            seq_name=seq_name,
            seq_dir=seq_dir,
            patient_id=patient_id,
            created_at=datetime.utcnow()
        )
        db.session.add(sequence)
        db.session.flush()  # 确保我们有 sequence.seq_id
            
        # 保存所有文件和创建数据库记录
        for i, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
            if rgb_file and rgb_file.filename and depth_file and depth_file.filename:
                try:
                    # 处理RGB图像
                    rgb_filename = secure_filename(rgb_file.filename)
                    rgb_base, rgb_ext = os.path.splitext(rgb_filename)
                    rgb_path = os.path.join(rgb_dir, f"image_{i+1}{rgb_ext}")
                    rgb_full_path = os.path.join(base_dir, rgb_path)
                    
                    # 处理深度图像
                    depth_filename = secure_filename(depth_file.filename)
                    depth_base, depth_ext = os.path.splitext(depth_filename)
                    depth_path = os.path.join(depth_dir, f"image_{i+1}{depth_ext}")
                    depth_full_path = os.path.join(base_dir, depth_path)
                    
                    # 保存文件
                    rgb_file.save(rgb_full_path)
                    saved_files.append(rgb_full_path)
                    depth_file.save(depth_full_path)
                    saved_files.append(depth_full_path)

                    # 创建RGB图像记录
                    rgb_item = MRISeqItem(
                        item_name=f"image_{i+1}{rgb_ext}",
                        file_path=rgb_path,
                        seq_id=sequence.seq_id,
                        uploaded_at=datetime.utcnow(),
                        item_type='rgb'
                    )
                    db.session.add(rgb_item)
                    
                    # 创建深度图像记录
                    depth_item = MRISeqItem(
                        item_name=f"image_{i+1}{depth_ext}",
                        file_path=depth_path,
                        seq_id=sequence.seq_id,
                        uploaded_at=datetime.utcnow(),
                        item_type='depth'
                    )
                    db.session.add(depth_item)
                    
                    uploaded_files.append({
                        'index': i + 1,
                        'rgb': {
                            'name': f"image_{i+1}{rgb_ext}",
                            'path': get_file_url(rgb_path)
                        },
                        'depth': {
                            'name': f"image_{i+1}{depth_ext}",
                            'path': get_file_url(depth_path)
                        }
                    })

                except Exception as e:
                    current_app.logger.error(f"Error saving files: {str(e)}")
                    # 删除已保存的文件
                    for saved_file in saved_files:
                        try:
                            if os.path.exists(saved_file):
                                os.remove(saved_file)
                        except Exception as fe:
                            current_app.logger.error(f"Error cleaning up file {saved_file}: {str(fe)}")
                    # 确保回滚数据库更改
                    db.session.rollback()
                    return jsonify({
                        'success': False,
                        'message': '文件上传失败，请检查网络连接后重试'
                    }), 500

        # 如果一切正常，提交数据库更改
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': '序列创建成功',
            'sequence': {
                'id': sequence.seq_id,
                'name': sequence.seq_name,
                'patient_id': patient_id,
                'patient_name': patient.patient_name,
                'files': uploaded_files,
                'created_at': sequence.created_at.isoformat()
            }
        })
            
    except Exception as e:
        current_app.logger.error(f"Error in create_sequence: {str(e)}")
        # 确保回滚任何数据库更改
        db.session.rollback()
        
        # 删除任何已保存的文件
        if base_dir and seq_dir:
            try:
                full_seq_dir = os.path.join(base_dir, seq_dir)
                if os.path.exists(full_seq_dir):
                    import shutil
                    shutil.rmtree(full_seq_dir)
            except Exception as e:
                current_app.logger.error(f"Error cleaning up sequence directory: {str(e)}")
        
        return jsonify({
            'success': False,
            'message': '序列创建失败，请稍后重试'
        }), 500

@bp.route('/patients/<int:patient_id>/sequences/<int:seq_id>', methods=['GET'])
@jwt_required()
def get_sequence(patient_id, seq_id):
    """获取序列详情"""
    sequence = MRISequence.query.filter_by(
        seq_id=seq_id,
        patient_id=patient_id
    ).first()
    
    if not sequence:
        return jsonify({
            'success': False,
            'message': '序列不存在'
        }), 404
    
    # 分别获取RGB和深度图像
    rgb_items = [item for item in sequence.items if item.item_type == 'rgb']
    depth_items = [item for item in sequence.items if item.item_type == 'depth']
    
    # 组织配对的图像数据
    paired_items = []
    for i in range(len(rgb_items)):
        paired_items.append({
            'index': i + 1,
            'rgb': {
                'id': rgb_items[i].item_id,
                'name': rgb_items[i].item_name,
                'path': get_file_url(rgb_items[i].file_path),
                'uploaded_at': rgb_items[i].uploaded_at.isoformat()
            },
            'depth': {
                'id': depth_items[i].item_id,
                'name': depth_items[i].item_name,
                'path': get_file_url(depth_items[i].file_path),
                'uploaded_at': depth_items[i].uploaded_at.isoformat()
            }
        })
    
    return jsonify({
        'success': True,
        'sequence': {
            'id': sequence.seq_id,
            'name': sequence.seq_name,
            'patient_id': patient_id,
            'patient_name': sequence.patient.patient_name,
            'created_at': sequence.created_at.isoformat(),
            'items': paired_items
        }
    })

@bp.route('/patients/<int:patient_id>/sequences', methods=['GET'])
@jwt_required()
def list_sequences(patient_id):
    """获取患者的所有序列"""
    patient = Patient.query.get(patient_id)
    if not patient:
        return jsonify({
            'success': False,
            'message': '患者不存在'
        }), 404
    
    sequences = MRISequence.query.filter_by(patient_id=patient_id).all()
    
    return jsonify({
        'success': True,
        'patient': {
            'id': patient.patient_id,
            'name': patient.patient_name
        },
        'sequences': [{
            'id': seq.seq_id,
            'name': seq.seq_name,
            'created_at': seq.created_at.isoformat(),
            'rgb_count': len([item for item in seq.items if item.item_type == 'rgb']),
            'depth_count': len([item for item in seq.items if item.item_type == 'depth'])
        } for seq in sequences]
    })

@bp.route('/patients/<int:patient_id>/sequences/<int:seq_id>/images/<int:image_index>', methods=['GET'])
@jwt_required()
def get_sequence_image_db(patient_id, seq_id, image_index):
    """从数据库获取序列中的图像"""
    try:
        # 检查序列是否存在
        sequence = MRISequence.query.filter_by(
            seq_id=seq_id,
            patient_id=patient_id
        ).first_or_404()
        
        # 获取指定索引的RGB和深度图像
        rgb_items = [item for item in sequence.items if item.item_type == 'rgb']
        depth_items = [item for item in sequence.items if item.item_type == 'depth']
        
        # 检查索引是否有效
        if not (0 <= image_index < len(rgb_items)):
            return jsonify({
                'success': False,
                'message': '图像索引超出范围'
            }), 404
        
        # 获取请求的图像类型
        image_type = request.args.get('type', 'rgb')  # 默认返回RGB图像
        if image_type not in ['rgb', 'depth']:
            return jsonify({
                'success': False,
                'message': '不支持的图像类型'
            }), 400
        
        # 根据类型选择图像
        image_item = rgb_items[image_index] if image_type == 'rgb' else depth_items[image_index]
        
        # 获取完整的文件路径
        file_path = os.path.join('uploads', image_item.file_path)
        full_path = os.path.join(current_app.root_path, '..', file_path)
        
        # 检查文件是否存在
        if not os.path.exists(full_path):
            current_app.logger.error(f"文件不存在: {full_path}")
            return jsonify({
                'success': False,
                'message': '图像文件不存在'
            }), 404
        
        # 从相对路径中获取文件名和目录
        directory = os.path.dirname(full_path)
        filename = os.path.basename(full_path)
        
        return send_from_directory(directory, filename)
        
    except FileNotFoundError:
        return jsonify({
            'success': False,
            'message': '图像文件不存在'
        }), 404
    except Exception as e:
        current_app.logger.error(f"Error in get_sequence_image: {str(e)}")
        return jsonify({
            'success': False,
            'message': '获取图像失败，请稍后重试'
        }), 500

@bp.route('/needle', methods=['POST'])
@jwt_required()
def save_needled_image():
    try:
        current_app.logger.info('开始处理布针图片保存请求')
        # 获取表单数据
        image = request.files.get('image')
        patient_id = request.form.get('patient_id')
        sequence_id = request.form.get('sequence_id')
        image_type = request.form.get('image_type')
        needle_points = request.form.get('needle_points')

        current_app.logger.info(f'接收到的参数: patient_id={patient_id}, sequence_id={sequence_id}, image_type={image_type}')
        current_app.logger.info(f'针点数据: {needle_points}')

        if not all([image, patient_id, sequence_id, image_type, needle_points]):
            missing = []
            if not image: missing.append('image')
            if not patient_id: missing.append('patient_id')
            if not sequence_id: missing.append('sequence_id')
            if not image_type: missing.append('image_type')
            if not needle_points: missing.append('needle_points')
            current_app.logger.error(f'缺少必要参数: {", ".join(missing)}')
            return jsonify({
                'success': False,
                'message': f'缺少必要的参数: {", ".join(missing)}'
            }), 400

        # 获取序列信息
        sequence = MRISequence.query.get(sequence_id)
        if not sequence:
            return jsonify({
                'success': False,
                'message': '序列不存在'
            }), 404
        
        # 使用序列的实际路径
        base_path = os.path.join(
            current_app.root_path,
            '..',
            'uploads',
            sequence.seq_dir
        )
        
        needled_path = os.path.join(base_path, 'needled')
        current_app.logger.info(f'创建布针图片保存路径: {needled_path}')
        
        # 确保目录存在
        os.makedirs(needled_path, exist_ok=True)

        # 获取当前最大的文件编号
        existing_files = [f for f in os.listdir(needled_path) if f.startswith('needled_') and f.endswith('.png')]
        current_max = 0
        for f in existing_files:
            try:
                num = int(f.replace('needled_', '').replace('.png', ''))
                current_max = max(current_max, num)
            except ValueError:
                continue
        
        next_num = current_max + 1
        
        # 保存图片
        image_filename = secure_filename(f'needled_{next_num}.png')
        image_path = os.path.join(needled_path, image_filename)
        current_app.logger.info(f'保存图片到: {image_path}')
        image.save(image_path)

        # 保存布针点信息
        points_filename = f'needle_points_{next_num}.json'
        points_path = os.path.join(needled_path, points_filename)
        current_app.logger.info(f'保存针点数据到: {points_path}')
        with open(points_path, 'w') as f:
            f.write(needle_points)

        # 返回相对路径
        relative_image_path = os.path.join(sequence.seq_dir, 'needled', image_filename)
        relative_points_path = os.path.join(sequence.seq_dir, 'needled', points_filename)

        return jsonify({
            'success': True,
            'message': '布针图片保存成功',
            'data': {
                'image_path': relative_image_path,
                'points_path': relative_points_path,
                'number': next_num
            }
        })

    except Exception as e:
        current_app.logger.error(f'保存布针图片失败: {str(e)}')
        current_app.logger.error(f'错误详情: {e.__class__.__name__}')
        import traceback
        current_app.logger.error(f'堆栈跟踪: {traceback.format_exc()}')
        return jsonify({
            'success': False,
            'message': f'保存布针图片失败: {str(e)}'
        }), 500

@bp.route('/patients/<patient_id>/sequences/<seq_id>/images/<int:image_index>', methods=['GET'])
@jwt_required()
def get_sequence_image_file(patient_id, seq_id, image_index):
    """从文件系统获取序列中的图像"""
    try:
        # 获取图像类型（RGB或深度）
        image_type = request.args.get('type', 'rgb')
        type_folder = 'depth' if image_type == 'depth' else 'rgb'

        # 构建图片路径
        image_path = os.path.join(
            current_app.root_path,
            '..',
            'uploads',
            f'patient_{patient_id}',
            'sequences',
            str(seq_id),
            type_folder,
            f'image_{image_index + 1}.png'
        )

        if not os.path.exists(image_path):
            return jsonify({
                'success': False,
                'message': '图片不存在'
            }), 404

        return send_file(image_path, mimetype='image/png')

    except Exception as e:
        current_app.logger.error(f'获取序列图片失败: {str(e)}')
        return jsonify({
            'success': False,
            'message': f'获取序列图片失败: {str(e)}'
        }), 500

@bp.route('/patients/<patient_id>/sequences', methods=['GET'])
@jwt_required()
def get_patient_sequences(patient_id):
    try:
        # 构建序列目录路径
        sequences_path = os.path.join(
            current_app.config['UPLOAD_FOLDER'],
            f'patient_{patient_id}',
            'sequences'
        )

        if not os.path.exists(sequences_path):
            return jsonify({
                'success': True,
                'sequences': []
            })

        # 获取所有序列目录
        sequences = []
        for seq_id in os.listdir(sequences_path):
            seq_path = os.path.join(sequences_path, seq_id)
            if os.path.isdir(seq_path):
                # 计算RGB和深度图像的数量
                rgb_path = os.path.join(seq_path, 'rgb')
                depth_path = os.path.join(seq_path, 'depth')
                
                rgb_count = len([f for f in os.listdir(rgb_path) if f.endswith('.png')]) if os.path.exists(rgb_path) else 0
                depth_count = len([f for f in os.listdir(depth_path) if f.endswith('.png')]) if os.path.exists(depth_path) else 0

                sequences.append({
                    'id': seq_id,
                    'name': f'序列{seq_id}',
                    'rgb_count': rgb_count,
                    'depth_count': depth_count,
                    'created_at': '2024-12-28T09:48:10'  # 这里可以从文件创建时间获取
                })

        return jsonify({
            'success': True,
            'patient': {
                'id': patient_id,
                'name': f'患者{patient_id}'  # 这里可以从数据库获取患者名字
            },
            'sequences': sequences
        })

    except Exception as e:
        current_app.logger.error(f'获取患者序列失败: {str(e)}')
        return jsonify({
            'success': False,
            'message': f'获取患者序列失败: {str(e)}'
        }), 500