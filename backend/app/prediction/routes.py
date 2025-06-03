"""
MRI图像预测模块 (MRI Image Prediction Module)

本模块提供了MRI图像预测的核心功能，包括：
1. 图像预处理
   - RGB图像裁剪
   - 深度图像处理
   - 针灸点位置标记
2. 预测流程管理
   - 创建预测任务
   - 执行预测
   - 保存预测结果
3. 预测记录管理
   - 创建预测记录
   - 查询预测历史

技术栈：
- PIL: 图像处理
- NumPy: 数学计算
- OpenCV: 图像处理
- Flask: Web框架
- SQLAlchemy: 数据库ORM

文件结构：
- gaussian(): 高斯函数实现
- process_depth_image(): 深度图像处理
- crop_image(): 图像裁剪
- save_image(): 图像保存
- create_prediction(): 创建预测API
- get_sequence_predictions(): 获取序列预测记录API

作者: Your Name
创建日期: 2025-06-16
"""

import os
import numpy as np
from PIL import Image
from flask import Blueprint, request, jsonify, current_app, send_file
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.models import MRISequence, PredRecord, MRISeqItem, Doctor
from app import db
from app.prediction import bp
from app.prediction.pred_utils.predict import predict, IMG_SIZE
from app.auth.auth_utils import token_required
import cv2
import json
from datetime import datetime

def gaussian(x, mu, sigma):
    """
    计算高斯函数值
    
    这个函数用于生成针灸点周围的强度衰减效果，创建平滑的过渡区域。
    
    参数:
        x (float|ndarray): 输入值或数组
        mu (float): 均值（中心位置）
        sigma (float): 标准差（决定扩散范围）
    
    返回:
        float|ndarray: 高斯函数计算结果
        
    公式:
        f(x) = exp(-(x - μ)² / (2σ²))
    
    示例:
        >>> gaussian(0, 0, 1)
        1.0
        >>> gaussian(1, 0, 1)
        0.60653065971263342
    """
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def process_depth_image(image_path, needle_positions, radius=9, sigma=4):
    """
    处理深度图像，在针灸点位置创建高斯分布的白色区域
    
    参数:
        image_path (str): 深度图像的文件路径
        needle_positions (list[dict]): 针灸点位置列表
            每个字典包含 'x' 和 'y' 坐标
        radius (int): 影响区域的半径，默认为9像素
        sigma (float): 高斯分布的标准差，控制扩散程度，默认为4
    
    返回:
        PIL.Image: 处理后的图像对象
    
    处理步骤:
        1. 加载深度图像并转换为灰度图
        2. 对每个针灸点:
           - 计算影响区域的边界
           - 创建高斯衰减的白色区域
           - 将高斯mask应用到原图上
        3. 将处理后的数组转换回PIL图像
    
    设计说明：
        - 使用高斯函数创建平滑的过渡区域
        - 考虑图像边界情况
        - 保持原始深度值和新添加的白色区域的最大值
    """
    # 打开图像
    image = Image.open(image_path).convert('L')
    # 转换为numpy数组以便处理
    img_array = np.array(image)
    
    # 对每个布针位置创建白色区域
    for pos in needle_positions:
        center_x, center_y = pos['x'], pos['y']
        
        # 在布针位置周围创建白色区域
        y_indices, x_indices = np.ogrid[-radius:radius+1, -radius:radius+1]
        distances = np.sqrt(x_indices**2 + y_indices**2)
        
        # 创建高斯衰减的白色区域
        mask = gaussian(distances, 0, sigma)
        
        # 确定区域边界
        y_start = max(0, center_y - radius)
        y_end = min(img_array.shape[0], center_y + radius + 1)
        x_start = max(0, center_x - radius)
        x_end = min(img_array.shape[1], center_x + radius + 1)
        
        # 应用高斯衰减
        region = img_array[y_start:y_end, x_start:x_end]
        mask_region = mask[:y_end-y_start, :x_end-x_start]
        img_array[y_start:y_end, x_start:x_end] = np.maximum(
            region,
            255 * mask_region
        )
    
    # 转换回PIL图像
    return Image.fromarray(img_array)

def crop_image(image_path, x, y, size):
    """
    从原始图像中裁剪指定大小的区域
    
    参数:
        image_path (str): 源图像的文件路径
        x (int): 裁剪区域左上角的X坐标
        y (int): 裁剪区域左上角的Y坐标
        size (int): 裁剪区域的边长（正方形）
    
    返回:
        PIL.Image: 裁剪后的图像对象
    
    处理流程:
        1. 打开源图像
        2. 验证并调整裁剪坐标
           - 确保不会超出图像边界
           - 自动调整到有效范围
        3. 执行裁剪操作
    
    异常处理:
        - 自动处理超出边界的情况
        - 使用with语句确保文件正确关闭
    """
    with Image.open(image_path) as img:
        # 确保坐标不会超出图像范围
        width, height = img.size
        x = max(0, min(x, width - size))
        y = max(0, min(y, height - size))
        
        # 裁切图像
        cropped = img.crop((x, y, x + size, y + size))
        return cropped

def save_image(image, directory, filename):
    """
    将PIL图像对象保存到指定目录
    
    参数:
        image (PIL.Image): 要保存的图像对象
        directory (str): 保存目录的路径
        filename (str): 文件名
    
    返回:
        str: 保存文件的相对路径
    
    功能:
        1. 自动创建目录（如果不存在）
        2. 生成完整的文件路径
        3. 保存图像
        4. 返回相对路径（用于数据库存储）
    
    说明:
        使用os.makedirs确保目录存在，使用os.path.relpath
        生成相对路径以便数据库存储和URL生成
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    image.save(filepath)
    return os.path.relpath(filepath)

@bp.route('/predict', methods=['POST'])
@jwt_required()
def create_prediction():
    """
    创建新的预测任务API
    
    路由: POST /api/prediction/predict
    认证: 需要JWT令牌
    
    请求体 (JSON):
    {
        "item_id": int,          # MRI图像项ID
        "crop_position": {       # 裁剪位置
            "x": int,           # X坐标
            "y": int            # Y坐标
        },
        "needle_positions": [    # 针灸点位置列表
            {
                "x": int,       # X坐标
                "y": int        # Y坐标
            },
            ...
        ]
    }
    
    返回值:
    成功 (200):
    {
        "success": true,
        "message": "预测完成",
        "prediction": {
            "id": int,              # 预测记录ID
            "doctor_id": int,       # 医生ID
            "item_id": int,         # 图像项ID
            "pred_time": str,       # 预测时间
            "processed_rgb_path": str,    # 处理后的RGB图像路径
            "processed_depth_path": str,  # 处理后的深度图像路径
            "result_path": str,          # 预测结果图像路径
            ...
        }
    }
    
    错误响应:
    - 400: 请求数据无效
    - 404: MRI图像不存在或找不到对应深度图像
    - 500: 服务器内部错误
    
    处理流程:
    1. 验证请求数据
    2. 获取并验证MRI图像
    3. 查找对应的深度图像
    4. 创建预测目录
    5. 处理RGB和深度图像
    6. 执行预测
    7. 保存预测结果
    8. 创建预测记录
    
    错误处理:
    - 使用事务确保数据一致性
    - 详细的错误日志记录
    - 友好的错误消息返回
    
    安全考虑:
    - JWT认证保护
    - 输入验证
    - 文件操作安全性
    - 数据库事务保护
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': '请求数据为空'
            }), 400

        # 验证必要字段
        required_fields = ['item_id', 'crop_position', 'needle_positions']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'message': f'缺少必要字段: {field}'
                }), 400

        # 获取MRI图像项
        mri_item = MRISeqItem.query.get(data['item_id'])
        if not mri_item:
            return jsonify({
                'success': False,
                'message': 'MRI图像不存在'
            }), 404
        
        # 获取对应的深度图像
        sequence = mri_item.sequence
        depth_items = [item for item in sequence.items if item.item_type == 'depth']
        rgb_items = [item for item in sequence.items if item.item_type == 'rgb']
        
        # 找到对应的深度图像
        try:
            rgb_index = rgb_items.index(mri_item)
            depth_item = depth_items[rgb_index]
        except ValueError:
            return jsonify({
                'success': False,
                'message': '无法找到对应的深度图像'
            }), 404
        
        # 创建预测目录
        pred_dir = os.path.join('uploads', 'predictions', f'pred_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(os.path.join(current_app.root_path, '..', pred_dir), exist_ok=True)
        
        # 处理RGB图像
        rgb_path = os.path.join(current_app.root_path, '..', 'uploads', mri_item.file_path)
        cropped_rgb = crop_image(
            rgb_path,
            data['crop_position']['x'],
            data['crop_position']['y'],
            IMG_SIZE
        )
        processed_rgb_path = save_image(cropped_rgb, os.path.join(current_app.root_path, '..', pred_dir), 'processed_rgb.png')
        
        
        # 在深度图像中创建白色区域
        processed_depth = process_depth_image(
            os.path.join(current_app.root_path, '..', 'uploads', depth_item.file_path),
            data['needle_positions']
        )
        processed_depth_path = save_image(processed_depth, os.path.join(current_app.root_path, '..', pred_dir), 'processed_depth.png')
        
        # 处理深度图像
        processed_depth = crop_image(
            processed_depth_path,
            data['crop_position']['x'],
            data['crop_position']['y'],
            IMG_SIZE
        )

        # 调用预测函数
        result = predict(
            os.path.join(current_app.root_path, '..', processed_rgb_path),
            os.path.join(current_app.root_path, '..', processed_depth_path)
        )
        
        # 保存预测结果
        result_img = Image.fromarray((result * 255).astype(np.uint8))
        result_path = save_image(result_img, os.path.join(current_app.root_path, '..', pred_dir), 'result.png')
        
        # 创建预测记录
        prediction = PredRecord(
            doctor_id=get_jwt_identity(),
            item_id=data['item_id'],
            crop_x=data['crop_position']['x'],
            crop_y=data['crop_position']['y'],
            needle_positions=data['needle_positions'],
            processed_rgb_path=processed_rgb_path,
            processed_depth_path=processed_depth_path,
            result_path=result_path
        )
        
        db.session.add(prediction)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': '预测完成',
            'prediction': prediction.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error in create_prediction: {str(e)}")
        return jsonify({
            'success': False,
            'message': '预测失败，请稍后重试'
        }), 500

@bp.route('/sequence/<int:sequence_id>', methods=['GET'])
@jwt_required()
def get_sequence_predictions(sequence_id):
    """
    获取MRI序列的所有预测记录
    
    路由: GET /api/prediction/sequence/<sequence_id>
    认证: 需要JWT令牌
    
    路径参数:
        sequence_id (int): MRI序列ID
    
    返回值:
    成功 (200):
    {
        "predictions": [
            {
                "id": int,           # 预测记录ID
                "result_name": str,  # 结果文件名
                "pred_time": str     # 预测时间（ISO格式）
            },
            ...
        ]
    }
    
    错误响应:
    - 404: 序列不存在
    - 401: 未认证
    
    功能说明:
    1. 验证序列存在性
    2. 获取该序列所有相关的预测记录
    3. 格式化返回数据
    
    查询优化:
    - 使用JOIN操作提高查询效率
    - 只返回必要的字段
    
    安全考虑:
    - JWT认证保护
    - 使用get_or_404自动处理不存在的情况
    """
    # 验证序列是否存在
    sequence = MRISequence.query.get_or_404(sequence_id)
    
    # 获取序列的所有预测记录
    predictions = PredRecord.query.join(MRISeqItem).filter(MRISeqItem.seq_id == sequence_id).all()
    
    return jsonify({
        'predictions': [{
            'id': pred.pred_id,
            'result_name': pred.result_name,
            'pred_time': pred.pred_time.isoformat()
        } for pred in predictions]
    })

@bp.route('/<int:id>', methods=['GET'])
@jwt_required()
def get_prediction(id):
    prediction = PredRecord.query.get_or_404(id)
    
    return jsonify({
        'id': prediction.pred_id,
        'result_name': prediction.result_name,
        'pred_time': prediction.pred_time.isoformat()
    })

@bp.route('/records', methods=['GET'])
@token_required
def get_prediction_records():
    """获取预测记录列表"""
    try:
        # 获取当前医生的ID
        doctor_id = get_jwt_identity()
        current_app.logger.info(f"获取医生 {doctor_id} 的预测记录")
        
        # 获取该医生的所有预测记录
        records = PredRecord.query.filter_by(doctor_id=doctor_id).order_by(PredRecord.pred_time.desc()).all()
        current_app.logger.info(f"找到 {len(records)} 条预测记录")
        
        # 构造返回数据
        records_data = []
        for record in records:
            try:
                # 获取MRI图像项
                mri_item = MRISeqItem.query.get(record.item_id)
                if not mri_item:
                    current_app.logger.warning(f"找不到MRI图像项 {record.item_id}")
                    continue
                    
                # 获取序列和患者信息
                sequence = mri_item.sequence
                patient = sequence.patient
                
                record_data = {
                    'id': record.pred_id,
                    'patientName': patient.patient_name,
                    'patientId': patient.patient_id,
                    'sequenceName': sequence.seq_name,
                    'sequenceId': sequence.seq_id,
                    'predTime': record.pred_time.isoformat(),
                    'processed_rgb_path': record.processed_rgb_path,
                    'processed_depth_path': record.processed_depth_path,
                    'result_path': record.result_path
                }
                records_data.append(record_data)
                current_app.logger.debug(f"添加预测记录: {record_data}")
            except Exception as e:
                current_app.logger.error(f"处理预测记录 {record.pred_id} 时出错: {str(e)}")
                continue
        
        return jsonify({
            'success': True,
            'records': records_data
        })
        
    except Exception as e:
        current_app.logger.error(f"获取预测记录失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'获取预测记录失败: {str(e)}'
        }), 500