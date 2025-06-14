"""
测试MRI序列创建功能
"""
import pytest
from flask import url_for
import json
from app.models import MRISequence
from app import db
import io
import os

def test_create_sequence_patient_not_exist(client, auth_headers):
    """测试当患者不存在时创建序列"""    # 使用不存在的患者ID
    response = client.post(
        '/api/mri/patients/999/sequences',
        headers=auth_headers,
        data={
            'seq_name': 'test_sequence'
        }
    )
    
    assert response.status_code == 404
    data = json.loads(response.data)
    assert data['success'] is False
    assert data['message'] == '患者不存在'
    assert data['should_create_patient'] is True

def test_create_sequence_missing_seq_name(client, auth_headers, existing_patient):
    """测试未提供序列名称"""
    # 不提供seq_name
    response = client.post(
        f'/api/mri/patients/{existing_patient.patient_id}/sequences',
        headers=auth_headers,
        data={}
    )
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['success'] is False
    assert data['message'] == '缺少序列名称'

def test_create_sequence_empty_seq_name(client, auth_headers, existing_patient):
    """测试提供空的序列名称"""
    # 提供空的seq_name
    response = client.post(
        f'/api/mri/patients/{existing_patient.patient_id}/sequences',
        headers=auth_headers,
        data={'seq_name': ''}
    )
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['success'] is False
    assert data['message'] == '缺少序列名称'

def test_create_sequence_patient_exists(client, auth_headers, existing_patient):
    """测试使用有效的患者ID"""
    response = client.post(
        f'/api/mri/patients/{existing_patient.patient_id}/sequences',
        headers=auth_headers,
        data={
            'seq_name': 'test_sequence'
        }
    )
    
    # 此测试用例验证患者存在的情况
    # 注意：由于后续流程需要检查文件上传，所以这里应该返回400状态码
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['success'] is False
    assert data['message'] == '请同时上传RGB图像和深度图像'

def test_create_sequence_valid_seq_name(client, auth_headers, existing_patient):
    """测试提供有效的序列名称"""
    response = client.post(
        f'/api/mri/patients/{existing_patient.patient_id}/sequences',
        headers=auth_headers,
        data={
            'seq_name': 'seq01'  # 提供一个非空且有效的序列名称
        }
    )
    
    # 此测试用例验证序列名称有效的情况
    # 注意：由于后续流程需要检查文件上传，所以这里应该返回400状态码
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['success'] is False
    assert data['message'] == '请同时上传RGB图像和深度图像'

def test_create_sequence_duplicate_name(client, auth_headers, existing_patient):
    """测试创建同名序列"""
    # 首先创建一个序列
    sequence = MRISequence(
        seq_name='test_sequence',
        seq_dir=f'patient_{existing_patient.patient_id}/sequences/test_sequence',
        patient_id=existing_patient.patient_id
    )
    db.session.add(sequence)
    db.session.commit()
    
    # 尝试创建同名序列
    response = client.post(
        f'/api/mri/patients/{existing_patient.patient_id}/sequences',
        headers=auth_headers,
        data={
            'seq_name': 'test_sequence'  # 使用相同的序列名
        }
    )
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['success'] is False
    assert data['message'] == '序列名称已存在，请修改序列名称'

def test_create_sequence_unique_name(client, auth_headers, existing_patient):
    """测试创建新的唯一序列名"""
    response = client.post(
        f'/api/mri/patients/{existing_patient.patient_id}/sequences',
        headers=auth_headers,
        data={
            'seq_name': 'unique_sequence'  # 使用新的序列名
        }
    )
    
    # 由于后续需要上传文件，所以此时应该返回400
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['success'] is False
    assert data['message'] == '请同时上传RGB图像和深度图像'

def test_create_sequence_missing_files(client, auth_headers, existing_patient):
    """测试缺少文件的情况"""
    # 创建测试文件
    rgb_file = (io.BytesIO(b'fake rgb image data'), 'rgb_image.jpg')
    
    # 测试只上传RGB文件
    data = {
        'seq_name': 'test_sequence',
        'rgb_files[]': rgb_file
    }
    response = client.post(
        f'/api/mri/patients/{existing_patient.patient_id}/sequences',
        headers=auth_headers,
        data=data,
        content_type='multipart/form-data'
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['success'] is False
    assert data['message'] == '请同时上传RGB图像和深度图像'

    # 测试只上传Depth文件
    depth_file = (io.BytesIO(b'fake depth image data'), 'depth_image.jpg')
    data = {
        'seq_name': 'test_sequence',
        'depth_files[]': depth_file
    }
    response = client.post(
        f'/api/mri/patients/{existing_patient.patient_id}/sequences',
        headers=auth_headers,
        data=data,
        content_type='multipart/form-data'
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['success'] is False
    assert data['message'] == '请同时上传RGB图像和深度图像'

def test_create_sequence_with_files(client, auth_headers, existing_patient):
    """测试上传完整的文件集"""
    # 创建测试文件
    rgb_data = (io.BytesIO(b'fake rgb image data'), 'rgb_image.jpg')
    depth_data = (io.BytesIO(b'fake depth image data'), 'depth_image.jpg')
    
    data = {
        'seq_name': 'test_sequence_with_files'
    }
    
    # 添加文件到data中
    data['rgb_files[]'] = rgb_data
    data['depth_files[]'] = depth_data
    
    response = client.post(
        f'/api/mri/patients/{existing_patient.patient_id}/sequences',
        headers=auth_headers,
        data=data,
        content_type='multipart/form-data'
    )
    
    # 由于文件验证通过，系统会尝试保存文件并创建序列
    # 即使可能因为其他原因失败，但不应该是因为文件验证失败
    data = json.loads(response.data)
    assert data['message'] != '请同时上传RGB图像和深度图像'
    
    # 注意：这里不断言具体的状态码和消息
    # 因为在实际环境中可能因为文件存储或其他原因导致不同的结果

def test_create_sequence_unmatched_files(client, auth_headers, existing_patient):
    """测试上传不匹配数量的文件"""
    # 创建测试文件
    rgb_file1 = (io.BytesIO(b'fake rgb image 1'), 'rgb_image1.jpg')
    rgb_file2 = (io.BytesIO(b'fake rgb image 2'), 'rgb_image2.jpg')
    depth_file = (io.BytesIO(b'fake depth image'), 'depth_image.jpg')
    
    # 准备上传数据：2个RGB文件，1个depth文件
    data = {
        'seq_name': 'test_sequence_unmatched',
        'rgb_files[]': [rgb_file1, rgb_file2],  # 2个RGB文件
        'depth_files[]': [depth_file]  # 1个depth文件
    }
    
    response = client.post(
        f'/api/mri/patients/{existing_patient.patient_id}/sequences',
        headers=auth_headers,
        data=data,
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['success'] is False
    assert data['message'] == 'RGB图像和深度图像数量不匹配'

def test_create_sequence_matched_files(client, auth_headers, existing_patient):
    """测试上传匹配数量的文件"""
    # 创建测试文件
    rgb_file1 = (io.BytesIO(b'fake rgb image 1'), 'rgb_image1.jpg')
    rgb_file2 = (io.BytesIO(b'fake rgb image 2'), 'rgb_image2.jpg')
    depth_file1 = (io.BytesIO(b'fake depth image 1'), 'depth_image1.jpg')
    depth_file2 = (io.BytesIO(b'fake depth image 2'), 'depth_image2.jpg')
    
    # 准备上传数据：每种类型2个文件
    data = {
        'seq_name': 'test_sequence_matched',
        'rgb_files[]': [rgb_file1, rgb_file2],
        'depth_files[]': [depth_file1, depth_file2]
    }
    
    response = client.post(
        f'/api/mri/patients/{existing_patient.patient_id}/sequences',
        headers=auth_headers,
        data=data,
        content_type='multipart/form-data'
    )
    
    # 由于文件数量匹配，系统会继续进行后续处理
    # 注意：实际测试中可能因为文件存储等原因返回其他错误
    # 但不应该是因为文件数量不匹配的错误
    data = json.loads(response.data)
    assert data['message'] != 'RGB图像和深度图像数量不匹配'
    assert data['message'] != '请同时上传RGB图像和深度图像'

def test_create_sequence_invalid_file_types(client, auth_headers, existing_patient):
    """测试上传非法文件类型"""
    # 创建测试文件，包括合法和非法类型
    rgb_file = (io.BytesIO(b'fake rgb image'), 'rgb_image.txt')  # 非法扩展名
    depth_file = (io.BytesIO(b'fake depth image'), 'depth_image.jpg')  # 合法扩展名
    
    # 准备上传数据
    data = {
        'seq_name': 'test_sequence_invalid_type',
        'rgb_files[]': [rgb_file],
        'depth_files[]': [depth_file]
    }
    
    response = client.post(
        f'/api/mri/patients/{existing_patient.patient_id}/sequences',
        headers=auth_headers,
        data=data,
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['success'] is False
    assert data['message'] == '存在不支持的文件类型'
    # 验证返回的错误文件列表包含非法文件名
    assert 'rgb_image.txt' in data.get('invalid_rgb_files', [])

def test_create_sequence_valid_file_types(client, auth_headers, existing_patient):
    """测试上传合法文件类型"""
    # 创建不同格式的合法测试文件
    rgb_files = [
        (io.BytesIO(b'fake rgb jpg'), 'rgb_image1.jpg'),
        (io.BytesIO(b'fake rgb png'), 'rgb_image2.png')
    ]
    depth_files = [
        (io.BytesIO(b'fake depth jpg'), 'depth_image1.jpg'),
        (io.BytesIO(b'fake depth png'), 'depth_image2.png')
    ]
    
    # 准备上传数据
    data = {
        'seq_name': 'test_sequence_valid_types',
        'rgb_files[]': rgb_files,
        'depth_files[]': depth_files
    }
    
    response = client.post(
        f'/api/mri/patients/{existing_patient.patient_id}/sequences',
        headers=auth_headers,
        data=data,
        content_type='multipart/form-data'
    )
    
    # 文件格式验证应该通过
    data = json.loads(response.data)
    assert 'invalid_rgb_files' not in data
    assert 'invalid_depth_files' not in data
    # 注意：此处不测试最终的成功状态，因为后续可能有其他验证

def test_create_sequence_file_save_failure(client, auth_headers, existing_patient, monkeypatch):
    """测试文件保存失败的情况"""
    # 创建测试文件
    rgb_file = (io.BytesIO(b'fake rgb image'), 'rgb_image.jpg')
    depth_file = (io.BytesIO(b'fake depth image'), 'depth_image.jpg')
    
    # 准备上传数据
    data = {
        'seq_name': 'test_sequence_save_failure',
        'rgb_files[]': rgb_file,
        'depth_files[]': depth_file
    }
    
    # 模拟文件保存失败
    def mock_save(*args):
        raise OSError("模拟文件保存失败")
    
    monkeypatch.setattr("werkzeug.datastructures.FileStorage.save", mock_save)
    
    response = client.post(
        f'/api/mri/patients/{existing_patient.patient_id}/sequences',
        headers=auth_headers,
        data=data,
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 500
    data = json.loads(response.data)
    assert data['success'] is False
    assert data['message'] == '文件上传失败，请检查网络连接后重试'
    
    # 验证数据库回滚
    sequence = MRISequence.query.filter_by(
        patient_id=existing_patient.patient_id,
        seq_name='test_sequence_save_failure'
    ).first()
    assert sequence is None

def test_create_sequence_file_save_success(client, auth_headers, existing_patient):
    """测试文件保存成功的情况"""
    # 创建测试文件
    rgb_file = (io.BytesIO(b'fake rgb image'), 'rgb_image.jpg')
    depth_file = (io.BytesIO(b'fake depth image'), 'depth_image.jpg')
    
    # 准备上传数据
    data = {
        'seq_name': 'test_sequence_save_success',
        'rgb_files[]': rgb_file,
        'depth_files[]': depth_file
    }
    
    response = client.post(
        f'/api/mri/patients/{existing_patient.patient_id}/sequences',
        headers=auth_headers,
        data=data,
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['success'] is True
    assert data['message'] == '序列创建成功'
    
    # 验证数据库记录创建成功
    sequence = MRISequence.query.filter_by(
        patient_id=existing_patient.patient_id,
        seq_name='test_sequence_save_success'
    ).first()
    assert sequence is not None
    
    # 验证文件实际保存成功
    assert os.path.exists(os.path.join('uploads', sequence.seq_dir, 'rgb', 'image_1.jpg'))
    assert os.path.exists(os.path.join('uploads', sequence.seq_dir, 'depth', 'image_1.jpg'))

def test_create_sequence_main_logic_failure(client, auth_headers, existing_patient, monkeypatch):
    """测试主体逻辑发生异常的情况（如数据库操作失败）"""
    # 创建测试文件
    rgb_file = (io.BytesIO(b'fake rgb image'), 'rgb_image.jpg')
    depth_file = (io.BytesIO(b'fake depth image'), 'depth_image.jpg')
    
    # 准备上传数据
    data = {
        'seq_name': 'test_sequence_logic_failure',
        'rgb_files[]': rgb_file,
        'depth_files[]': depth_file
    }
    
    # 模拟数据库操作失败
    def mock_db_add(*args):
        raise Exception("模拟数据库操作失败")
    
    # 注入错误到数据库session.add操作
    monkeypatch.setattr(db.session, "add", mock_db_add)
    
    response = client.post(
        f'/api/mri/patients/{existing_patient.patient_id}/sequences',
        headers=auth_headers,
        data=data,
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 500
    data = json.loads(response.data)
    assert data['success'] is False
    assert data['message'] == '序列创建失败，请稍后重试'
    
    # 验证数据库回滚
    sequence = MRISequence.query.filter_by(
        patient_id=existing_patient.patient_id,
        seq_name='test_sequence_logic_failure'
    ).first()
    assert sequence is None
    
    # 验证文件未被创建
    base_path = os.path.join('uploads', f'patient_{existing_patient.patient_id}', 'sequences', 'test_sequence_logic_failure')
    assert not os.path.exists(base_path)


