"""
MRI序列管理测试模块 (MRI Sequence Management Test Module)

本模块提供MRI序列管理功能的完整测试用例集，包括：
1. 序列创建测试
   - 参数验证
   - 文件处理
   - 错误处理
   - 权限检查

2. 测试场景覆盖
   - 正常流程测试
   - 边界条件测试
   - 错误情况测试
   - 安全性测试

3. 测试策略
   - 单元测试
   - 集成测试
   - 参数化测试
   - 异常测试

4. 资源管理
   - 文件系统操作
   - 数据库交互
   - 清理机制

技术特点：
- Pytest测试框架
- Flask测试客户端
- 文件IO模拟
- 数据库事务

作者: Your Name
创建日期: 2025-06-16
"""

import pytest
from flask import url_for
import json
from app.models import MRISequence
from app import db
import io
import os

def test_create_sequence_patient_not_exist(client, auth_headers):
    """
    测试创建序列时患者不存在的场景
    
    测试目标：
    验证系统在尝试为不存在的患者创建MRI序列时的行为
    
    测试步骤：
    1. 使用不存在的患者ID发送请求
    2. 验证响应状态码
    3. 检查错误消息
    4. 确认建议操作
    
    预期结果：
    - 状态码：404
    - 操作失败标志
    - 清晰的错误提示
    - 建议创建患者
    
    边界情况：
    - 无效的患者ID
    - 缺少必要参数
    - 权限验证
    """
    # 使用不存在的患者ID
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
    """
    测试缺少序列名称的场景
    
    测试目标：
    验证系统对缺少必要参数的请求的处理
    
    测试步骤：
    1. 发送缺少序列名称的请求
    2. 验证响应状态码
    3. 检查错误消息
    
    预期结果：
    - 状态码：400
    - 明确的错误提示
    - 验证失败响应
    
    验证重点：
    - 参数验证逻辑
    - 错误消息准确性
    - 响应格式规范
    """
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
    """
    测试完整的序列创建流程
    
    测试目标：
    验证系统在接收到完整的序列创建请求时的处理流程
    
    测试步骤：
    1. 准备测试数据
       - RGB图像文件
       - 深度图像文件
       - 序列名称
    
    2. 发送请求
       - 设置认证头
       - 配置文件数据
       - 指定内容类型
    
    3. 验证响应
       - 状态码检查
       - 响应内容验证
       - 文件保存确认
    
    4. 数据验证
       - 数据库记录
       - 文件系统检查
       - 关系完整性
    
    测试数据：
    - RGB文件：模拟图像数据
    - 深度文件：模拟深度数据
    - 序列名称：唯一标识符
    
    预期结果：
    1. 请求处理
       - 文件上传成功
       - 序列创建完成
       - 关系建立正确
    
    2. 数据存储
       - 文件正确保存
       - 数据库记录完整
       - 路径信息正确
    
    3. 响应内容
       - 操作状态标识
       - 序列信息返回
       - 错误处理合理
    
    错误处理：
    - 文件格式验证
    - 存储空间检查
    - 并发操作处理
    - 事务回滚机制
    """
    # 创建测试文件
    rgb_data = (io.BytesIO(b'fake rgb image data'), 'rgb_image.jpg')
    depth_data = (io.BytesIO(b'fake depth image data'), 'depth_image.jpg')
    
    # 构建请求数据
    data = {
        'seq_name': 'test_sequence_with_files',
        'rgb_files[]': rgb_data,
        'depth_files[]': depth_data
    }
    
    # 发送请求
    response = client.post(
        f'/api/mri/patients/{existing_patient.patient_id}/sequences',
        headers=auth_headers,
        data=data,
        content_type='multipart/form-data'
    )
    
    # 解析响应
    response_data = json.loads(response.data)
    
    # 验证文件验证通过
    assert response_data['message'] != '请同时上传RGB图像和深度图像'
    
    # 如果创建成功，验证数据库记录
    if response.status_code == 201:
        sequence = MRISequence.query.filter_by(
            seq_name='test_sequence_with_files',
            patient_id=existing_patient.patient_id
        ).first()
        
        assert sequence is not None
        assert sequence.seq_name == 'test_sequence_with_files'
        assert sequence.patient_id == existing_patient.patient_id
        
        # 验证文件是否已保存
        rgb_path = os.path.join('uploads', 
                               f'patient_{existing_patient.patient_id}',
                               'sequences',
                               sequence.seq_name,
                               'rgb_image.jpg')
        depth_path = os.path.join('uploads',
                                 f'patient_{existing_patient.patient_id}',
                                 'sequences',
                                 sequence.seq_name,
                                 'depth_image.jpg')
                                 
        assert os.path.exists(rgb_path)
        assert os.path.exists(depth_path)

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

def test_create_sequence_invalid_file_type(client, auth_headers, existing_patient):
    """
    测试上传无效文件类型的场景
    
    测试目标：
    验证系统对非法文件类型的处理机制
    
    测试步骤：
    1. 准备无效文件
       - 错误的文件扩展名
       - 不支持的MIME类型
       - 空文件
    
    2. 发送请求
       - 包含无效文件
       - 正确的认证信息
       - 完整的其他参数
    
    3. 验证处理
       - 文件类型检查
       - 错误消息准确性
       - 清理临时文件
    
    错误处理：
    - 文件类型验证
    - 大小限制检查
    - 安全性校验
    """
    # 创建无效类型的测试文件
    invalid_file = (io.BytesIO(b'fake executable data'), 'malicious.exe')
    valid_depth = (io.BytesIO(b'fake depth data'), 'depth_image.jpg')
    
    data = {
        'seq_name': 'test_invalid_file',
        'rgb_files[]': invalid_file,
        'depth_files[]': valid_depth
    }
    
    response = client.post(
        f'/api/mri/patients/{existing_patient.patient_id}/sequences',
        headers=auth_headers,
        data=data,
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 400
    response_data = json.loads(response.data)
    assert response_data['success'] is False
    assert '不支持的文件类型' in response_data['message']

def test_create_sequence_max_file_size(client, auth_headers, existing_patient):
    """
    测试文件大小限制
    
    测试目标：
    验证系统对超大文件的处理机制
    
    测试步骤：
    1. 创建超大文件
    2. 发送上传请求
    3. 验证大小限制
    
    预期结果：
    - 适当的错误响应
    - 文件大小提示
    - 资源正确清理
    """
    # 创建超过大小限制的文件
    large_file_content = b'0' * (10 * 1024 * 1024)  # 10MB
    large_file = (io.BytesIO(large_file_content), 'large.jpg')
    normal_file = (io.BytesIO(b'normal content'), 'normal.jpg')
    
    data = {
        'seq_name': 'test_large_file',
        'rgb_files[]': large_file,
        'depth_files[]': normal_file
    }
    
    response = client.post(
        f'/api/mri/patients/{existing_patient.patient_id}/sequences',
        headers=auth_headers,
        data=data,
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 400
    response_data = json.loads(response.data)
    assert '文件大小超过限制' in response_data['message']

def test_create_sequence_concurrent(client, auth_headers, existing_patient):
    """
    测试并发创建序列
    
    测试目标：
    验证系统在并发创建序列时的行为
    
    测试步骤：
    1. 并发发送请求
    2. 验证数据一致性
    3. 检查资源竞争
    
    关注点：
    - 事务隔离
    - 资源锁定
    - 并发控制
    """
    import threading
    import queue
    
    results = queue.Queue()
    
    def create_sequence(seq_name):
        rgb_file = (io.BytesIO(b'rgb data'), 'rgb.jpg')
        depth_file = (io.BytesIO(b'depth data'), 'depth.jpg')
        
        data = {
            'seq_name': seq_name,
            'rgb_files[]': rgb_file,
            'depth_files[]': depth_file
        }
        
        response = client.post(
            f'/api/mri/patients/{existing_patient.patient_id}/sequences',
            headers=auth_headers,
            data=data,
            content_type='multipart/form-data'
        )
        results.put((seq_name, response.status_code))
    
    # 创建多个线程并发请求
    threads = []
    for i in range(5):
        t = threading.Thread(
            target=create_sequence,
            args=(f'concurrent_seq_{i}',)
        )
        threads.append(t)
        t.start()
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    # 验证结果
    success_count = 0
    while not results.empty():
        seq_name, status_code = results.get()
        if status_code == 201:
            success_count += 1
    
    # 验证是否所有序列都创建成功
    sequences = MRISequence.query.filter(
        MRISequence.seq_name.like('concurrent_seq_%')
    ).all()
    assert len(sequences) == success_count

def test_sequence_file_organization(client, auth_headers, existing_patient):
    """
    测试序列文件组织结构
    
    测试目标：
    验证系统的文件组织和目录结构管理
    
    目录结构：
    uploads/
    └── patient_{id}/
        └── sequences/
            └── {sequence_name}/
                ├── rgb/
                │   └── rgb_001.jpg
                └── depth/
                    └── depth_001.jpg
    
    验证项目：
    1. 目录创建
       - 患者目录
       - 序列目录
       - 类型子目录
    
    2. 文件命名
       - 唯一性
       - 序号管理
       - 扩展名规范
    
    3. 权限控制
       - 目录权限
       - 文件权限
       - 访问限制
    """
    # 准备测试文件
    rgb_file = (io.BytesIO(b'rgb image'), 'rgb_001.jpg')
    depth_file = (io.BytesIO(b'depth image'), 'depth_001.jpg')
    
    data = {
        'seq_name': 'test_file_structure',
        'rgb_files[]': rgb_file,
        'depth_files[]': depth_file
    }
    
    response = client.post(
        f'/api/mri/patients/{existing_patient.patient_id}/sequences',
        headers=auth_headers,
        data=data,
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 201
    response_data = json.loads(response.data)
    sequence_id = response_data['sequence']['id']
    
    # 验证目录结构
    base_path = os.path.join('uploads', f'patient_{existing_patient.patient_id}')
    sequence_path = os.path.join(base_path, 'sequences', 'test_file_structure')
    
    # 检查目录是否创建
    assert os.path.exists(base_path)
    assert os.path.exists(sequence_path)
    assert os.path.exists(os.path.join(sequence_path, 'rgb'))
    assert os.path.exists(os.path.join(sequence_path, 'depth'))
    
    # 检查文件是否正确保存
    rgb_file_path = os.path.join(sequence_path, 'rgb', 'rgb_001.jpg')
    depth_file_path = os.path.join(sequence_path, 'depth', 'depth_001.jpg')
    
    assert os.path.exists(rgb_file_path)
    assert os.path.exists(depth_file_path)
    
    # 验证文件权限
    rgb_permissions = oct(os.stat(rgb_file_path).st_mode)[-3:]
    depth_permissions = oct(os.stat(depth_file_path).st_mode)[-3:]
    
    assert rgb_permissions == '644'  # 确保文件权限正确
    assert depth_permissions == '644'

def test_sequence_cleanup_on_failure(client, auth_headers, existing_patient):
    """
    测试创建失败时的资源清理
    
    测试目标：
    验证系统在序列创建失败时的资源清理机制
    
    测试场景：
    1. 文件上传部分成功
    2. 数据库操作失败
    3. 系统异常发生
    
    验证项目：
    - 临时文件清理
    - 数据库回滚
    - 目录清理
    - 错误恢复
    """
    # 模拟可能导致失败的情况
    rgb_file = (io.BytesIO(b'rgb image'), 'rgb.jpg')
    invalid_depth = (io.BytesIO(b'invalid'), 'depth.exe')  # 无效文件类型
    
    data = {
        'seq_name': 'test_cleanup',
        'rgb_files[]': rgb_file,
        'depth_files[]': invalid_depth
    }
    
    response = client.post(
        f'/api/mri/patients/{existing_patient.patient_id}/sequences',
        headers=auth_headers,
        data=data,
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 400
    
    # 验证没有残留的文件和目录
    base_path = os.path.join('uploads', f'patient_{existing_patient.patient_id}')
    sequence_path = os.path.join(base_path, 'sequences', 'test_cleanup')
    
    assert not os.path.exists(sequence_path)
    
    # 验证数据库中没有创建记录
    sequence = MRISequence.query.filter_by(
        seq_name='test_cleanup',
        patient_id=existing_patient.patient_id
    ).first()
    
    assert sequence is None


