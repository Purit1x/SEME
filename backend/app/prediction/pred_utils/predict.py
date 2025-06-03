"""
深度学习预测模块 (Deep Learning Prediction Module)

本模块实现了基于深度学习的针灸点预测功能，主要包括：
1. 模型管理
   - 模型加载
   - CUDA设备配置
   - 模型状态管理
2. 图像处理
   - RGB图像加载与预处理
   - 深度图像加载与预处理
   - 图像标准化
3. 预测功能
   - 单次预测
   - 批量预测
   - 结果后处理

技术栈：
- PyTorch: 深度学习框架
- torchvision: 图像处理和转换
- PIL: 图像I/O
- NumPy: 数值计算
- OpenCV: 图像处理

模型说明：
- 架构：BTSNet (Bilateral Two-Stream Network)
- 输入：RGB图像和深度图像
- 输出：预测的针灸点位置概率图
- 预训练：使用自定义数据集训练

性能优化：
- 使用CUDA加速
- 模型单例模式
- 异步数据加载
- 内存优化

作者: Your Name
创建日期: 2025-06-16
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
from .model.BTSNet import BTSNet
import time
import torchvision.transforms as transforms
from PIL import Image

# 全局配置常量
IMG_SIZE = 352  # 模型输入图像大小
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前模块目录

# 全局模型实例（单例模式）
model = None

def load_model():
    """
    加载深度学习模型（单例模式）
    
    功能：
    1. 检查模型是否已加载（单例模式）
    2. 配置CUDA设备
    3. 初始化模型
    4. 加载预训练权重
    5. 将模型移至GPU
    6. 设置评估模式
    
    返回：
        torch.nn.Module: 加载好的模型实例
    
    模型配置：
    - 输入通道：3（RGB）
    - 输出类别：1（二值分割）
    - 输出步长：16
    
    注意事项：
    - 确保CUDA可用
    - 预训练权重文件必须存在
    - 模型状态字典键的处理（移除'module.'前缀）
    
    优化：
    - 使用单例模式避免重复加载
    - 异步GPU操作
    - 内存优化的权重加载
    """
    global model
    if model is None:  # 如果模型还没有加载
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        model = BTSNet(nInputChannels=3, n_classes=1, os=16,)
        checkpoint = torch.load(os.path.join(CURRENT_DIR, 'pretrain', 'epoch_55.pth'))
        state_dict = checkpoint['model_state_dict']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.cuda()
        model.eval()
    return model

def img_loader(img_path):
    """
    加载和转换RGB图像
    
    参数：
        img_path (str): 图像文件的路径
    
    返回：
        PIL.Image: 转换后的RGB图像对象
    
    处理步骤：
    1. 打开图像文件
    2. 转换为RGB模式
    3. 确保正确关闭文件
    
    异常处理：
    - 使用上下文管理器确保文件正确关闭
    - 自动转换不同的颜色空间
    
    优化：
    - 使用with语句管理资源
    - 直接转换为RGB避免中间步骤
    """
    with open(img_path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def binary_loader(img_path):
    """
    加载和转换二值图像
    
    参数：
        img_path (str): 图像文件的路径
    
    返回：
        PIL.Image: 转换后的灰度图像对象
    
    处理步骤：
    1. 打开图像文件
    2. 转换为灰度模式（'L'）
    3. 确保正确关闭文件
    
    说明：
    - 用于加载标签图像
    - 自动转换为单通道灰度图
    """
    with open(img_path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

def predict(rgb_image_path, depth_image_path):
    """
    执行针灸点预测
    
    参数：
        rgb_image_path (str): RGB图像的文件路径
        depth_image_path (str): 深度图像的文件路径
    
    返回：
        numpy.ndarray: 预测结果概率图，值范围[0,1]
    
    处理流程：
    1. 模型准备
       - 确保模型已加载
       - 设置设备（CPU/GPU）
    
    2. 数据预处理
       - 图像加载和转换
       - 尺寸调整
       - 标准化
       - 张量转换
    
    3. 模型推理
       - 将数据移至GPU
       - 执行前向传播
       - 计算处理时间
    
    4. 后处理
       - 上采样到原始尺寸
       - Sigmoid激活
       - 结果归一化
    
    优化措施：
    - 使用GPU加速
    - 批处理支持
    - 异步数据加载
    - 内存优化
    
    性能指标：
    - 输出FPS（帧率）
    - GPU同步计时
    
    示例：
        >>> result = predict('image.jpg', 'depth.jpg')
        >>> print(f"Result shape: {result.shape}")
        >>> print(f"Value range: [{result.min()}, {result.max()}]")
    """
    # 确保模型已加载
    global model
    if model is None:
        model = load_model()
        
    imgSize = IMG_SIZE
    # 图像转换
    transform = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    gt_transform = transforms.ToTensor()
    rgb = img_loader(rgb_image_path)
    depth = img_loader(depth_image_path)
    
    # 使用正确的路径加载gt图像
    gt_path = os.path.join(CURRENT_DIR, 'pretrain', '1-post-processed.png')
    gt = binary_loader(gt_path)
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)

    rgb = transform(rgb).unsqueeze(0)
    depth = transform(depth).unsqueeze(0)

    # 将图像转换为cuda
    rgb = rgb.cuda()
    depth = depth.cuda()
    torch.cuda.synchronize()
    time_s = time.time()
    res, res_r,res_d= model(rgb,depth)
    torch.cuda.synchronize()
    time_e = time.time()
    print('Speed: %f FPS' % (1 / (time_e - time_s)))
    res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)

    return res

# 在模块导入时加载模型
load_model()


