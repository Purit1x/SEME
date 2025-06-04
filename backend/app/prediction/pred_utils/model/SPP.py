"""
空间金字塔池化模块 (Spatial Pyramid Pooling, SPP)

本模块实现了三种SPP变体，专门用于多尺度特征提取和上下文信息整合：
1. ASPP_simple: 简化版空洞空间金字塔池化
2. ASPP_module: 基础ASPP构建单元
3. ASPP: 完整的空洞空间金字塔池化实现

核心创新：
1. 多尺度感知
   - 并行空洞卷积
   - 全局上下文捕获
   - 特征自适应融合

2. 计算效率
   - 共享参数设计
   - 并行特征提取
   - 优化的内存使用

3. 特征增强
   - 多尺度特征整合 
   - 感受野自适应调节
   - 上下文信息注入

技术亮点：
- 支持任意输入尺寸
- 可配置扩张率
- 动态特征融合
- 高效内存使用

应用场景：
- 语义分割
- 实例分割
- 目标检测
- 针灸点定位

作者: Your Name
创建日期: 2025-06-16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ASPP_simple(nn.Module):
    """
    简化版空洞空间金字塔池化模块
    
    设计理念：
    通过多个并行的空洞卷积分支和全局上下文分支，
    在不同尺度上捕获特征，实现高效的多尺度特征提取。
    
    结构示意图：
                     ┌── 1x1 Conv ──┐
                     ├── 3x3 Conv(r1)┤
    Input ──► Split ├── 3x3 Conv(r2)├──► Concat ──► 1x1 Conv ──► Output
                     ├── 3x3 Conv(r3)┤
                     └── GlobalPool ──┘
    
    参数配置：
        inplanes (int): 输入特征通道数
        planes (int): 输出特征通道数
        rates (list): 空洞卷积扩张率列表，默认[1,6,12,18]
    
    分支说明：
    1. 标准卷积分支
       - 1x1卷积
       - 降低计算量
       - 特征变换
    
    2. 空洞卷积分支
       - 三个3x3空洞卷积
       - 不同扩张率
       - 多尺度特征
    
    3. 全局上下文分支
       - 全局平均池化
       - 获取全局信息
       - 上采样恢复尺寸
    
    特性优势：
    1. 计算效率
       - 并行处理
       - 参数共享
       - 内存优化
    
    2. 特征提取
       - 多尺度感知
       - 局部-全局结合
       - 上下文理解
    
    3. 易用性
       - 模块化设计
       - 简单集成
       - 可配置性强
    
    使用示例：
        >>> aspp = ASPP_simple(256, 128)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> out = aspp(x)  # 输出尺寸保持不变
    """
    def __init__(self, inplanes, planes, rates=[1, 6, 12, 18]):
        super(ASPP_simple, self).__init__()

        self.aspp0 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, 
                     dilation=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.aspp1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, 
                     padding=rates[1], dilation=rates[1], bias=False),
            nn.BatchNorm2d(planes)
        )        
        self.aspp2 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, 
                     padding=rates[2], dilation=rates[2], bias=False),
            nn.BatchNorm2d(planes)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, 
                     padding=rates[3], dilation=rates[3], bias=False),
            nn.BatchNorm2d(planes)
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        )

        self.reduce = nn.Sequential(
            nn.Conv2d(planes*5, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x (Tensor): 输入特征图 [B, C, H, W]
        
        返回:
            Tensor: 多尺度特征融合结果
        
        处理流程:
        1. 并行处理各分支
        2. 全局特征上采样
        3. 特征拼接融合
        4. 通道压缩
        """
        x0 = self.aspp0(x)           # 1x1 常规卷积
        x1 = self.aspp1(x)           # 3x3 空洞卷积(r1)
        x2 = self.aspp2(x)           # 3x3 空洞卷积(r2)
        x3 = self.aspp3(x)           # 3x3 空洞卷积(r3)
        x4 = self.global_avg_pool(x)  # 全局上下文
        x4 = F.upsample(x4, x3.size()[2:], mode='bilinear', align_corners=True)
        
        # 特征融合
        x = torch.cat((x0, x1, x2, x3, x4), dim=1)
        x = self.reduce(x)
        return x 

class ASPP_module(nn.Module):
    """
    ASPP基础构建单元
    
    设计目标：
    提供一个灵活的空洞卷积单元，可根据rate参数
    自动调整为普通1x1卷积或带扩张率的3x3卷积。
    
    结构设计：
                                    ┌── BN ──┐
    Input ──► Atrous Conv (1x1/3x3) ├── ReLU ├──► Output
                                    └────────┘
    
    参数说明：
        inplanes (int): 输入特征通道数
        planes (int): 输出特征通道数
        rate (int): 空洞卷积扩张率
                   - rate=1: 使用1x1常规卷积
                   - rate>1: 使用3x3空洞卷积
    
    实现特点：
    1. 自适应卷积
       - 根据rate自动选择卷积类型
       - 动态调整padding
       - 优化参数使用
    
    2. 标准化和激活
       - 批归一化层
       - ReLU激活函数
       - 改善收敛性
    
    3. 权重初始化
       - Kaiming初始化
       - 针对ReLU优化
       - 稳定训练过程
    
    最佳实践：
    1. 选择合适的rate
       - 小rate: 关注局部特征
       - 大rate: 捕获远程依赖
    
    2. 通道数配置
       - 降维: inplanes > planes
       - 升维: inplanes < planes
       - 维持: inplanes = planes
    """
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        
        # 根据rate选择卷积类型
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
            
        self.atrous_convolution = nn.Conv2d(
            inplanes, planes, kernel_size=kernel_size,
            stride=1, padding=padding, dilation=rate, bias=False
        )
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.__init_weight()

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x (Tensor): 输入特征图
            
        返回:
            Tensor: 特征变换结果
        """
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)

    def __init_weight(self):
        """
        权重初始化函数
        
        特点:
        - 使用Kaiming初始化卷积层
        - BatchNorm层使用常数初始化
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    """
    完整的空洞空间金字塔池化模块
    
    架构特点：
    整合多个ASPP模块和全局上下文特征，实现全方位的
    多尺度特征提取和上下文信息融合。
    
    系统架构：
                     ┌── ASPP(r1) ──┐
                     ├── ASPP(r2) ──┤
    Input ──► Split ├── ASPP(r3) ──├──► Concat ──► Conv+BN ──► ReLU ──► Output
                     ├── ASPP(r4) ──┤
                     └── GlobalPool ─┘
    
    参数配置：
        inplanes (int): 输入特征通道数
        planes (int): 输出特征通道数
        rates (list): 空洞卷积扩张率列表
    
    模块组成：
    1. 特征提取分支
       - 四个并行ASPP模块
       - 不同扩张率配置
       - 独立特征学习
    
    2. 全局信息分支
       - 自适应平均池化
       - 1x1卷积降维
       - 上采样还原尺寸
    
    3. 特征融合部分
       - 通道拼接
       - 1x1卷积整合
       - ReLU激活
    
    技术优势：
    1. 感受野覆盖
       - 多尺度特征
       - 全局上下文
       - 局部细节
    
    2. 计算效率
       - 并行计算
       - 特征重用
       - 内存优化
    
    3. 特征表达
       - 多层次语义
       - 空间依赖关系
       - 上下文理解
    
    使用建议：
    1. 输入配置
       - 合适的特征图尺寸
       - 规范化的特征分布
       - 合理的通道数设置
    
    2. 训练策略
       - 学习率调整
       - 批量大小选择
       - 梯度裁剪
    
    示例应用：
        >>> aspp = ASPP(2048, 256, [1,6,12,18])
        >>> x = torch.randn(1, 2048, 32, 32)
        >>> out = aspp(x)  # 通道数变为256
    """
    def __init__(self, inplanes, planes, rates):
        super(ASPP, self).__init__()

        # 初始化四个并行ASPP模块
        self.aspp1 = ASPP_module(inplanes, planes, rate=rates[0])
        self.aspp2 = ASPP_module(inplanes, planes, rate=rates[1])
        self.aspp3 = ASPP_module(inplanes, planes, rate=rates[2])
        self.aspp4 = ASPP_module(inplanes, planes, rate=rates[3])

        self.relu = nn.ReLU()

        # 全局上下文分支
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, planes, 1, stride=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )

        # 特征融合
        self.conv1 = nn.Conv2d(planes*5, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x (Tensor): 输入特征图 [B, C, H, W]
            
        返回:
            Tensor: 多尺度特征融合结果
            
        处理流程:
        1. 并行ASPP处理
        2. 全局上下文提取
        3. 特征对齐融合
        4. 通道降维
        """
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x