"""
ResNet-ASPP融合网络 (ResNet with Atrous Spatial Pyramid Pooling)

本模块实现了ResNet主干网络与ASPP模块的融合架构，专门用于密集预测任务。

主要特点：
1. 架构融合
   - ResNet骨干网络（支持18/34/50）
   - ASPP多尺度特征提取
   - 多层特征输出

2. 可配置选项
   - 输入通道数自适应
   - 可选输出步长(8/16/32)
   - 灵活的扩张率设置
   - 多种主干网络选择

3. 应用场景
   - 语义分割
   - 实例分割
   - 针灸点检测
   - 密集预测任务

技术规格：
- 输入尺寸：512x512
- 特征层级：5级
- 主干网络：ResNet系列
- 后处理：ASPP模块

参考：
- DeepLab v3+
- ResNet论文
- ASPP原理

作者: Your Name
创建日期: 2025-06-16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .SPP import ASPP_simple, ASPP
from .ResNet import ResNet101, ResNet18, ResNet34, ResNet50

import time
INPUT_SIZE = 512

class ResNet_ASPP(nn.Module):
    """
    ResNet与ASPP的融合网络
    
    架构设计：
                                           ┌-> conv1_feat
                                           ├-> low_level_feat
    Input -> ResNet Backbone -> Features --├-> layer2_feat
                          └-> ASPP --------├-> layer3_feat
                                          ├-> layer4_feat
                                          └-> aspp_feat
    
    参数：
        nInputChannels (int): 输入通道数
        n_classes (int): 输出类别数
        os (int): 输出步长(8/16/32)
        backbone_type (str): 主干网络类型('resnet18'/'resnet34'/'resnet50')
    
    特征提取：
        - conv1_feat: 第一层卷积特征(1/2分辨率)
        - low_level_feat: 浅层特征(1/4分辨率)
        - layer2_feat: 中层特征(1/8分辨率)
        - layer3_feat: 深层特征(1/16分辨率)
        - layer4_feat: 最深层特征(1/32分辨率)
        - aspp_feat: ASPP处理后的特征
    
    配置说明：
    1. 扩张率(rates)设置：
       - os=16: [1, 6, 12, 18]
       - os=8/32: [1, 12, 24, 36]
    
    2. 通道数配置：
       - ResNet18/34: asppInputChannels=512
       - ResNet50: asppInputChannels=2048
       - ASPP输出: 256通道
       - 低层特征: 12通道
    
    注意事项：
    - 确保主干网络类型正确
    - 注意特征图尺寸的变化
    - 合理设置扩张率
    - 考虑内存占用
    """
    def __init__(self, nInputChannels, n_classes, os, backbone_type):
        super(ResNet_ASPP, self).__init__()

        self.os = os
        self.backbone_type = backbone_type
        
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8 or os == 32:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        if backbone_type == 'resnet18':
            self.backbone_features = ResNet18(nInputChannels, os, pretrained=False)
        elif backbone_type == 'resnet34':
            self.backbone_features = ResNet34(nInputChannels, os, pretrained=False)
        elif backbone_type == 'resnet50':
            self.backbone_features = ResNet50(nInputChannels, os, pretrained=False)
        else:
            raise NotImplementedError

        asppInputChannels = 512
        asppOutputChannels = 256
        lowInputChannels = 64
        lowOutputChannels = 12
        if backbone_type == 'resnet50': asppInputChannels = 2048
        
        self.aspp = ASPP(asppInputChannels, asppOutputChannels, rates)
        # self.last_conv_flow = nn.Sequential(
        #         nn.Conv2d(asppOutputChannels+lowOutputChannels, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #         nn.BatchNorm2d(256),
        #         nn.ReLU(),
        #         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #         nn.BatchNorm2d(256),
        #         nn.ReLU(),
        #         nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        #     )

    def forward(self, input):
        x, low_level_features, conv1_feat, layer2_feat, layer3_feat = self.backbone_features(input)
        layer4_feat = x
        if self.os == 32:
            x = F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.aspp(x)
        aspp_x = x
        x = self.last_conv(x)
        x = F.upsample(x, input.size()[2:], mode='bilinear', align_corners=True)

        return x, conv1_feat, low_level_features, layer2_feat, layer3_feat, layer4_feat, aspp_x
