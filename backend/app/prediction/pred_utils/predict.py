import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
from .model.BTSNet import BTSNet
import time
import torchvision.transforms as transforms
from PIL import Image

# 定义全局变量
IMG_SIZE = 352
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 初始化全局模型变量
model = None

def load_model():
    """加载模型"""
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
    """加载图像"""
    with open(img_path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def binary_loader(img_path):
    """加载二值图像"""
    with open(img_path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

def predict(rgb_image_path, depth_image_path):
    """根据rgb和深度图预测结果，以numpy数组形式返回预测结果"""
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
    

