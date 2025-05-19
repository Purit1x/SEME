# MRI模块初始化文件
# 创建MRI相关的蓝图，用于组织MRI相关的路由和视图函数

from flask import Blueprint

# 创建名为 'mri' 的蓝图实例
bp = Blueprint('mri', __name__)

# 导入路由模块
from . import routes 