# 认证模块初始化文件
# 创建认证相关的蓝图，用于组织认证相关的路由和视图函数

from flask import Blueprint

# 创建名为 'auth' 的蓝图实例
bp = Blueprint('auth', __name__)

# 导入路由模块
from . import routes

# 确保路由被导入，使所有路由函数可用
from .routes import *