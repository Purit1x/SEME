# 患者模块初始化文件
# 创建患者相关的蓝图，用于组织患者相关的路由和视图函数

from flask import Blueprint

# 创建名为 'patient' 的蓝图实例
bp = Blueprint('patient', __name__)

# 导入路由模块
from app.patient import routes 