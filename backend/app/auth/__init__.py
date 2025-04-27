from flask import Blueprint

bp = Blueprint('auth', __name__)

from . import routes

# 确保路由被导入
from .routes import *