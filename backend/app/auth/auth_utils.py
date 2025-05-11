# 认证工具模块
# 提供用于处理用户认证的装饰器和工具函数

from functools import wraps
from flask import jsonify, request
from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity

def token_required(f):
    """
    用于保护需要认证的路由的装饰器
    
    使用方法：
    @bp.route('/protected')
    @token_required
    def protected_route():
        return 'This is a protected route'
    
    功能：
    1. 验证请求中的JWT token
    2. 如果token有效，允许访问被装饰的路由
    3. 如果token无效或过期，返回401错误
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            # 验证JWT token的有效性
            verify_jwt_in_request()
            # 获取当前用户的ID
            current_user_id = get_jwt_identity()
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({
                'success': False,
                'message': '认证失败，请重新登录'
            }), 401
    return decorated 