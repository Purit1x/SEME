from functools import wraps
from flask import jsonify, request
from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            # 验证JWT token
            verify_jwt_in_request()
            # 获取当前用户ID
            current_user_id = get_jwt_identity()
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({
                'success': False,
                'message': '认证失败，请重新登录'
            }), 401
    return decorated 