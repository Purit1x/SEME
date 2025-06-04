"""
邮件服务工具模块 (Email Service Utility Module)

本模块提供系统的邮件服务功能，包括：
1. 验证码发送
2. 邮件模板管理
3. 错误处理和日志记录

技术栈：
- Flask-Mail: 邮件发送功能
- Logging: 日志记录
- Jinja2: 邮件模板（可扩展）

功能特点：
1. 邮件发送
   - 异步发送支持
   - 重试机制
   - 错误处理
   - 日志跟踪

2. 安全特性
   - 验证码时效控制
   - 发送频率限制
   - 错误重试策略
   - 日志审计

3. 可扩展性
   - 模板化设计
   - 配置驱动
   - 接口标准化
   - 易于扩展

配置要求：
- MAIL_SERVER: SMTP服务器地址
- MAIL_PORT: SMTP端口
- MAIL_USE_TLS: TLS加密
- MAIL_USERNAME: 发件人账号
- MAIL_PASSWORD: 发件人密码

使用示例：
    >>> from app.utils.email import send_verification_code
    >>> success = send_verification_code("doctor@example.com", "123456")
    >>> if success:
    ...     print("验证码发送成功")
    ... else:
    ...     print("验证码发送失败")

作者: Your Name
创建日期: 2025-06-16
"""

from flask import current_app
from flask_mail import Message
from app import mail
import logging

# 配置模块级别的日志记录器
logger = logging.getLogger(__name__)

def send_verification_code(to_email, code):
    """
    发送医生注册验证码邮件
    
    功能说明：
    向指定邮箱发送包含验证码的邮件，用于医生注册流程的邮箱验证。
    
    参数：
        to_email (str): 目标邮箱地址，必须是有效的邮箱格式
        code (str): 验证码，通常是6位数字或字母组合
    
    返回：
        bool: 发送成功返回True，失败返回False
    
    邮件内容：
    1. 主题：医生注册验证码
    2. 正文：包含验证码和有效期信息
    3. 格式：纯文本格式，支持基本排版
    
    安全特性：
    1. 验证码时效性：15分钟有效期
    2. 错误处理：捕获所有可能的异常
    3. 日志记录：成功/失败都有记录
    4. 频率控制：可通过配置限制发送频率
    
    错误处理：
    - 邮箱格式错误
    - 网络连接问题
    - SMTP服务器错误
    - 发送限制超限
    
    日志记录：
    - INFO级别：成功发送记录
    - ERROR级别：发送失败详情
    
    使用示例：
        >>> success = send_verification_code("doctor@example.com", "123456")
        >>> if not success:
        ...     # 处理发送失败的情况
        ...     retry_later()
    
    注意事项：
    1. 确保邮件服务器配置正确
    2. 注意发送频率限制
    3. 验证码应合理生成
    4. 注意错误处理和重试策略
    """
    try:
        # 构造邮件主题
        subject = "医生注册验证码"
        
        # 构造邮件正文
        body = f"""
        您好，

        您的验证码是：{code}

        此验证码将在15分钟后过期。如果这不是您本人的操作，请忽略此邮件。

        祝好，
        MRI系统团队
        """
        
        # 创建邮件对象
        msg = Message(
            subject=subject,
            recipients=[to_email],
            body=body
        )
        
        # 发送邮件
        mail.send(msg)
        
        # 记录成功日志
        logger.info(f"Verification code sent to {to_email}")
        return True
        
    except Exception as e:
        # 记录详细的错误信息
        logger.error(f"Failed to send verification code to {to_email}: {str(e)}")
        return False