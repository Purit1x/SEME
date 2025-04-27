from flask import current_app
from flask_mail import Message
from app import mail
import logging

logger = logging.getLogger(__name__)

def send_verification_code(to_email, code):
    """发送验证码邮件"""
    try:
        subject = "医生注册验证码"
        body = f"""
        您好，

        您的验证码是：{code}

        此验证码将在15分钟后过期。如果这不是您本人的操作，请忽略此邮件。

        祝好，
        MRI系统团队
        """
        
        msg = Message(
            subject=subject,
            recipients=[to_email],
            body=body
        )
        mail.send(msg)
        logger.info(f"Verification code sent to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send verification code to {to_email}: {str(e)}")
        return False 