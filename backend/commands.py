import click
from flask.cli import with_appcontext
from app.models import Administrator, Patient
from app import db

@click.command('create-admin')
@click.argument('admin_id')
@click.argument('password')
@with_appcontext
def create_admin(admin_id, password):
    """创建管理员账户"""
    try:
        admin = Administrator(admin_id=admin_id)
        admin.set_password(password)
        db.session.add(admin)
        db.session.commit()
        click.echo(f'成功创建管理员账户 {admin_id}')
    except Exception as e:
        db.session.rollback()
        click.echo(f'创建管理员账户失败: {str(e)}')

@click.command('fix-photo-paths')
@with_appcontext
def fix_photo_paths():
    """修复数据库中的照片路径（移除 uploads 前缀）"""
    try:
        patients = Patient.query.all()
        fixed_count = 0
        
        for patient in patients:
            if patient.photo_path and patient.photo_path.startswith('uploads\\'):
                # 移除 uploads 前缀
                patient.photo_path = patient.photo_path[8:]  # len('uploads\\') == 8
                fixed_count += 1
        
        if fixed_count > 0:
            db.session.commit()
            click.echo(f'成功修复 {fixed_count} 个照片路径')
        else:
            click.echo('没有需要修复的照片路径')
            
    except Exception as e:
        db.session.rollback()
        click.echo(f'修复照片路径失败: {str(e)}') 