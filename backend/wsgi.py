import os
import sys
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 添加当前目录到Python路径
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_dir)
logger.debug(f"Python path: {sys.path}")

from app import create_app
from commands import create_admin, fix_photo_paths

app = create_app()

# 注册命令
app.cli.add_command(create_admin)
app.cli.add_command(fix_photo_paths)

# 打印所有路由
logger.debug("Registered routes:")
for rule in app.url_map.iter_rules():
    logger.debug(f"{rule.endpoint}: {rule.methods} {rule.rule}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 