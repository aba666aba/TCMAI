# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0

"""
中医药智能问答系统 - 启动器
"""

import sys
import os

# 添加所有可能的路径
if getattr(sys, 'frozen', False):
    # 打包后运行
    ROOT_DIR = os.path.dirname(sys.executable)
    _internal = os.path.join(ROOT_DIR, '_internal')
    src = os.path.join(_internal, 'src')
    
    # 添加路径
    sys.path.insert(0, _internal)
    sys.path.insert(0, src)
    sys.path.insert(0, os.path.join(src, 'web'))
    sys.path.insert(0, os.path.join(src, 'core'))
    
    os.environ['TCMAI_ROOT'] = ROOT_DIR
else:
    # 开发环境
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(ROOT_DIR, 'src')
    sys.path.insert(0, src)
    os.environ['TCMAI_ROOT'] = ROOT_DIR

# 现在导入
from web.app import create_app

if __name__ == '__main__':
    app = create_app()
    app.run(host='127.0.0.1', port=5000, debug=False)
