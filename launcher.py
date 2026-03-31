# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0

"""
中医药智能问答系统 - Windows 启动器

打包后自动启动 Flask 服务器并打开浏览器
"""

import os
import sys
import webbrowser
import time
import socket
from pathlib import Path

# 获取运行目录
if getattr(sys, 'frozen', False):
    # 打包后运行
    ROOT_DIR = Path(sys.executable).parent
    print(f"运行目录：{ROOT_DIR}")
    
    # 添加内部路径到 sys.path
    internal_path = ROOT_DIR / '_internal'
    if internal_path.exists():
        sys.path.insert(0, str(internal_path))
        sys.path.insert(0, str(internal_path / 'src'))
else:
    # 开发环境运行
    ROOT_DIR = Path(__file__).parent.parent
    print(f"开发目录：{ROOT_DIR}")
    sys.path.insert(0, str(ROOT_DIR / 'src'))

# 设置环境变量
os.environ['TCMAI_ROOT'] = str(ROOT_DIR)

# 端口配置：优先从环境变量读取
_port_str = os.environ.get('FLASK_PORT', '5000')
try:
    PORT = int(_port_str)
except ValueError:
    print(f"[警告] 环境变量 FLASK_PORT='{_port_str}' 非法，已回退为默认端口 5000")
    PORT = 5000
HOST = os.environ.get('FLASK_HOST', '127.0.0.1')

def check_port(port):
    """检查端口是否可用"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result == 0

def main():
    print("="*60)
    print("  中医药智能问答系统")
    print("="*60)
    print()
    
    # 导入 Flask 应用
    print("[1/3] 加载系统...")
    try:
        from web.app import create_app
        print("      ✓ 系统加载成功")
    except Exception as e:
        error_msg = f"系统加载失败：{e}"
        print(f"      ✗ {error_msg}")
        print()
        print("按回车键退出...")
        try:
            input()
        except:
            time.sleep(3)
        return
    
    # 创建应用
    print("[2/3] 启动服务器...")
    app = create_app()
    
    # 检查端口
    if check_port(PORT):
        print(f"      ✗ 端口 {PORT} 已被占用")
        print("      请关闭其他程序后重试")
        print()
        print("按回车键退出...")
        try:
            input()
        except:
            time.sleep(3)
        return
    
    print(f"      ✓ 服务器准备就绪")
    print()
    print("-"*60)
    print(f"  访问地址：http://localhost:{PORT}")
    print("  按 Ctrl+C 可停止服务器")
    print("-"*60)
    print()
    
    # 打开浏览器
    print("[3/3] 打开浏览器...")
    time.sleep(2)  # 等待服务器启动
    webbrowser.open(f'http://localhost:{PORT}')
    print("      ✓ 浏览器已打开")
    print()
    
    # 运行服务器
    print("服务器运行中...")
    try:
        app.run(host=HOST, port=PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\n服务器已停止")
    except Exception as e:
        print(f"\n错误：{e}")
        print()
        print("按回车键退出...")
        try:
            input()
        except:
            time.sleep(3)

if __name__ == '__main__':
    main()
