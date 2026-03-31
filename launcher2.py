# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0

"""
中医药智能问答系统 - 启动器
"""

import sys
import os
import traceback

print("="*60)
print("  中医药智能问答系统 - 启动中...")
print("="*60)
print()

# 获取运行目录
if getattr(sys, 'frozen', False):
    ROOT_DIR = os.path.dirname(sys.executable)
    print(f"运行目录：{ROOT_DIR}")
    
    # 添加内部路径
    internal = os.path.join(ROOT_DIR, '_internal')
    src = os.path.join(internal, 'src')
    
    print(f"添加路径：{internal}")
    print(f"添加路径：{src}")
    
    sys.path.insert(0, internal)
    sys.path.insert(0, src)
    os.environ['TCMAI_ROOT'] = ROOT_DIR
else:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(ROOT_DIR, 'src')
    sys.path.insert(0, src)
    os.environ['TCMAI_ROOT'] = ROOT_DIR
    print(f"开发模式：{ROOT_DIR}")

print()
print("[1/3] 导入模块...")

try:
    print(f"  sys.path: {len(sys.path)} 路径")
    print(f"  尝试导入 web.app...")
    
    from web.app import create_app
    
    print("  ✓ 导入成功")
    
except Exception as e:
    print(f"  ✗ 导入失败：{e}")
    print()
    print("错误详情:")
    traceback.print_exc()
    print()
    input("按回车键退出...")
    sys.exit(1)

print()
print("[2/3] 创建应用...")

try:
    app = create_app()
    print("  ✓ 应用创建成功")
except Exception as e:
    print(f"  ✗ 创建失败：{e}")
    traceback.print_exc()
    input("按回车键退出...")
    sys.exit(1)

print()
print("[3/3] 启动服务器...")
print()
print("-"*60)
print("  访问地址：http://localhost:5000")
print("  按 Ctrl+C 停止服务器")
print("-"*60)
print()

try:
    print("服务器运行中...")
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
except Exception as e:
    print(f"错误：{e}")
    traceback.print_exc()
    input("按回车键退出...")
