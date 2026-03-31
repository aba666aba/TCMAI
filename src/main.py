# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0
# See LICENSE file for details

"""
中医药智能问答系统主入口

使用 Flask 作为 Web 框架，统一入口点

注意：本系统设计为可移动应用，所有路径均相对于 TCMAI 目录
"""

import logging
import sys
import os

# 获取 TCMAI 根目录（可移动应用的核心目录）
TCMAI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 添加 src 目录到 Python 路径
sys.path.insert(0, os.path.join(TCMAI_ROOT, 'src'))

# 设置环境变量，供其他模块使用
os.environ['TCMAI_ROOT'] = TCMAI_ROOT

from web.app import create_app

# 配置日志输出，让用户能看到启动进度
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    print("=" * 50)
    print("  中医药智能问答系统")
    print("  版本：V1.0")
    print("=" * 50)
    print()
    print(f"[信息] 运行目录：{TCMAI_ROOT}")
    print("[启动] 正在初始化系统...")
    print("[提示] 首次启动需要加载 AI 模型，可能需要 20-40 秒")
    print()
    
    try:
        print("[1/3] 正在创建 Flask 应用...")
        app = create_app()
        print("      [OK] Flask 应用创建完成")
        
        print("[2/3] 正在加载 AI 模型和服务...")
        print("      - 加载 MiniMind 大模型 (1.45 亿参数)")
        print("      - 初始化中医药知识库")
        print("      - 启动症状分析引擎")
        print("      这可能需要一些时间，请耐心等待...")
        print()
        
        print("[3/3] 正在启动 Web 服务器...")
        print()
        print("-" * 50)
        print("  服务器地址：http://localhost:5000")
        print("  按 Ctrl+C 可停止服务器")
        print("-" * 50)
        print()
        print("[OK] 系统启动完成！")
        print()
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except Exception as e:
        print()
        print("[错误] 系统启动失败：" + str(e))
        print()
        print("可能的原因:")
        print("  1. 模型文件不存在，请检查 models/ 目录")
        print("  2. 端口 5000 被占用")
        print("  3. 依赖包未安装，请运行：pip install -r requirements.txt")
        print()
        logger.error(f"启动异常：{e}", exc_info=True)
        input("按回车键退出...")
