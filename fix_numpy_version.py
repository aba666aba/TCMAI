#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复NumPy版本兼容性问题的脚本
此脚本会检查NumPy版本并在需要时降级到1.x版本
"""

import sys
import subprocess
import pkg_resources

def check_and_fix_numpy_version():
    """检查并修复NumPy版本兼容性问题"""
    try:
        # 获取当前NumPy版本
        numpy_version = pkg_resources.get_distribution("numpy").version
        print(f"当前NumPy版本: {numpy_version}")
        
        # 检查是否为2.x版本
        major_version = int(numpy_version.split('.')[0])
        if major_version >= 2:
            print("检测到NumPy 2.x版本，可能会与PyTorch Geometric产生兼容性问题")
            print("正在降级到NumPy 1.x版本...")
            
            # 降级NumPy到1.x版本
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "numpy<2"
            ])
            
            print("NumPy已成功降级到1.x版本")
            print("请重新启动程序以应用更改")
        else:
            print("NumPy版本兼容，无需更改")
            
    except Exception as e:
        print(f"检查或修复NumPy版本时出错: {e}")

if __name__ == "__main__":
    check_and_fix_numpy_version()