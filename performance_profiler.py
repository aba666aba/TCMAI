#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
性能分析脚本
用于诊断中药知识图谱训练过程中的性能问题
"""

import torch
import time
import psutil
import os
from typing import Dict, Any

def get_system_info():
    """获取系统信息"""
    # CPU信息
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # 内存信息
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_used_gb = memory.used / (1024**3)
    memory_total_gb = memory.total / (1024**3)
    
    # GPU信息（如果可用）
    gpu_info = "N/A"
    if torch.cuda.is_available():
        gpu_info = f"{torch.cuda.get_device_name()}"
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_info += f" (已用: {gpu_memory:.2f}GB / 总计: {gpu_memory_total:.2f}GB)"
    
    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
        "memory_used_gb": memory_used_gb,
        "memory_total_gb": memory_total_gb,
        "gpu_info": gpu_info
    }

def log_system_status(prefix: str = ""):
    """记录系统状态"""
    info = get_system_info()
    print(f"{prefix}系统状态:")
    print(f"  CPU使用率: {info['cpu_percent']:.1f}%")
    print(f"  内存使用: {info['memory_used_gb']:.2f}GB / {info['memory_total_gb']:.2f}GB ({info['memory_percent']:.1f}%)")
    print(f"  GPU信息: {info['gpu_info']}")

def profile_function(func, *args, **kwargs):
    """性能分析函数"""
    print(f"开始执行 {func.__name__}...")
    log_system_status("执行前 ")
    
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    log_system_status("执行后 ")
    print(f"{func.__name__} 执行完成，耗时: {end_time - start_time:.2f}秒")
    print("-" * 50)
    
    return result

# 示例使用
if __name__ == "__main__":
    # 测试系统信息获取
    log_system_status("初始 ")
    
    # 创建一个简单的测试函数
    def test_function():
        # 模拟一些计算
        x = torch.randn(1000, 1000)
        y = torch.randn(1000, 1000)
        z = torch.matmul(x, y)
        return z
    
    # 性能分析
    result = profile_function(test_function)