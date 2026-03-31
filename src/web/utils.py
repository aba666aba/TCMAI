"""Web 工具模块"""
import json
import csv
import io
from typing import Dict, Any


def export_result(data: Dict[str, Any], format_type: str = 'json') -> str:
    """导出查询结果
    
    Args:
        data: 要导出的数据
        format_type: 导出格式，支持 'json' 和 'csv'
    
    Returns:
        导出的内容字符串
    """
    if format_type == 'csv':
        output = io.StringIO()
        writer = csv.writer(output)
        # 实现 CSV 导出逻辑
        for key, value in data.items():
            writer.writerow([key, str(value)])
        return output.getvalue()
    else:
        return json.dumps(data, ensure_ascii=False, indent=2)
