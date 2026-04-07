# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0
# See LICENSE file for details

"""
数据脱敏模块

原创功能：用户数据脱敏、隐私保护、日志匿名化
"""

import logging
import re
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DataDesensitizer:
    """
    数据脱敏器
    
    负责用户数据的脱敏处理，保护用户隐私
    """
    
    def __init__(self):
        """初始化脱敏器"""
        # 个人信息匹配规则
        self.patterns = {
            'phone': r'1[3-9]\d{9}',
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            # 使用非捕获组优化，添加词边界避免匹配更长数字串中的片段
            'id_card': r'\b(?:\d{17}[\dXx]|\d{15})\b',
            'address': r'(?:省|市|区|县|镇|村|路|街|道|号|栋|单元|室)',
            'name': r'[张王李赵刘陈杨黄吴周徐孙朱马胡郭何高林罗郑梁谢宋唐冯韩曹彭曾萧田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦傅方白邹孟熊秦邱侯江尹薛阎段雷贺龙史陶毛邱常康武乔汤严顾钱覃]',
        }
        
        # 脱敏替换符
        self.replacement = {
            'phone': '138****8888',
            'email': 'u***@***.com',
            'id_card': '************',
            'address': '**省**市**区',
            'name': '张*',
        }
    
    def desensitize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        对数据进行脱敏
        
        Args:
            data: 原始数据
            
        Returns:
            脱敏后的数据
        """
        logger.info("开始数据脱敏")
        
        desensitized = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                desensitized[key] = self._desensitize_text(value)
            elif isinstance(value, dict):
                desensitized[key] = self.desensitize(value)
            elif isinstance(value, list):
                desensitized[key] = [
                    self._desensitize_text(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                desensitized[key] = value
        
        logger.info("数据脱敏完成")
        return desensitized
    
    def _desensitize_text(self, text: str) -> str:
        """
        对文本进行脱敏
        
        Args:
            text: 原始文本
            
        Returns:
            脱敏后的文本
        """
        if not text:
            return text
        
        desensitized = text
        
        # 手机号脱敏
        desensitized = re.sub(
            self.patterns['phone'],
            self.replacement['phone'],
            desensitized
        )
        
        # 邮箱脱敏
        desensitized = re.sub(
            self.patterns['email'],
            self.replacement['email'],
            desensitized
        )
        
        # 身份证号脱敏
        desensitized = re.sub(
            self.patterns['id_card'],
            self.replacement['id_card'],
            desensitized
        )
        
        return desensitized
    
    def anonymize_query(self, query: str, user_id: str = None) -> str:
        """
        匿名化查询记录
        
        Args:
            query: 查询内容
            user_id: 用户 ID
            
        Returns:
            匿名化后的查询记录
        """
        # 脱敏查询内容
        anonymized_query = self._desensitize_text(query)
        
        # 生成匿名标识
        if user_id:
            anon_id = self._generate_anon_id(user_id)
        else:
            anon_id = "anonymous"
        
        logger.info(f"查询已匿名化：{anon_id}")
        return anonymized_query
    
    def _generate_anon_id(self, user_id: str) -> str:
        """
        生成匿名 ID
        
        Args:
            user_id: 原始用户 ID
            
        Returns:
            匿名 ID
        """
        # 使用哈希生成匿名 ID
        hash_obj = hashlib.sha256(user_id.encode('utf-8'))
        return f"user_{hash_obj.hexdigest()[:8]}"
    
    def sanitize_log(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        日志数据脱敏
        
        Args:
            log_data: 原始日志数据
            
        Returns:
            脱敏后的日志数据
        """
        # 移除敏感字段
        sensitive_fields = ['password', 'token', 'cookie', 'session_id']
        
        sanitized = {}
        for key, value in log_data.items():
            if key.lower() in sensitive_fields:
                sanitized[key] = '***REDACTED***'
            elif isinstance(value, str):
                sanitized[key] = self._desensitize_text(value)
            else:
                sanitized[key] = value
        
        return sanitized
    
    def remove_pii(self, text: str) -> str:
        """
        移除个人身份信息 (PII)
        
        Args:
            text: 包含 PII 的文本
            
        Returns:
            移除 PII 后的文本
        """
        cleaned = text
        
        # 移除所有匹配的个人信息
        for field, pattern in self.patterns.items():
            if field != 'name':  # 姓名只做脱敏，不移除
                cleaned = re.sub(pattern, f'[{field}已移除]', cleaned)
        
        return cleaned
    
    def get_desensitize_stats(self, original: str, desensitized: str) -> Dict[str, int]:
        """
        获取脱敏统计信息
        
        Args:
            original: 原始文本
            desensitized: 脱敏后文本
            
        Returns:
            统计信息
        """
        stats = {
            'phone_count': 0,
            'email_count': 0,
            'id_card_count': 0,
        }
        
        # 统计各类信息数量
        stats['phone_count'] = len(re.findall(self.patterns['phone'], original))
        stats['email_count'] = len(re.findall(self.patterns['email'], original))
        stats['id_card_count'] = len(re.findall(self.patterns['id_card'], original))
        
        return stats


# 全局脱敏器实例
_global_desensitizer = None


def get_desensitizer() -> DataDesensitizer:
    """获取全局脱敏器实例"""
    global _global_desensitizer
    if _global_desensitizer is None:
        _global_desensitizer = DataDesensitizer()
    return _global_desensitizer


def desensitize_query(query: str) -> str:
    """
    便捷函数：脱敏查询
    
    Args:
        query: 查询内容
        
    Returns:
        脱敏后的查询
    """
    return get_desensitizer().anonymize_query(query)
