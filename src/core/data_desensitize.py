# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0
# See LICENSE file for details

"""数据脱敏 — 用户隐私保护、日志匿名化（含中医药白名单防误伤）"""

import hashlib
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# 中医药专用白名单
# 这些词虽符合"姓氏+1~2汉字"的姓名格式，但实际是中药/方剂名称，
# 脱敏姓名正则不应匹配它们。从 tcm_config 提取。
TCM_WHITELIST: set[str] = {
    # 药材（常见易误伤）
    "黄芪", "黄连", "黄柏", "黄芩", "黄精", "黄药子", "黄藤", "黄荆子",
    "白术", "白芍", "白芷", "白前", "白蔹", "白及", "白茅根", "白附子",
    "白芥子", "白鲜皮", "白扁豆", "白果", "白花蛇舌草",
    "陈皮", "陈仓米",
    "生姜", "干姜", "姜黄",
    "石斛", "石决明", "石膏", "石韦", "石菖蒲",
    "龙眼肉", "龙骨", "龙胆草", "龙齿",
    "何首乌",
    "吴茱萸", "吴梧桐",
    "朱砂",
    "苏木", "苏合香",
    "杜仲",
    "秦艽", "秦皮",
    "防风",
    "柴胡",
    "马勃", "马兜铃", "马齿苋", "马钱子",
    "刘寄奴",
    "红花", "红藤",
    "金银花",
    "连翘",
    # 方剂（常见）
    "四君子汤", "四物汤", "八珍汤", "六味地黄丸",
    "逍遥散", "小柴胡汤", "桂枝汤", "银翘散",
    "补中益气汤", "归脾汤", "生脉散",
    "龙胆泻肝汤", "藿香正气散",
    "温胆汤", "平胃散", "二陈汤",
    "麻黄汤", "麻杏石甘汤", "小青龙汤",
    "桑菊饮", "白虎汤", "清胃散",
    "天麻钩藤饮", "镇肝熄风汤",
    "血府逐瘀汤", "补阳还五汤",
    "肾气丸", "右归丸", "左归丸",
    "茵陈蒿汤", "青蒿鳖甲汤",
    "金匮肾气丸", "归芍地黄丸",
    "四逆散", "导赤散", "泻白散",
    "黄连解毒汤",
}
# 最长白名单长度（用于预检加速）
_MAX_WHITELIST_LEN = max(len(w) for w in TCM_WHITELIST) if TCM_WHITELIST else 0
# 编译正则：匹配以百家姓开头的白名单词条
_TCM_PROTECT_PATTERN = re.compile(
    "|".join(re.escape(w) for w in sorted(TCM_WHITELIST, key=len, reverse=True))
)


class DataDesensitizer:
    """
    数据脱敏器

    负责用户数据的脱敏处理，保护用户隐私
    """

    def __init__(self):
        """初始化脱敏器"""
        # 个人信息匹配规则
        # 常见百家姓（前 100 个）
        _surnames = (
            r"[张王李赵刘陈杨黄吴周徐孙朱马胡郭何高林罗郑梁谢宋唐冯韩曹彭曾萧"
            r"田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦"
            r"傅方白邹孟熊秦邱侯江尹薛阎段雷贺龙史陶毛常康武乔汤严顾钱覃]"
        )
        self.patterns = {
            "phone": r"1[3-9]\d{9}",
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "id_card": r"(?<!\d)(?:\d{17}[\dXx]|\d{15})(?!\d)",
            "address": r"(?:省|市|区|县|镇|村|路|街|道|号|栋|单元|室)",
            "name": _surnames + r"[一-鿿]{1,2}",
        }

        # 脱敏替换符
        self.replacement = {
            'phone': '138****8888',
            'email': 'u***@***.com',
            'id_card': '************',
            'address': '**省**市**区',
            'name': '张*',
        }

    def desensitize(self, data: dict[str, Any]) -> dict[str, Any]:
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

        # 身份证号优先脱敏（18 位数字更特异，避免被手机号正则误匹配）
        desensitized = re.sub(
            self.patterns["id_card"],
            self.replacement["id_card"],
            desensitized,
        )

        # 手机号脱敏
        desensitized = re.sub(
            self.patterns["phone"],
            self.replacement["phone"],
            desensitized,
        )

        # 邮箱脱敏
        desensitized = re.sub(
            self.patterns["email"],
            self.replacement["email"],
            desensitized,
        )

        # 姓名脱敏（匹配"姓氏+1~2个汉字"的姓名）
        # 先保护中医药白名单词条（如"黄芪"→不脱敏）
        protected_spans = []
        for m in _TCM_PROTECT_PATTERN.finditer(desensitized):
            protected_spans.append((m.start(), m.end()))

        # 用自定义替换函数检查每个匹配是否在白名单范围内
        def _replace_name(match):
            start, end = match.start(), match.end()
            # 检查是否落在受保护的白名单区间内
            for ps, pe in protected_spans:
                if start >= ps and end <= pe:
                    return match.group(0)  # 不脱敏
            # 正常脱敏：姓氏保留，其余变*
            token = match.group(0)
            return token[0] + "*" * (len(token) - 1)

        desensitized = re.sub(self.patterns["name"], _replace_name, desensitized)

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

    def sanitize_log(self, log_data: dict[str, Any]) -> dict[str, Any]:
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

    def get_desensitize_stats(self, original: str, desensitized: str) -> dict[str, int]:
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
