# Copyright (C) 2026 aba666aba
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# 本文件是中医药智能问答系统 (TCMAI) 的一部分。
#
# 本程序是自由软件：你可以依据自由软件基金会发布的 GNU Affero 通用公共
# 许可证条款（第 3 版或你选择的任何更新版本）重新分发和/或修改本程序。
#
# 本程序按“希望有用”的原则分发，但不提供任何担保，甚至不提供适销性或
# 特定用途适用性的默示担保。详见 GNU Affero 通用公共许可证。
#
# 你应已随本程序收到 GNU Affero 通用公共许可证副本。如未收到，
# 请见 <https://www.gnu.org/licenses/>。
#
# 注意：以网络服务方式提供本程序的服务端修改版本，同样必须按 AGPL-3.0
# 第 13 条向所有用户提供对应源代码。
# See LICENSE file for details

"""结果校验 — 溯源、合规提示、内容过滤"""

import logging
import re
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class ResultVerifier:
    """结果校验器 — AI 回答的溯源和合规处理"""

    def __init__(self):
        """初始化校验器"""
        # 合规提示模板
        self.disclaimer_templates = [
            "本回答仅供参考，不构成医疗诊断或治疗建议。",
            "如有健康问题，请咨询专业医疗机构或执业医师。",
            "中医药使用需辨证施治，个体差异较大，请谨慎参考。"
        ]

        # 敏感词列表（需要过滤或提示的内容）
        # 使用更精确的上下文匹配模式，只匹配绝对化断言语境
        self.sensitive_patterns = [
            r'(?:保证|确保|肯定|一定|绝对|百分百|百分之百).{0,5}(?:治愈|治好|痊愈|康复|有效)',
            r'(?:治愈|治好|痊愈|康复).{0,5}(?:保证|确保|肯定|一定|绝对|百分百|百分之百)',
            r'根治',
            r'无效退款',
            r'绝对.{0,3}(?:有效|安全|无副作用|无害)',
        ]

        # 需要溯源的关键词
        self.source_keywords = [
            '本草纲目',
            '黄帝内经',
            '伤寒论',
            '金匮要略',
        ]

    def verify(self, answer: str, knowledge: dict = None) -> dict[str, Any]:
        """
        校验回答

        Args:
            answer: AI 生成的回答
            knowledge: 相关知识信息

        Returns:
            校验后的结果
        """
        logger.info("开始校验回答")

        # 内容过滤
        filtered_answer = self._filter_content(answer)

        # 溯源处理
        sourced_answer = self._add_source(filtered_answer, knowledge)

        # 生成合规提示
        disclaimer = self._generate_disclaimer()

        # 提取相关知识
        extracted_knowledge = self._extract_knowledge(knowledge)

        result = {
            'answer': sourced_answer,
            'knowledge': extracted_knowledge,
            'disclaimer': disclaimer,
            'verified_at': datetime.now().isoformat(),
            'has_warning': self._check_warning(sourced_answer)
        }

        logger.info("回答校验完成")
        return result

    def _filter_content(self, content):
        filtered = content

        for pattern in self.sensitive_patterns:
            filtered = re.sub(
                pattern,
                "[该表述涉及绝对化断言，需谨慎参考]",
                filtered,
                flags=re.IGNORECASE,
            )

        return filtered

    def _add_source(self, answer, knowledge=None):
        if not knowledge:
            return answer

        sourced_answer = answer

        # 检查是否引用了典籍
        for keyword in self.source_keywords:
            if keyword in answer:
                sourced_answer += f"\n\n【典籍来源】{keyword}"
                break

        return sourced_answer

    def _generate_disclaimer(self) -> str:
        return " ".join(self.disclaimer_templates)

    def _extract_knowledge(self, knowledge=None):
        if not knowledge:
            return "暂无相关知识"

        parts = []

        # 提取药材信息
        if 'herbs' in knowledge and knowledge['herbs']:
            herb_names = [h.get('name', '') for h in knowledge['herbs'][:3]]
            parts.append(f"相关药材：{', '.join(herb_names)}")

        # 提取方剂信息
        if 'formulas' in knowledge and knowledge['formulas']:
            formula_names = [f.get('name', '') for f in knowledge['formulas'][:3]]
            parts.append(f"相关方剂：{', '.join(formula_names)}")

        return "; ".join(parts) if parts else "暂无相关知识"

    def _check_warning(self, content):
        warning_keywords = [
            '毒性',
            '禁忌',
            '不良反应',
            '慎用',
            '禁用',
        ]

        for keyword in warning_keywords:
            if keyword in content:
                return True

        return False

    def add_compliance_note(self, content: str) -> str:
        """
        添加合规标注

        Args:
            content: 原始内容

        Returns:
            添加合规标注后的内容
        """
        note = "\n\n---\n**合规提示**：以上内容基于中医药知识库生成，仅供参考学习。"
        return content + note

    def validate_answer_completeness(self, answer: str) -> dict[str, Any]:
        """
        验证回答完整性

        Args:
            answer: 回答内容

        Returns:
            验证结果
        """
        result = {
            'is_complete': True,
            'missing_parts': [],
            'suggestions': []
        }

        # 检查是否过短
        if len(answer) < 50:
            result['is_complete'] = False
            result['missing_parts'].append('回答内容过短')
            result['suggestions'].append('建议补充详细说明')

        # 检查是否有结论
        if not any(word in answer for word in ['建议', '可以', '应该', '推荐']):
            result['suggestions'].append('可添加具体建议')

        return result
