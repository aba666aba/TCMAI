# -*- coding: utf-8 -*-
# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0

"""
中医药推荐系统 - 推理接口

根据症状推荐药材和方剂，并进行安全校验。
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# 项目路径
if getattr(sys, 'frozen', False):
    PROJECT_ROOT = Path(sys.executable).parent.parent
else:
    PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tcm_recommender.model import TCMNet, create_model
from tcm_recommender.train import build_herb_features
from tcm_recommender.data_pipeline import TCMDataPipeline
from tcm_recommender.extended_config import EXTENDED_SYMPTOM_LIST, EXTENDED_SYNDROME_DEFINITIONS

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TCMRecommender:
    """中医药推荐器"""

    def __init__(
        self,
        model_path: str,
        db_path: str,
        encoder_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.db_path = db_path
        self.encoder_path = encoder_path

        # 加载数据管线
        logger.info("加载数据...")
        self._pipeline = TCMDataPipeline(db_path)
        self._pipeline.load_all()

        # 加载模型
        logger.info("加载模型: %s", model_path)
        checkpoint = torch.load(model_path, map_location="cpu")
        cfg = checkpoint.get("config", {})

        herb_feat = build_herb_features(self._pipeline)
        self._herb_feat = herb_feat

        # 症状/证型配置
        self._symptom_list = EXTENDED_SYMPTOM_LIST
        self._syndrome_names = ["未知"] + list(EXTENDED_SYNDROME_DEFINITIONS.keys())

        # 创建模型
        self.model = create_model(
            num_symptoms=len(self._symptom_list),
            num_herbs=len(self._pipeline.herb_names),
            num_formulas=len(self._pipeline.formula_names) + 1,
            num_syndromes=len(self._syndrome_names),
            num_chars=herb_feat["num_chars"],
            num_categories=herb_feat["num_categories"],
            d_model=cfg.get("d_model", 384),
            nhead=8,
            num_layers=cfg.get("num_layers", 4),
            dropout=0.1,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        logger.info("模型加载完成")

        # 症状名称→ID 映射
        self.symptom_to_id = {s: i + 1 for i, s in enumerate(self._symptom_list)}

        # 药材索引映射
        self._herb_name_list = self._pipeline.herb_names
        self._formula_name_list = self._pipeline.formula_names

        # 缓存药材特征
        self._cached_herb_features = herb_feat

        logger.info("推荐器初始化完成")

    def parse_symptoms(self, symptom_text: str) -> Tuple[List[int], List[str], List[str]]:
        """解析症状文本"""
        for sep in [' ', ',', '，', '、', ';', '；']:
            symptom_text = symptom_text.replace(sep, ' ')

        symptoms = [s.strip() for s in symptom_text.split() if s.strip()]
        symptom_ids = []
        valid_symptoms = []
        unknown_symptoms = []

        for symp in symptoms:
            if symp in self.symptom_to_id:
                symptom_ids.append(self.symptom_to_id[symp])
                valid_symptoms.append(symp)
            else:
                # 尝试别名匹配
                matched = False
                for std_name, sid in self.symptom_to_id.items():
                    if symp in std_name or std_name in symp:
                        symptom_ids.append(sid)
                        valid_symptoms.append(std_name)
                        matched = True
                        break
                if not matched:
                    unknown_symptoms.append(symp)

        if unknown_symptoms:
            logger.warning("未知症状：%s", unknown_symptoms)
        if symptoms and not symptom_ids:
            logger.warning("所有症状都无效，输入：%s", symptoms)

        return symptom_ids, valid_symptoms, unknown_symptoms

    def recommend(
        self, symptoms: List[int], top_k_herb: int = 10, top_k_formula: int = 5
    ) -> Dict:
        """推荐药材和方剂"""
        if not symptoms:
            return {
                "error": "无有效症状",
                "herbs": [], "formulas": [],
                "suggestion": "请提供有效症状描述",
            }

        symptom_tensor = torch.tensor([symptoms], dtype=torch.long).to(self.device)
        mask = torch.ones_like(symptom_tensor, dtype=torch.float32)

        with torch.no_grad():
            result = self.model.predict(
                symptom_tensor, mask,
                self._herb_feat["char_multihot"].to(self.device),
                self._herb_feat["category_idx"].to(self.device),
                self._herb_feat["nature_idx"].to(self.device),
                self._herb_feat["flavor_idx"].to(self.device),
                top_k=max(top_k_herb, top_k_formula),
            )

        # 药材结果
        herb_results = []
        for herb_idx, score in result["herbs"][:top_k_herb]:
            name = self._herb_name_list[herb_idx] if herb_idx < len(self._herb_name_list) else f"药材{herb_idx}"
            herb_results.append({
                "id": int(herb_idx),
                "name": name,
                "score": float(score),
                "efficacy": "",
                "is_toxic": False,
            })

        # 方剂结果
        formula_results = []
        for formula_idx, score in result["formulas"][:top_k_formula]:
            name = self._formula_name_list[formula_idx] if formula_idx < len(self._formula_name_list) else f"方剂{formula_idx}"
            formula_results.append({
                "id": int(formula_idx),
                "name": name,
                "score": float(score),
                "composition": "",
                "functions": "",
            })

        # 证型
        top_syndrome = None
        if result.get("syndromes"):
            sid, sp = result["syndromes"][0]
            if 0 <= sid < len(self._syndrome_names) and sid > 0:
                top_syndrome = {
                    "id": int(sid),
                    "name": self._syndrome_names[sid],
                    "confidence": round(float(sp), 4),
                }

        return {
            "symptoms": [self._symptom_list[s - 1] for s in symptoms if 0 < s <= len(self._symptom_list)],
            "syndrome": top_syndrome,
            "herbs": herb_results,
            "formulas": formula_results,
        }

    def check_safety(self, herb_names: List[str]) -> Dict:
        """安全检查（十八反/十九畏/孕妇禁用）"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        warnings = []
        for h in herb_names:
            cur.execute("SELECT contraindications FROM herbs WHERE name = ?", (h,))
            row = cur.fetchone()
            if row and row[0]:
                warnings.append("%s: %s" % (h, row[0][:100]))

        conn.close()
        return {"warnings": warnings}

    def format_output(self, result: Dict) -> str:
        """格式化输出"""
        if 'error' in result:
            return "错误：%s" % result['error']

        lines = []
        lines.append("=" * 60)
        lines.append("  中医药智能推荐结果")
        lines.append("=" * 60)
        lines.append("\n症状：%s" % ", ".join(result['symptoms']))

        if result.get('syndrome'):
            s = result['syndrome']
            lines.append("\n证型：%s (置信度: %.4f)" % (s['name'], s['confidence']))

        lines.append("\n推荐药材:")
        for i, h in enumerate(result['herbs'], 1):
            lines.append("  %d. %s (置信度: %.4f)" % (i, h['name'], h['score']))

        lines.append("\n推荐方剂:")
        for i, f in enumerate(result['formulas'], 1):
            lines.append("  %d. %s (置信度: %.4f)" % (i, f['name'], f['score']))

        lines.append("\n安全校验:")
        herb_names = [h['name'] for h in result['herbs'][:10]]
        safety = self.check_safety(herb_names)
        if safety['warnings']:
            for w in safety['warnings']:
                lines.append("  %s" % w)
        else:
            lines.append("  未发现明显配伍禁忌")

        lines.append("\n" + "=" * 60)
        lines.append("注：本结果仅供参考，请由专业中医师判断使用")
        lines.append("=" * 60)
        return "\n".join(lines)

    def close(self):
        """释放资源"""
        if hasattr(self, '_pipeline'):
            self._pipeline.close()


def main():
    parser = argparse.ArgumentParser(description='中医药智能推荐系统')
    parser.add_argument('symptoms', type=str, help='症状列表（空格/逗号分隔）')
    parser.add_argument('--model', type=str, default='tcm_recommender/best_model.pth', help='模型文件路径')
    parser.add_argument('--db', type=str, default='data/tcm_knowledge.db', help='数据库路径')
    parser.add_argument('--top-herb', type=int, default=10, help='返回药材数')
    parser.add_argument('--top-formula', type=int, default=5, help='返回方剂数')
    parser.add_argument('--device', type=str, default='cpu', help='运行设备')
    parser.add_argument('--json', action='store_true', help='JSON 格式输出')

    args = parser.parse_args()
    model_path = Path(PROJECT_ROOT) / args.model
    db_path = Path(PROJECT_ROOT) / args.db

    if not model_path.exists():
        logger.error("模型文件不存在：%s", model_path)
        return
    if not db_path.exists():
        logger.error("数据库不存在：%s", db_path)
        return

    r = TCMRecommender(str(model_path), str(db_path), device=args.device)
    sids, _, _ = r.parse_symptoms(args.symptoms)
    if not sids:
        print("无有效症状")
        return

    result = r.recommend(sids, top_k_herb=args.top_herb, top_k_formula=args.top_formula)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(r.format_output(result))
    r.close()


if __name__ == '__main__':
    main()
