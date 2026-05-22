# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0

"""
中医药推荐系统 - 推理脚本

根据症状推荐药材和方剂，并进行安全校验
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# 添加项目路径
# 支持正常运行和 PyInstaller 打包后的情况
if getattr(sys, 'frozen', False):
    # PyInstaller 打包后的情况
    PROJECT_ROOT = Path(sys.executable).parent.parent
else:
    # 正常 Python 运行
    PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tcm_recommender.data_loader import TCMDataLoader
from tcm_recommender.model import create_model, SafetyChecker

# V3 模型支持（可选）
try:
    from tcm_recommender.model_v3 import (
        create_model_v3,
        HerbFeatureEncoderV3,
        SafetyChecker as SafetyCheckerV3,
    )
    _V3_AVAILABLE = True
except ImportError:
    _V3_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TCMRecommender:
    """中医药推荐器（支持 V2/V3 模型自动检测）"""

    def __init__(
        self,
        model_path: str,
        db_path: str,
        encoder_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.db_path = db_path

        logger.info("加载数据...")
        self.data_loader = TCMDataLoader(db_path)
        self.data_loader.load_all_data()

        # 检测模型版本
        checkpoint = torch.load(model_path, map_location="cpu")
        self._model_version = self._detect_version(checkpoint)
        logger.info("检测到模型版本: V%s", self._model_version)

        # 药材字符特征
        if self._model_version == 3:
            from tcm_recommender.train_v3 import build_herb_char_features_v3
            herb_char_multihot, num_chars, char_list = build_herb_char_features_v3(self.data_loader)
        else:
            from tcm_recommender.train import build_herb_char_features
            herb_char_multihot, num_chars, char_list = build_herb_char_features(self.data_loader)
        self.herb_char_multihot = herb_char_multihot

        # 加载扩展症状/证型（V2.1/V3 需要）
        self._extended_symptoms = None
        self._extended_syndromes = None
        if self._model_version in (3, 21):
            try:
                from tcm_recommender.extended_config import EXTENDED_SYMPTOM_LIST, EXTENDED_SYNDROME_DEFINITIONS
                self._extended_symptoms = EXTENDED_SYMPTOM_LIST
                self._extended_syndromes = EXTENDED_SYNDROME_DEFINITIONS
            except ImportError:
                pass

        # 特征编码器
        if self._model_version == 3:
            self.herb_feature_encoder = HerbFeatureEncoderV3(num_chars, feature_dim=128)
        elif self._model_version == 21:
            from tcm_recommender.model import HerbFeatureEncoder
            self.herb_feature_encoder = HerbFeatureEncoder(num_chars, feature_dim=64)
        else:
            from tcm_recommender.model import HerbFeatureEncoder
            self.herb_feature_encoder = HerbFeatureEncoder(num_chars, feature_dim=32)

        # 模型
        logger.info("加载模型...")
        num_symptoms = len(self._extended_symptoms) if self._extended_symptoms else len(self.data_loader.symptoms)
        num_syndromes = len(self._extended_syndromes) if self._extended_syndromes else self.data_loader.num_syndromes

        if self._model_version == 3:
            self.model = create_model_v3(
                num_symptoms=num_symptoms,
                num_herbs=self.data_loader.num_herbs,
                num_formulas=len(self.data_loader.formulas) + 1,
                num_syndromes=num_syndromes + 1,
                herb_char_dim=num_chars,
                d_model=256,
                herb_feature_dim=128,
                num_transformer_layers=4,
            )
        elif self._model_version == 21:
            self.model = create_model(
                num_symptoms=num_symptoms,
                num_herbs=self.data_loader.num_herbs,
                num_formulas=len(self.data_loader.formulas) + 1,
                num_syndromes=num_syndromes + 1,
                herb_char_dim=num_chars,
                embed_dim=128,
                herb_feature_dim=64,
            )
        else:
            self.model = create_model(
                num_symptoms=num_symptoms,
                num_herbs=self.data_loader.num_herbs,
                num_formulas=len(self.data_loader.formulas) + 1,
                num_syndromes=num_syndromes + 1,
                herb_char_dim=num_chars,
                embed_dim=64,
                herb_feature_dim=32,
            )

        self._load_model(checkpoint, encoder_path)
        self.model.to(device)
        self.herb_feature_encoder.to(device)
        self.herb_char_multihot = self.herb_char_multihot.to(device)
        self.model.eval()
        self.herb_feature_encoder.eval()

        # 预计算药材特征
        with torch.no_grad():
            self._cached_herb_features = self.herb_feature_encoder(
                self.herb_char_multihot
            )

        # 安全校验器
        self.safety_checker = SafetyChecker(self.data_loader.incompatible_herbs)

        # 症状名称→ID 映射
        symptom_list = self._extended_symptoms if self._extended_symptoms else [
            v["name"] for v in sorted(self.data_loader.symptoms.values(), key=lambda x: x["id"])
        ]
        self.symptom_to_id = {s: i + 1 for i, s in enumerate(symptom_list)}

        # 证型名称
        if self._extended_syndromes:
            self.syndrome_names = ["未知"] + list(self._extended_syndromes.keys())
        else:
            from core.tcm_config import SYNDROME_RULES
            self.syndrome_names = ["未知"] + list(SYNDROME_RULES.keys())

        logger.info("推荐器初始化完成 (V%s)", self._model_version)

    @staticmethod
    def _detect_version(checkpoint: dict) -> int:
        """检测模型版本"""
        config = checkpoint.get("model_config", {})
        d_model = config.get("d_model", 64)
        version_str = config.get("version", "")
        if version_str == "v2.1" or (d_model >= 100 and d_model < 200):
            return 21  # V2.1
        elif d_model >= 200:
            return 3
        return 2

    def _load_model(self, checkpoint: dict, encoder_path: Optional[str] = None):
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("模型状态已加载")

        # 加载编码器
        if encoder_path and os.path.exists(encoder_path):
            enc_checkpoint = torch.load(encoder_path, map_location=self.device)
            self.herb_feature_encoder.load_state_dict(
                enc_checkpoint["encoder_state_dict"]
            )
            logger.info("编码器已加载：%s", encoder_path)
    
    def parse_symptoms(self, symptom_text: str) -> Tuple[List[int], List[str], List[str]]:
        """
        解析症状文本
        
        Args:
            symptom_text: 症状文本（空格或逗号分隔）
        
        Returns:
            三元组：(症状 ID 列表, 有效症状名称列表, 未识别症状列表)
        """
        # 分割症状
        separators = [' ', ',', '，', '、', ';', '；']
        for sep in separators:
            symptom_text = symptom_text.replace(sep, ' ')
        
        symptoms = [s.strip() for s in symptom_text.split() if s.strip()]
        
        # 转换为 ID
        symptom_ids = []
        valid_symptoms = []
        unknown_symptoms = []
        for symp in symptoms:
            if symp in self.symptom_to_id:
                symptom_ids.append(self.symptom_to_id[symp])
                valid_symptoms.append(symp)
            else:
                unknown_symptoms.append(symp)
        
        if unknown_symptoms:
            logger.warning(f"未知症状：{unknown_symptoms}")
        
        # 如果所有症状都无效，记录警告日志
        if symptoms and not symptom_ids:
            logger.warning(f"所有症状都无效，输入症状：{symptoms}")
            logger.info(f"支持的症状示例：{list(self.symptom_to_id.keys())[:10]}...")
        
        return symptom_ids, valid_symptoms, unknown_symptoms
    
    def recommend(
        self,
        symptoms: List[int],
        top_k_herb: int = 10,
        top_k_formula: int = 5,
    ) -> Dict:
        """推荐药材和方剂"""
        if not symptoms:
            return {
                "error": "无有效症状",
                "herbs": [],
                "formulas": [],
                "suggestion": f"支持的症状示例：{', '.join(list(self.symptom_to_id.keys())[:10])}...",
            }

        symptom_tensor = torch.tensor([symptoms], dtype=torch.long).to(self.device)

        with torch.no_grad():
            result = self.model.predict(
                symptom_tensor, self._cached_herb_features,
                top_k=max(top_k_herb, top_k_formula),
            )

        herb_names = self.data_loader.get_herb_names()
        formula_names = self.data_loader.get_formula_names()
        herbs_data = self.data_loader.herbs
        formulas_data = self.data_loader.formulas

        # 药材
        herb_results = []
        for herb_idx, score in result["herbs"][:top_k_herb]:
            herb_id = self.data_loader.idx_to_herb_id.get(herb_idx, herb_idx)
            herb = herbs_data.get(herb_id, {})
            herb_results.append({
                "id": herb_idx,
                "name": herb.get("name", "Unknown"),
                "score": float(score),
                "efficacy": herb.get("efficacy", ""),
                "is_toxic": herb.get("is_toxic", False),
            })

        # 方剂
        formula_results = []
        for formula_idx, score in result["formulas"][:top_k_formula]:
            formula_id = self.data_loader.idx_to_formula_id.get(
                formula_idx, formula_idx
            )
            formula = formulas_data.get(formula_id, {})
            formula_results.append({
                "id": formula_idx,
                "name": formula.get("name", "Unknown"),
                "score": float(score),
                "composition": formula.get("composition", ""),
                "functions": formula.get("functions", ""),
            })

        # 证型
        top_syndrome = None
        if result.get("syndromes"):
            sid, sp = result["syndromes"][0]
            if 0 <= sid < len(self.syndrome_names) and sid > 0:
                top_syndrome = {
                    "id": sid,
                    "name": self.syndrome_names[sid],
                    "confidence": round(float(sp), 4),
                }

        return {
            "symptoms": [
                self.data_loader.symptoms.get(s, {}).get("name", "")
                for s in symptoms
            ],
            "syndrome": top_syndrome,
            "herbs": herb_results,
            "formulas": formula_results,
        }
    
    def check_safety(self, herb_names: List[str]) -> Dict:
        """
        安全检查
        
        Args:
            herb_names: 药材名称列表
        
        Returns:
            安全检查结果
        """
        return self.safety_checker.check_compatibility(herb_names)
    
    def format_output(self, result: Dict) -> str:
        """
        格式化输出
        
        Args:
            result: 推荐结果
        
        Returns:
            格式化的文本输出
        """
        if 'error' in result:
            return f"❌ 错误：{result['error']}"
        
        lines = []
        lines.append("=" * 60)
        lines.append("  中医药智能推荐结果")
        lines.append("=" * 60)
        
        # 症状
        lines.append(f"\n📋 症状：{', '.join(result['symptoms'])}")
        
        # 药材推荐
        lines.append("\n🌿 推荐药材:")
        for i, herb in enumerate(result['herbs'], 1):
            toxic_mark = " ☠️" if herb['is_toxic'] else ""
            lines.append(
                f"  {i}. {herb['name']}{toxic_mark} (置信度：{herb['score']:.4f})"
            )
            lines.append(f"     功效：{herb['efficacy'][:50]}...")
        
        # 方剂推荐
        lines.append("\n📜 推荐方剂:")
        for i, formula in enumerate(result['formulas'], 1):
            lines.append(
                f"  {i}. {formula['name']} (置信度：{formula['score']:.4f})"
            )
            lines.append(f"     组成：{formula['composition']}")
            lines.append(f"     功效：{formula['functions']}")
        
        # 安全检查
        lines.append("\n⚠️ 安全校验:")
        herb_names = [h['name'] for h in result['herbs'][:10]]
        safety_result = self.check_safety(herb_names)
        
        if safety_result['warnings']:
            for w in safety_result['warnings']:
                lines.append(f"  {w}")
        else:
            lines.append("  ✓ 未发现配伍禁忌")
        
        lines.append("\n" + "=" * 60)
        lines.append("注：本结果仅供参考，请由专业中医师判断使用")
        lines.append("=" * 60)
        
        return '\n'.join(lines)
    
    def close(self):
        """关闭数据加载器"""
        self.data_loader.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='中医药智能推荐系统')
    parser.add_argument(
        'symptoms',
        type=str,
        help='症状列表（空格或逗号分隔），如："乏力 食欲不振 失眠"'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='tcm_recommender/best_model.pth',
        help='模型文件路径'
    )
    parser.add_argument(
        '--db',
        type=str,
        default='data/tcm_knowledge.db',
        help='数据库路径'
    )
    parser.add_argument(
        '--top-herb',
        type=int,
        default=10,
        help='返回药材数量'
    )
    parser.add_argument(
        '--top-formula',
        type=int,
        default=5,
        help='返回方剂数量'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='运行设备 (cpu/cuda)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='以 JSON 格式输出'
    )
    
    args = parser.parse_args()
    
    # 路径
    model_path = Path(PROJECT_ROOT) / args.model
    db_path = Path(PROJECT_ROOT) / args.db
    
    # 检查文件
    if not model_path.exists():
        logger.error(f"模型文件不存在：{model_path}")
        logger.error("请先运行训练：python tcm_recommender/train.py")
        return
    
    if not db_path.exists():
        logger.error(f"数据库不存在：{db_path}")
        return
    
    # 创建推荐器
    recommender = TCMRecommender(
        model_path=str(model_path),
        db_path=str(db_path),
        device=args.device
    )
    
    # 解析症状
    symptom_ids, valid_symptoms, unknown_symptoms = recommender.parse_symptoms(args.symptoms)
    
    if not symptom_ids:
        print("❌ 无有效症状，请检查输入")
        if unknown_symptoms:
            print(f"未识别的症状：{unknown_symptoms}")
        print(f"支持的症状示例：{list(recommender.symptom_to_id.keys())[:10]}...")
        return
    
    # 推荐
    result = recommender.recommend(
        symptoms=symptom_ids,
        top_k_herb=args.top_herb,
        top_k_formula=args.top_formula
    )
    
    # 输出
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(recommender.format_output(result))
    
    recommender.close()


if __name__ == '__main__':
    main()
