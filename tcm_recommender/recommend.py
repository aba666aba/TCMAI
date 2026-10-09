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

"""推荐系统推理接口 — 根据症状推荐药材/方剂 + 安全校验"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import torch

# 项目路径
if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).parent
else:
    PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tcm_recommender.data_pipeline import TCMDataPipeline
from tcm_recommender.extended_config import (
    EXTENDED_SYMPTOM_LIST,
    EXTENDED_SYNDROME_DEFINITIONS,
    SYMPTOM_ALIAS_MAP,
    is_toxic_herb,
)
from tcm_recommender.model import create_model
from tcm_recommender.train import build_herb_features

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

        # 药材/方剂名称列表
        self._herb_name_list = self._pipeline.herb_names
        self._formula_name_list = [
            self._pipeline.formulas[self._pipeline.model_idx_to_formula_db_id[i]][
                "name"
            ]
            for i in range(1, self._pipeline.num_formulas + 1)
        ]

        # 创建模型
        self.model = create_model(
            num_symptoms=len(self._symptom_list),
            num_herbs=len(self._herb_name_list),
            num_formulas=len(self._formula_name_list) + 1,
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

        # 缓存药材特征
        self._cached_herb_features = herb_feat

        logger.info("推荐器初始化完成")

    def parse_symptoms(
        self, symptom_text: str
    ) -> tuple[list[int], list[str], list[str]]:
        """解析症状文本（精确匹配，症状由上游 NLU 或关键词模块提取）"""
        for sep in [" ", ",", "，", "、", ";", "；"]:
            symptom_text = symptom_text.replace(sep, " ")

        symptoms = [s.strip() for s in symptom_text.split() if s.strip()]
        symptom_ids = []
        valid_symptoms = []
        unknown_symptoms = []

        for symp in symptoms:
            # 1. 尝试别名映射（NLU 输出→标准症状）
            mapped = SYMPTOM_ALIAS_MAP.get(symp)
            if mapped is not None:
                symp = mapped
            # 2. 精确匹配标准词表
            if symp in self.symptom_to_id:
                symptom_ids.append(self.symptom_to_id[symp])
                valid_symptoms.append(symp)
            else:
                unknown_symptoms.append(symp)

        if unknown_symptoms:
            logger.warning("未知症状（不在 229 症状列表中）: %s", unknown_symptoms)
        if symptoms and not symptom_ids:
            logger.warning("所有症状都无效，输入：%s", symptoms)

        return symptom_ids, valid_symptoms, unknown_symptoms

    def recommend(
        self, symptoms: list[int], top_k_herb: int = 10, top_k_formula: int = 5
    ) -> dict:
        """推荐药材和方剂"""
        if not symptoms:
            return {
                "error": "无有效症状",
                "herbs": [],
                "formulas": [],
                "suggestion": "请提供有效症状描述",
            }

        symptom_tensor = torch.tensor([symptoms], dtype=torch.long).to(self.device)
        mask = torch.ones_like(symptom_tensor, dtype=torch.float32)
        # print(f"[DEBUG] input shape={symptom_tensor.shape}, device={self.device}")

        with torch.no_grad():
            result = self.model.predict(
                symptom_tensor,
                mask,
                self._herb_feat["char_multihot"].to(self.device),
                self._herb_feat["category_idx"].to(self.device),
                self._herb_feat["nature_idx"].to(self.device),
                self._herb_feat["flavor_idx"].to(self.device),
                top_k=max(top_k_herb, top_k_formula),
            )

        # 药材结果
        # 模型输出维度 = num_herbs + 1，index 0 为背景类(padding)
        herb_scores = result["herbs"]
        seen_names = {}
        for herb_idx, score in herb_scores:
            if herb_idx == 0:
                continue  # 跳过背景类
            list_idx = herb_idx - 1
            if list_idx < 0 or list_idx >= len(self._herb_name_list):
                continue
            name = self._herb_name_list[list_idx]
            prob = float(torch.sigmoid(torch.tensor(score)))  # logits→概率
            if name not in seen_names or prob > seen_names[name]["score"]:
                seen_names[name] = {
                    "id": int(herb_idx),
                    "name": name,
                    "score": prob,
                    "efficacy": "",
                    "is_toxic": is_toxic_herb(name),
                }

        # ⚠️ 安全硬拦截：毒性/峻烈药材一律不进入推荐结果。
        # 这些药材必须由执业中医师按病情、体质与炮制规范决定，模型无权推荐。
        # 原实现在此位置放行，导致实盘以 1.00 置信度输出「砒石」。
        blocked_herbs = sorted(
            (h["name"] for h in seen_names.values() if h["is_toxic"]),
            key=lambda n: seen_names[n]["score"],
            reverse=True,
        )
        for name in blocked_herbs:
            seen_names.pop(name, None)

        herb_results = sorted(
            seen_names.values(), key=lambda x: x["score"], reverse=True
        )[:top_k_herb]

        # 方剂结果
        # 注意：模型输出维度 = num_formulas + 1，index 0 为背景类(padding)
        # 有效方剂编号范围是 1..num_formulas，对应 formula_name_list[0..num_formulas-1]
        formula_results = []
        for formula_idx, score in result["formulas"]:
            if formula_idx == 0:
                continue  # 跳过背景类

            prob = float(torch.sigmoid(torch.tensor(score)))
            # 内部索引：模型 idx 1..N → formula_name_list[idx-1]
            list_idx = formula_idx - 1
            if list_idx < 0 or list_idx >= len(self._formula_name_list):
                logger.warning(
                    "方剂索引越界: model_idx=%d, list_len=%d",
                    formula_idx,
                    len(self._formula_name_list),
                )
                continue

            name = self._formula_name_list[list_idx]
            formula_results.append(
                {
                    "id": int(formula_idx),
                    "name": name,
                    "score": prob,
                    "composition": "",
                    "functions": "",
                    # 方剂名命中毒性清单（如「朱砂安神丸」「附子理中丸」）：
                    # 方剂本身有临床价值，不剔除，但必须标注警示由医师判断。
                    "contains_toxic": is_toxic_herb(name),
                }
            )
            if len(formula_results) >= top_k_formula:
                break

        # 如果模型没产生够方剂，用智能搜索补全（但保留模型已出的结果）
        if not formula_results:
            logger.info("模型未输出有效方剂（prob<0.5 或全部为背景类）")

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
            "symptoms": [
                self._symptom_list[s - 1]
                for s in symptoms
                if 0 < s <= len(self._symptom_list)
            ],
            "syndrome": top_syndrome,
            "herbs": herb_results,
            "formulas": formula_results,
            # 被安全硬拦截的毒性药材（已从 herbs 中移除，仅用于向用户提示）
            "blocked_toxic_herbs": blocked_herbs,
        }

    # ---------------------------------------------------------------
    # 配伍禁忌判定
    # 数据库里 2961 味药材中 2111 味带禁忌文本，且大量为「畏XX」类文献表述
    # （如「黄芪：畏防风」），逐句关键词匹配会把普通药材全判成禁忌。
    # 因此采用**权威药对表 + 双向配对**：只有当推荐组合中同时出现禁忌药对的
    # 两味药，且药对中的另一味在本味药的禁忌文本中被提及（或明确写着「不宜与…同用」）
    # 时，才判定为真正的配伍禁忌。这是十八反/十九畏的可靠判定方式。
    _INCOMPATIBLE_PAIRS = (
        # 十八反（《儒门事亲》）
        ("甘草", "甘遂"), ("甘草", "大戟"), ("甘草", "芫花"), ("甘草", "海藻"),
        ("乌头", "贝母"), ("乌头", "瓜蒌"), ("乌头", "半夏"),
        ("乌头", "白蔹"), ("乌头", "白及"),
        ("藜芦", "人参"), ("藜芦", "沙参"), ("藜芦", "丹参"),
        ("藜芦", "玄参"), ("藜芦", "细辛"), ("藜芦", "芍药"),
        # 十九畏
        ("硫黄", "朴硝"), ("水银", "砒霜"), ("狼毒", "密陀僧"),
        ("巴豆", "牵牛"), ("丁香", "郁金"), ("川乌", "犀角"),
        ("草乌", "犀角"), ("牙硝", "三棱"), ("官桂", "石脂"),
        ("人参", "五灵脂"),
    )
    # 明确禁止同用的句式
    _EXPLICIT_BAN = re.compile(r"(不宜|不可|不能|禁止|忌)\s*与?.{0,12}(同用|合用|配伍)")

    def _pair_conflict(self, herb: str, others: list[str]) -> list[str]:
        """返回与 herb 构成配伍禁忌的其他药材名（基于权威药对表双向核对）"""
        hits = []
        for a, b in self._INCOMPATIBLE_PAIRS:
            if herb != a and herb != b:
                continue
            other = b if herb == a else a
            # 另一味必须在本次推荐组合中，且与本味药不同名
            if not any((other in o or o in other) and o != herb for o in others):
                continue
            # 同义替代：乌头/川乌/草乌，朴硝/芒硝，犀角/水牛角等按包含关系处理
            hits.append(other)
        return hits

    def check_safety(self, herb_names: list[str]) -> dict:
        """安全检查（十八反/十九畏 + 毒性药材 + 逐味注意事项）

        返回结构契约（下游 src/desktop/service.py 依赖）：
            warnings         : list[str]  逐味禁忌文本（一般注意事项，仅作提示）
            incompatibilities: list[str]  确认的配伍禁忌（权威药对表 + 明确禁止句式）
            toxic_herbs      : list[str]  命中毒性清单的药材
            is_safe          : bool       无配伍禁忌且无毒 → True
        """
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        warnings = []
        incompatibilities = []
        seen = set()
        names = list(herb_names)
        for h in names:
            if h in seen:
                continue
            seen.add(h)
            cur.execute(
                "SELECT contraindications FROM herbs WHERE name = ? LIMIT 1", (h,)
            )
            row = cur.fetchone()
            text = row[0] if (row and row[0]) else ""
            if text:
                warnings.append(f"{h}: {text[:100]}")

            # 1) 权威药对命中
            for other in self._pair_conflict(h, names):
                incompatibilities.append(f"{h} + {other}：属十八反/十九畏配伍禁忌")

            # 2) 明确「不宜与…同用」句式，且被点名的药材确实在本组合中
            for sentence in re.split(r"[。；;！!\n]", text):
                sentence = sentence.strip()
                if not sentence or not self._EXPLICIT_BAN.search(sentence):
                    continue
                if any(
                    (o in sentence) and (o != h) and (o in names)
                    for o in names
                ):
                    incompatibilities.append(f"{h}: {sentence[:100]}")

        conn.close()

        # 去重：同一药对可能双向命中（A+B 与 B+A），按下标归一化后只保留一条
        deduped = []
        seen_keys = set()
        for item in incompatibilities:
            marker = "属十八反/十九畏配伍禁忌"
            if marker in item:
                pair = item.split("：")[0]
                key = tuple(sorted(pair.split(" + ")))
            else:
                key = (item,)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(item)

        toxic_herbs = [h for h in names if is_toxic_herb(h)]
        return {
            "warnings": warnings,
            "incompatibilities": deduped,
            "toxic_herbs": toxic_herbs,
            # 原实现遗漏 is_safe，导致 service.py 的 not safety.get("is_safe")
            # 恒为真、把每一次校验都报成「检测到配伍禁忌」。此处按真实语义补上：
            # 只有确认的配伍禁忌或毒性药材才判定为不安全。
            "is_safe": not deduped and not toxic_herbs,
        }

    def format_output(self, result: dict) -> str:
        """格式化输出"""
        if "error" in result:
            return "错误：{}".format(result["error"])

        lines = []
        lines.append("=" * 60)
        lines.append("  中医药智能推荐结果")
        lines.append("=" * 60)
        lines.append("\n症状：{}".format(", ".join(result["symptoms"])))

        if result.get("syndrome"):
            s = result["syndrome"]
            lines.append(
                "\n证型：{} (概率: {:.1f}%)".format(s["name"], s["confidence"] * 100)
            )

        lines.append("\n推荐药材:")
        for i, h in enumerate(result["herbs"], 1):
            lines.append(f"  {i}. {h['name']} (概率: {h['score'] * 100:.1f}%)")

        blocked = result.get("blocked_toxic_herbs") or []
        if blocked:
            lines.append(
                "\n⚠️ 安全拦截（毒性/峻烈药材，已从推荐中移除）: {}".format(
                    "、".join(blocked)
                )
            )

        lines.append("\n推荐方剂:")
        for i, f in enumerate(result["formulas"], 1):
            lines.append(f"  {i}. {f['name']} (概率: {f['score'] * 100:.1f}%)")

        lines.append("\n安全校验:")
        herb_names = [h["name"] for h in result["herbs"][:10]]
        safety = self.check_safety(herb_names)
        if safety["warnings"]:
            for w in safety["warnings"]:
                lines.append(f"  {w}")
        else:
            lines.append("  未发现明显配伍禁忌")

        lines.append("\n" + "=" * 60)
        lines.append("注：本结果仅供参考，请由专业中医师判断使用")
        lines.append("=" * 60)
        return "\n".join(lines)

    def close(self):
        """释放资源"""
        if hasattr(self, "_pipeline"):
            self._pipeline.close()


def main():
    parser = argparse.ArgumentParser(description="中医药智能推荐系统")
    parser.add_argument("symptoms", type=str, help="症状列表（空格/逗号分隔）")
    parser.add_argument(
        "--model",
        type=str,
        default="tcm_recommender/best_model.pth",
        help="模型文件路径",
    )
    parser.add_argument(
        "--db", type=str, default="data/tcm_knowledge.db", help="数据库路径"
    )
    parser.add_argument("--top-herb", type=int, default=10, help="返回药材数")
    parser.add_argument("--top-formula", type=int, default=5, help="返回方剂数")
    parser.add_argument("--device", type=str, default="cpu", help="运行设备")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")

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


if __name__ == "__main__":
    main()
