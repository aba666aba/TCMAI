# Copyright (C) 2026 中医药智能问答系统
# Licensed under Apache License 2.0

"""
中医药推荐系统 - 大规模数据管线

从 2,000+ 药材、500+ 方剂、200 症状、17 证型的知识库中，
通过组合增强 + 噪声注入 + 证型映射生成 100,000+ 高质量训练样本。

管线：
  1. 加载基础数据（药材/方剂/症状/证型）
  2. 构建症状-药材关联矩阵
  3. 构建症状-方剂关联矩阵
  4. 证型驱动样本生成（主要来源，~60K）
  5. 药材驱动样本生成（~20K）
  6. 方剂驱动样本生成（~10K）
  7. 组合增强 + 噪声注入（~30K）
  8. 输出 HDF5/NPZ 格式的高效训练数据
"""

import logging
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
_project = Path(__file__).resolve().parents[1]
if str(_project) not in sys.path:
    sys.path.insert(0, str(_project))

logger = logging.getLogger(__name__)

# ── 常量 ─────────────────────────────────────────
MAX_SYMPTOMS_PER_SAMPLE = 32
MAX_HERBS_PER_SAMPLE = 40
MIN_SYMPTOMS = 2
RANDOM_SEED = 42


class TCMDataPipeline:
    """大规模中医药训练数据管线"""

    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self.rng = np.random.RandomState(RANDOM_SEED)

        # 核心数据
        self.herbs: Dict[int, dict] = {}
        self.formulas: Dict[int, dict] = {}
        self.symptoms: List[str] = []
        self.syndromes: Dict[str, dict] = {}

        # 索引
        self.symptom_to_idx: Dict[str, int] = {}
        self.herb_name_to_db_id: Dict[str, int] = {}
        self.herb_db_id_to_model_idx: Dict[int, int] = {}
        self.formula_db_id_to_model_idx: Dict[int, int] = {}
        self.model_idx_to_herb_db_id: Dict[int, int] = {}
        self.model_idx_to_formula_db_id: Dict[int, int] = {}

        # 关联矩阵
        self.symptom_herb_matrix: Optional[np.ndarray] = None  # [num_s, num_h]
        self.symptom_formula_matrix: Optional[np.ndarray] = None  # [num_s, num_f]
        self.herb_cooccurrence: Optional[np.ndarray] = None  # [num_h, num_h]

        # 药材元数据
        self.herb_names: List[str] = []
        self.herb_categories: List[str] = []
        self.herb_toxic_mask: np.ndarray = np.array([])

        self.num_herbs = 0
        self.num_formulas = 0
        self.num_symptoms = 0
        self.num_syndromes = 0

    # ═══════════════════════════════════════════════
    # 1. 数据加载
    # ═══════════════════════════════════════════════

    def load_all(self) -> "TCMDataPipeline":
        """加载全部基础数据"""
        logger.info("=" * 50)
        logger.info("TCM 数据管线启动")
        self._load_herbs()
        self._load_formulas()
        self._load_symptoms()
        self._load_syndromes()
        self._build_indices()
        self._build_association_matrices()
        self._build_cooccurrence()
        logger.info("数据加载完成: %s 药材 | %s 方剂 | %s 症状 | %s 证型",
                     self.num_herbs, self.num_formulas, self.num_symptoms, self.num_syndromes)
        return self

    def _load_herbs(self):
        rows = self.db.execute(
            "SELECT id, name, category, nature, flavor, meridian, efficacy, indications, "
            "contraindications FROM herbs"
        ).fetchall()
        for r in rows:
            hid = r["id"]
            self.herbs[hid] = dict(r)
            self.herb_name_to_db_id[r["name"]] = hid
        self.num_herbs = len(self.herbs)
        logger.info("  [OK] 药材: %s", self.num_herbs)

    def _load_formulas(self):
        rows = self.db.execute(
            "SELECT id, name, category, composition, functions, indications, modification "
            "FROM formulas"
        ).fetchall()
        for r in rows:
            self.formulas[r["id"]] = dict(r)
        self.num_formulas = len(self.formulas)
        logger.info("  [OK] 方剂: %s", self.num_formulas)

    def _load_symptoms(self):
        # 统一使用 extended_config（同包，无需 try-import fallback）
        from tcm_recommender.extended_config import EXTENDED_SYMPTOM_LIST
        self.symptoms = list(EXTENDED_SYMPTOM_LIST)
        self.num_symptoms = len(self.symptoms)
        self.symptom_to_idx = {s: i for i, s in enumerate(self.symptoms)}
        logger.info("  [OK] 症状: %s", self.num_symptoms)

    def _load_syndromes(self):
        # 统一使用 extended_config（17 证型），替代早期的 SYNDROME_RULES（3 证型）
        from tcm_recommender.extended_config import EXTENDED_SYNDROME_DEFINITIONS
        self.syndromes = EXTENDED_SYNDROME_DEFINITIONS
        self.num_syndromes = len(self.syndromes)
        logger.info("  [OK] 证型: %s", self.num_syndromes)

    def _build_indices(self):
        """构建 DB ID ↔ 模型索引的双向映射（1-based model idx, 0=padding）"""
        for idx, hid in enumerate(sorted(self.herbs.keys()), 1):
            self.herb_db_id_to_model_idx[hid] = idx
            self.model_idx_to_herb_db_id[idx] = hid
        for idx, fid in enumerate(sorted(self.formulas.keys()), 1):
            self.formula_db_id_to_model_idx[fid] = idx
            self.model_idx_to_formula_db_id[idx] = fid
        self.herb_names = [
            self.herbs[self.model_idx_to_herb_db_id[i]]["name"]
            for i in range(1, self.num_herbs + 1)
        ]
        self.herb_categories = [
            self.herbs[self.model_idx_to_herb_db_id[i]].get("category", "")
            for i in range(1, self.num_herbs + 1)
        ]
        from tcm_recommender.extended_config import EXTENDED_TOXIC_HERBS
        toxic_set = set(EXTENDED_TOXIC_HERBS)
        self.herb_toxic_mask = np.array(
            [1.0 if self.herb_names[i] in toxic_set else 0.0 for i in range(self.num_herbs)],
            dtype=np.float32,
        )

    def _build_association_matrices(self):
        """构建症状-药材和症状-方剂关联矩阵"""
        logger.info("  构建关联矩阵...")
        ns, nh, nf = self.num_symptoms, self.num_herbs, self.num_formulas

        # 症状-药材：基于 efficacy + indications 文本匹配
        sym_herb = np.zeros((ns, nh), dtype=np.float32)
        for hid, herb in self.herbs.items():
            h_idx = self.herb_db_id_to_model_idx.get(hid)
            if h_idx is None:
                continue
            text = f"{herb.get('efficacy', '')} {herb.get('indications', '')}"
            for si, s_name in enumerate(self.symptoms):
                if s_name in text:
                    sym_herb[si, h_idx - 1] = 1.0
        self.symptom_herb_matrix = sym_herb

        # 症状-方剂
        sym_formula = np.zeros((ns, nf), dtype=np.float32)
        for fid, formula in self.formulas.items():
            f_idx = self.formula_db_id_to_model_idx.get(fid)
            if f_idx is None:
                continue
            text = f"{formula.get('functions', '')} {formula.get('indications', '')}"
            for si, s_name in enumerate(self.symptoms):
                if s_name in text:
                    sym_formula[si, f_idx - 1] = 1.0
        self.symptom_formula_matrix = sym_formula
        logger.info("    症状-药材: %s 关联, 症状-方剂: %s 关联",
                     int(sym_herb.sum()), int(sym_formula.sum()))

    def _build_cooccurrence(self):
        """构建药材共现矩阵（基于方剂组成）"""
        cooc = np.zeros((self.num_herbs, self.num_herbs), dtype=np.float32)
        for formula in self.formulas.values():
            comp = formula.get("composition", "")
            herb_indices = self._parse_composition_indices(comp)
            for i in range(len(herb_indices)):
                for j in range(i + 1, len(herb_indices)):
                    hi, hj = herb_indices[i], herb_indices[j]
                    cooc[hi, hj] += 1
                    cooc[hj, hi] += 1
        # 归一化
        row_sum = cooc.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        self.herb_cooccurrence = cooc / row_sum
        logger.info("  药材共现矩阵: %s×%s, %s 非零边",
                     self.num_herbs, self.num_herbs, int((cooc > 0).sum()))

    def _parse_composition_indices(self, composition: str) -> List[int]:
        """解析方剂组成 → model_idx 列表"""
        if not composition:
            return []
        import re

        def _clean(name: str) -> str:
            return re.sub(r"[（(][^)）]*[)）]|\d+\s*[克g枚片个毫升ml两钱分斤]|[.．～~]", "", name).strip()

        text = composition.replace("、", "，").replace("四君子汤 + 四物汤", "")
        parts = text.split("，")
        indices = []
        for part in parts:
            part = part.strip()
            if not part or "+" in part:
                continue
            cleaned = _clean(part)
            if not cleaned:
                continue
            hid = self.herb_name_to_db_id.get(cleaned) or self.herb_name_to_db_id.get(part.strip())
            if hid and hid in self.herb_db_id_to_model_idx:
                idx = self.herb_db_id_to_model_idx[hid] - 1
                if idx not in indices:
                    indices.append(idx)
        return indices

    # ═══════════════════════════════════════════════
    # 2. 样本生成
    # ═══════════════════════════════════════════════

    def generate_samples(self, target_count: int = 150000) -> Dict[str, np.ndarray]:
        """主生成函数：生成 target_count 个训练样本"""
        logger.info("=" * 50)
        logger.info("开始生成 %s 个训练样本...", target_count)

        all_symptoms = []       # List[List[int]]
        all_herb_labels = []    # List[np.ndarray]
        all_formula_labels = [] # List[int]
        all_syndrome_labels = []# List[int]
        all_weights = []        # List[float]

        # ── 阶段 1: 证型驱动 ──
        syn_samples = self._generate_syndrome_samples(target_count // 2)
        all_symptoms.extend(syn_samples[0])
        all_herb_labels.extend(syn_samples[1])
        all_formula_labels.extend(syn_samples[2])
        all_syndrome_labels.extend(syn_samples[3])
        all_weights.extend(syn_samples[4])
        logger.info("  证型驱动: %s 样本", len(syn_samples[0]))

        # ── 阶段 2: 药材驱动 ──
        herb_samples = self._generate_herb_samples(target_count // 5)
        all_symptoms.extend(herb_samples[0])
        all_herb_labels.extend(herb_samples[1])
        all_formula_labels.extend(herb_samples[2])
        all_syndrome_labels.extend(herb_samples[3])
        all_weights.extend(herb_samples[4])
        logger.info("  药材驱动: %s 样本", len(herb_samples[0]))

        # ── 阶段 3: 方剂驱动 ──
        formula_samples = self._generate_formula_samples(target_count // 8)
        all_symptoms.extend(formula_samples[0])
        all_herb_labels.extend(formula_samples[1])
        all_formula_labels.extend(formula_samples[2])
        all_syndrome_labels.extend(formula_samples[3])
        all_weights.extend(formula_samples[4])
        logger.info("  方剂驱动: %s 样本", len(formula_samples[0]))

        # ── 阶段 4: 组合增强 ──
        aug_samples = self._generate_augmented_samples(target_count // 4)
        all_symptoms.extend(aug_samples[0])
        all_herb_labels.extend(aug_samples[1])
        all_formula_labels.extend(aug_samples[2])
        all_syndrome_labels.extend(aug_samples[3])
        all_weights.extend(aug_samples[4])
        logger.info("  组合增强: %s 样本", len(aug_samples[0]))

        # ── 阶段 5: 负样本 ──
        neg_samples = self._generate_negative_samples(target_count // 6)  # 提高负样本比例
        all_symptoms.extend(neg_samples[0])
        all_herb_labels.extend(neg_samples[1])
        all_formula_labels.extend(neg_samples[2])
        all_syndrome_labels.extend(neg_samples[3])
        all_weights.extend(neg_samples[4])
        logger.info("  负样本: %s 样本", len(neg_samples[0]))

        # 打包
        result = self._pack_samples(all_symptoms, all_herb_labels, all_formula_labels,
                                     all_syndrome_labels, all_weights)
        logger.info("=" * 50)
        logger.info("生成完成: %s 样本", len(all_symptoms))
        self._print_stats(result)
        return result

    def _generate_syndrome_samples(self, count: int):
        """证型驱动：为每个证型生成症状子集变体"""
        symptoms, herbs, formulas, syndromes, weights = [], [], [], [], []
        ns, nh = self.num_symptoms, self.num_herbs
        syn_names = list(self.syndromes.keys())

        for syn_idx, (syn_name, syn_info) in enumerate(self.syndromes.items(), 1):
            syn_symptom_names = syn_info.get("symptoms", [])
            syn_herb_names = syn_info.get("recommend_herbs", [])
            syn_formula_names = syn_info.get("recommend_formulas", [])

            if len(syn_symptom_names) < 2:
                continue

            # 症状 → indices
            syn_s_indices = [self.symptom_to_idx[s] for s in syn_symptom_names if s in self.symptom_to_idx]
            if len(syn_s_indices) < 2:
                continue

            # 药材 → indices
            syn_h_indices = []
            for h_name in syn_herb_names:
                hid = self.herb_name_to_db_id.get(h_name)
                if hid and hid in self.herb_db_id_to_model_idx:
                    syn_h_indices.append(self.herb_db_id_to_model_idx[hid] - 1)

            # 方剂 → index（遍历所有推荐方剂，取第一个匹配的）
            syn_f_label = 0
            if syn_formula_names:
                for f_name in syn_formula_names:
                    for fid, f_info in self.formulas.items():
                        if f_info.get("name") == f_name and fid in self.formula_db_id_to_model_idx:
                            syn_f_label = self.formula_db_id_to_model_idx[fid]
                            break
                    if syn_f_label != 0:
                        break

            # 基础样本
            num_base = len(syn_s_indices)
            samples_per_syndrome = max(count // self.num_syndromes, 50)

            for _ in range(samples_per_syndrome):
                # 随机选择 2~num_base 个症状
                k = self.rng.randint(2, min(num_base + 1, MAX_SYMPTOMS_PER_SAMPLE // 2))
                chosen = sorted(self.rng.choice(syn_s_indices, size=k, replace=False))

                # 噪声注入：10% 概率加入无关症状
                if self.rng.random() < 0.15:
                    unrelated = [si for si in range(ns) if si not in set(syn_s_indices)]
                    if unrelated:
                        noise_n = self.rng.randint(1, min(3, len(unrelated)))
                        noise = list(self.rng.choice(unrelated, size=noise_n, replace=False))
                        chosen = sorted(set(chosen) | set(noise))

                hl = np.zeros(nh, dtype=np.float32)
                if syn_h_indices:
                    for hi in syn_h_indices:
                        if 0 <= hi < nh:
                            hl[hi] = 1.0

                symptoms.append(chosen)
                herbs.append(hl)
                formulas.append(syn_f_label)
                syndromes.append(syn_idx)
                weights.append(1.0)

        return symptoms, herbs, formulas, syndromes, weights

    def _generate_herb_samples(self, count: int):
        """药材驱动：为每味药材匹配其关联症状"""
        symptoms_list, herbs_list, formulas_list, syndromes_list, weights = [], [], [], [], []
        ns, nh, nf = self.num_symptoms, self.num_herbs, self.num_formulas

        for hid, herb in self.herbs.items():
            h_model_idx = self.herb_db_id_to_model_idx.get(hid)
            if h_model_idx is None:
                continue

            text = f"{herb.get('efficacy', '')} {herb.get('indications', '')}"
            related_s = [si for si, s_name in enumerate(self.symptoms) if s_name in text]
            if len(related_s) < 2:
                continue

            h_idx = h_model_idx - 1
            samples_per_herb = max(count // nh, 2)

            for _ in range(samples_per_herb):
                k = self.rng.randint(2, min(len(related_s) + 1, MAX_SYMPTOMS_PER_SAMPLE // 2))
                chosen = sorted(self.rng.choice(related_s, size=k, replace=False))

                # 噪声
                if self.rng.random() < 0.12:
                    unrelated = [si for si in range(ns) if si not in set(related_s)]
                    if unrelated:
                        noise_n = self.rng.randint(1, min(2, len(unrelated)))
                        chosen = sorted(set(chosen) | set(self.rng.choice(unrelated, size=noise_n, replace=False)))

                hl = np.zeros(nh, dtype=np.float32)
                hl[h_idx] = 1.0

                # 补充同症状组合下的其他相关药材（通过关联矩阵）
                if self.symptom_herb_matrix is not None:
                    co_herb_scores = self.symptom_herb_matrix[list(chosen)].sum(axis=0)  # [nh]
                    co_herb_scores[h_idx] = 0  # 排除自身
                    # 选择关联得分 ≥ 50% 最大值的其他药材，最多3味
                    max_score = co_herb_scores.max()
                    if max_score > 0:
                        threshold = max_score * 0.5
                        extra_candidates = np.where(co_herb_scores >= threshold)[0]
                        extra_n = min(len(extra_candidates), 3)
                        if extra_n > 0:
                            extra = self.rng.choice(extra_candidates, size=extra_n, replace=False)
                            for ei in extra:
                                hl[ei] = 1.0

                # 推断证型
                syn_label = self._infer_syndrome_from_symptoms(set(chosen))

                # 推断方剂
                f_scores = self.symptom_formula_matrix[list(chosen)].sum(axis=0)
                f_label = int(np.argmax(f_scores)) + 1 if f_scores.max() > 0 else 0

                symptoms_list.append(chosen)
                herbs_list.append(hl)
                formulas_list.append(f_label)
                syndromes_list.append(syn_label)
                weights.append(0.8)

        return symptoms_list, herbs_list, formulas_list, syndromes_list, weights

    def _generate_formula_samples(self, count: int):
        """方剂驱动：为每个方剂生成训练样本"""
        syms, herbs, forms, syns, weights = [], [], [], [], []
        nh = self.num_herbs

        for fid, formula in self.formulas.items():
            f_model_idx = self.formula_db_id_to_model_idx.get(fid)
            if f_model_idx is None:
                continue

            # 解析药材组成
            herb_indices = self._parse_composition_indices(formula.get("composition", ""))
            if len(herb_indices) < 2:
                continue

            # 匹配症状
            text = f"{formula.get('functions', '')} {formula.get('indications', '')}"
            related_s = [si for si, s_name in enumerate(self.symptoms) if s_name in text]
            if not related_s:
                continue

            samples_per = max(count // self.num_formulas, 1)
            for _ in range(samples_per):
                max_k = min(len(related_s), 6)
                if max_k < 2:
                    continue
                k = self.rng.randint(2, max_k + 1)
                chosen = sorted(self.rng.choice(related_s, size=k, replace=False))

                hl = np.zeros(nh, dtype=np.float32)
                for hi in herb_indices:
                    if 0 <= hi < nh:
                        hl[hi] = 1.0

                syn_label = self._infer_syndrome_from_symptoms(set(chosen))

                syms.append(chosen)
                herbs.append(hl)
                forms.append(f_model_idx)
                syns.append(syn_label)
                weights.append(0.9)

        return syms, herbs, forms, syns, weights

    def _generate_augmented_samples(self, count: int):
        """组合增强：随机组合症状 + 基于关联矩阵生成标签（限制正标签数≤20）"""
        syms, herbs, forms, syns, weights = [], [], [], [], []
        ns, nh = self.num_symptoms, self.num_herbs
        all_s_indices = list(range(ns))
        MAX_POS_HERBS = 20

        for _ in range(count):
            # 随机 2-6 个症状
            k = self.rng.randint(MIN_SYMPTOMS, 7)
            chosen = sorted(self.rng.choice(all_s_indices, size=min(k, ns), replace=False))

            # 基于关联矩阵计算药材标签
            herb_scores = self.symptom_herb_matrix[chosen].sum(axis=0)  # [nh]
            hl = np.zeros(nh, dtype=np.float32)

            if herb_scores.max() > 0:
                # 要求药材至少匹配2个症状，且得分 ≥ 0.5*max
                min_hits = min(2, len(chosen))
                threshold = herb_scores.max() * 0.5
                # 得分满足条件的候选药材
                candidates = np.where((herb_scores >= threshold) & (herb_scores > 0))[0]
                if len(candidates) == 0:
                    # 降级：取 top-10
                    top_n = min(10, int((herb_scores > 0).sum()))
                    if top_n > 0:
                        candidates = np.argsort(herb_scores)[-top_n:]
                # 如果还是太多，只取 top-20
                if len(candidates) > MAX_POS_HERBS:
                    candidates = candidates[np.argsort(herb_scores[candidates])[-MAX_POS_HERBS:]]
                for hi in candidates:
                    hl[hi] = 1.0

            # 方剂标签
            f_scores = self.symptom_formula_matrix[chosen].sum(axis=0)
            f_label = int(np.argmax(f_scores)) + 1 if f_scores.max() > 0 else 0

            # 证型
            syn_label = self._infer_syndrome_from_symptoms(set(chosen))

            syms.append(chosen)
            herbs.append(hl)
            forms.append(f_label)
            syns.append(syn_label)
            weights.append(0.5)  # 较低权重

        return syms, herbs, forms, syns, weights

    def _generate_negative_samples(self, count: int):
        """负样本：语义不匹配的症状-药材对（对比学习用）"""
        syms, herbs, forms, syns, weights = [], [], [], [], []
        ns, nh = self.num_symptoms, self.num_herbs
        all_s = list(range(ns))

        for _ in range(count):
            k = self.rng.randint(MIN_SYMPTOMS, 5)
            chosen = sorted(self.rng.choice(all_s, size=k, replace=False))

            # 负样本：选择与这些症状关联度为 0 的药材
            herb_scores = self.symptom_herb_matrix[chosen].sum(axis=0)
            zero_mask = herb_scores == 0
            if zero_mask.sum() == 0:
                continue

            # 随机选一些零关联药材作为"不应推荐的"
            neg_herb_indices = np.where(zero_mask)[0]
            n_neg = self.rng.randint(1, min(4, len(neg_herb_indices)))
            neg_chosen = self.rng.choice(neg_herb_indices, size=n_neg, replace=False)

            hl = np.zeros(nh, dtype=np.float32)
            for hi in neg_chosen:
                hl[hi] = -1.0  # 负标签

            syn_label = self._infer_syndrome_from_symptoms(set(chosen))

            syms.append(chosen)
            herbs.append(hl)
            forms.append(0)
            syns.append(0)
            weights.append(0.3)

        return syms, herbs, forms, syns, weights

    def _infer_syndrome_from_symptoms(self, symptom_set: Set[int]) -> int:
        """从症状集合推断最匹配证型"""
        best_id, best_score = 0, 0.0
        for syn_idx, (syn_name, syn_info) in enumerate(self.syndromes.items(), 1):
            required = syn_info.get("symptoms", [])
            if not required:
                continue
            req_indices = {self.symptom_to_idx.get(s, -1) for s in required}
            req_indices.discard(-1)
            if not req_indices:
                continue
            matched = len(symptom_set & req_indices)
            score = matched / len(req_indices)
            if score > best_score and score >= 0.3:
                best_score = score
                best_id = syn_idx
        return best_id

    # ═══════════════════════════════════════════════
    # 3. 数据打包
    # ═══════════════════════════════════════════════

    def _pack_samples(self, symptoms, herb_labels, formula_labels, syndrome_labels, weights):
        """将变长样本列表打包为固定格式 NDArray（padding）"""
        n = len(symptoms)
        ns, nh = self.num_symptoms, self.num_herbs

        # 症状张量
        symptom_tensor = np.zeros((n, MAX_SYMPTOMS_PER_SAMPLE), dtype=np.int32)
        symptom_mask = np.zeros((n, MAX_SYMPTOMS_PER_SAMPLE), dtype=np.float32)
        for i, s_list in enumerate(symptoms):
            s_list = s_list[:MAX_SYMPTOMS_PER_SAMPLE]
            symptom_tensor[i, :len(s_list)] = [s + 1 for s in s_list]  # 1-based
            symptom_mask[i, :len(s_list)] = 1.0

        # 药材多标签
        herb_label_tensor = np.zeros((n, nh), dtype=np.float32)
        for i, hl in enumerate(herb_labels):
            herb_label_tensor[i] = hl[:nh]

        # 方剂标签
        formula_label_tensor = np.array(formula_labels, dtype=np.int32)

        # 证型标签
        syndrome_label_tensor = np.array(syndrome_labels, dtype=np.int32)

        # 样本权重
        weight_tensor = np.array(weights, dtype=np.float32)

        return {
            "symptom_ids": symptom_tensor,
            "symptom_mask": symptom_mask,
            "herb_labels": herb_label_tensor,
            "formula_labels": formula_label_tensor,
            "syndrome_labels": syndrome_label_tensor,
            "sample_weights": weight_tensor,
            "num_samples": n,
            "num_symptoms": ns,
            "num_herbs": nh,
            "num_formulas": self.num_formulas,
            "num_syndromes": self.num_syndromes,
        }

    def _print_stats(self, result: dict):
        """打印数据统计"""
        hl = result["herb_labels"]
        fl = result["formula_labels"]
        sl = result["syndrome_labels"]

        pos_count = (hl > 0).sum(axis=1)
        has_herb = (pos_count > 0).mean()
        has_formula = (fl > 0).mean()
        has_syndrome = (sl > 0).mean()
        avg_herbs = pos_count.mean() if has_herb > 0 else 0
        avg_symptoms = result["symptom_mask"].sum(axis=1).mean()

        logger.info("  统计:")
        logger.info("    平均症状数: %.1f", avg_symptoms)
        logger.info("    平均正药材数: %.1f", avg_herbs)
        logger.info("    有方剂标签: %.1f%%", has_formula * 100)
        logger.info("    有证型标签: %.1f%%", has_syndrome * 100)
        logger.info("    有药材标签: %.1f%%", has_herb * 100)

    # ═══════════════════════════════════════════════
    # 4. 药材特征构建（持久化到 npz，避免 train.py 重复计算）
    # ═══════════════════════════════════════════════

    def build_herb_features(self) -> Dict[str, np.ndarray]:
        """构建药材特征（与 train.py build_herb_features 输出格式一致）

        Returns:
            dict 包含 char_multihot, category_idx, nature_idx, flavor_idx
        """
        nh = self.num_herbs
        if nh == 0:
            return {}

        # 字符多热矩阵
        NATURE_LIST = ["寒", "热", "温", "凉", "平"]
        FLAVOR_LIST = ["酸", "苦", "甘", "辛", "咸"]

        herb_texts: Dict[int, str] = {}
        for model_idx in range(1, nh + 1):
            db_id = self.model_idx_to_herb_db_id[model_idx]
            herb = self.herbs[db_id]
            herb_texts[model_idx] = f"{herb.get('efficacy', '')} {herb.get('indications', '')}"

        char_set: set = set()
        for text in herb_texts.values():
            for ch in text:
                if '\u4e00' <= ch <= '\u9fff':
                    char_set.add(ch)
        char_list = sorted(char_set)
        char_to_idx = {ch: i for i, ch in enumerate(char_list)}
        num_chars = len(char_list)

        char_mh = np.zeros((nh, num_chars), dtype=np.float32)
        for mi, text in herb_texts.items():
            for ch in text:
                if ch in char_to_idx:
                    char_mh[mi - 1, char_to_idx[ch]] = 1.0

        # 类别索引
        all_categories = sorted(set(
            self.herbs[self.model_idx_to_herb_db_id[mi]].get("category", "") or "未知"
            for mi in range(1, nh + 1)
        ))
        cat_to_idx = {c: i + 1 for i, c in enumerate(all_categories)}
        cat_idx = np.zeros(nh, dtype=np.int32)
        for mi in range(1, nh + 1):
            cat = self.herbs[self.model_idx_to_herb_db_id[mi]].get("category", "") or "未知"
            cat_idx[mi - 1] = cat_to_idx.get(cat, 0)

        # 药性索引
        nat_idx = np.zeros(nh, dtype=np.int32)
        for mi in range(1, nh + 1):
            nat = self.herbs[self.model_idx_to_herb_db_id[mi]].get("nature", "") or ""
            found = False
            for ni, n_name in enumerate(NATURE_LIST, 1):
                if n_name in nat:
                    nat_idx[mi - 1] = ni
                    found = True
                    break
            if not found:
                nat_idx[mi - 1] = len(NATURE_LIST) + 1

        # 药味索引
        flv_idx = np.zeros(nh, dtype=np.int32)
        for mi in range(1, nh + 1):
            flv = self.herbs[self.model_idx_to_herb_db_id[mi]].get("flavor", "") or ""
            found = False
            for fi, f_name in enumerate(FLAVOR_LIST, 1):
                if f_name in flv:
                    flv_idx[mi - 1] = fi
                    found = True
                    break
            if not found:
                flv_idx[mi - 1] = len(FLAVOR_LIST) + 1

        logger.info("药材特征: chars=%s, categories=%s", num_chars, len(all_categories))
        return {
            "char_multihot": char_mh,
            "category_idx": cat_idx,
            "nature_idx": nat_idx,
            "flavor_idx": flv_idx,
            "num_chars": num_chars,
            "num_categories": len(all_categories) + 1,
            "num_natures": len(NATURE_LIST) + 2,
            "num_flavors": len(FLAVOR_LIST) + 2,
        }

    # ═══════════════════════════════════════════════
    # 5. IO
    # ═══════════════════════════════════════════════

    def save(self, result: dict, path: str, include_herb_features: bool = True):
        """保存为压缩 NPZ 格式（可选包含药材特征缓存）"""
        save_data = dict(result)
        if include_herb_features and self.num_herbs > 0:
            feat = self.build_herb_features()
            save_data["herb_features"] = feat
            logger.info("药材特征已缓存到 npz")
        np.savez_compressed(path, **save_data)
        import os
        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info("数据已保存: %s (%.1f MB)", path, size_mb)

    def load(self, path: str) -> dict:
        return dict(np.load(path, allow_pickle=True))

    def close(self):
        self.db.close()


def main():
    """CLI 入口"""
    import argparse
    parser = argparse.ArgumentParser(description="TCM 数据管线")
    parser.add_argument("--db", default="data/tcm_knowledge.db")
    parser.add_argument("--output", default="data/train_data.npz")
    parser.add_argument("--count", type=int, default=100000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    pipeline = TCMDataPipeline(args.db)
    pipeline.load_all()
    result = pipeline.generate_samples(args.count)
    pipeline.save(result, args.output)
    pipeline.close()


if __name__ == "__main__":
    main()
