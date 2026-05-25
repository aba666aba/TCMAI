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
from typing import Optional, Tuple

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




# ── 功能关键词→症状映射表（用于方剂反推增强） ──

_FUNCTION_SYMPTOM_MAP = {
        "风寒": ["恶寒", "发热", "无汗", "头痛", "鼻塞"],
        "风热": ["发热", "恶风", "咽喉肿痛", "口渴", "头痛"],
        "风邪": ["恶寒", "发热", "头痛", "鼻塞"],
        "暑邪": ["发热", "口渴", "心烦"],
        "湿邪": ["脘痞", "腹胀", "食欲不振", "便溏", "水肿"],
        "燥邪": ["口干", "咽干", "干咳", "皮肤干燥"],
        "清热": ["发热", "口渴", "咽喉肿痛", "口干", "心烦", "面赤"],
        "解毒": ["发热", "咽喉肿痛", "疮疡", "痈肿"],
        "泻火": ["发热", "口渴", "心烦", "面赤", "目赤", "牙龈肿痛"],
        "凉血": ["发热", "斑疹", "便血", "鼻衄", "皮肤发黄"],
        "滋阴": ["口干", "盗汗", "五心烦热", "手足心热", "消瘦"],
        "养阴": ["口干", "盗汗", "五心烦热", "咽干"],
        "生津": ["口干", "口渴", "咽干"],
        "温阳": ["畏寒", "手足不温", "面色苍白", "腰膝酸软"],
        "助阳": ["畏寒", "腰膝酸软", "夜尿多"],
        "壮阳": ["畏寒", "腰膝酸软", "阳痿"],
        "补肾": ["腰膝酸软", "耳鸣", "头晕", "夜尿多", "腰酸"],
        "补肝": ["头晕", "目眩", "视物模糊", "胁痛"],
        "健脾": ["食欲不振", "乏力", "便溏", "腹胀", "面色萎黄"],
        "补脾": ["食欲不振", "乏力", "便溏", "腹胀"],
        "养心": ["心悸", "失眠", "心烦", "健忘"],
        "补心": ["心悸", "气短", "乏力", "失眠"],
        "润肺": ["干咳", "咽干", "咳嗽", "少痰"],
        "宣肺": ["咳嗽", "鼻塞", "无汗"],
        "清肺": ["咳嗽", "痰黄", "发热", "咽喉肿痛"],
        "疏肝": ["胁痛", "胁胀", "善太息", "抑郁", "易怒"],
        "平肝": ["头晕", "头痛", "眩晕", "耳鸣"],
        "潜阳": ["头晕", "头痛", "眩晕", "耳鸣"],
        "理气": ["胸闷", "胁胀", "腹胀", "胃胀", "嗳气"],
        "行气": ["胸闷", "胁痛", "腹胀", "胃痛"],
        "活血": ["胸痛", "胁痛", "痛经", "月经不调", "经行腹痛"],
        "化瘀": ["胸痛", "胁痛", "痛经", "经行腹痛"],
        "止血": ["便血", "牙龈出血", "鼻衄", "月经过多", "崩漏"],
        "散瘀": ["胸痛", "胁痛", "刺痛"],
        "祛瘀": ["胸痛", "胁痛", "月经不调"],
        "益气": ["乏力", "气短", "自汗", "神疲", "懒言"],
        "补气": ["乏力", "气短", "自汗", "神疲", "面色苍白"],
        "养血": ["面色萎黄", "心悸", "头晕", "失眠", "月经不调"],
        "补血": ["面色萎黄", "心悸", "头晕", "失眠", "月经不调"],
        "安神": ["失眠", "多梦", "心悸", "心烦", "健忘"],
        "开窍": ["神昏", "谵语"],
        "化痰": ["咳嗽", "咳痰", "痰多", "气喘"],
        "祛痰": ["咳嗽", "咳痰", "痰多"],
        "止咳": ["咳嗽"],
        "平喘": ["气喘", "哮喘"],
        "消食": ["食欲不振", "纳呆", "腹胀", "嗳气"],
        "导滞": ["腹胀", "便秘", "食欲不振"],
        "利水": ["水肿", "小便不利", "浮肿"],
        "通淋": ["小便不利", "小便涩痛", "小便短赤"],
        "退黄": ["皮肤发黄", "目黄"],
        "祛湿": ["便溏", "水肿", "脘痞", "腹胀"],
        "燥湿": ["便溏", "脘痞", "食欲不振"],
        "化湿": ["脘痞", "腹胀", "恶心", "食欲不振"],
        "祛风": ["头痛", "关节痛", "皮肤瘙痒", "恶寒"],
        "除痹": ["关节痛", "关节肿胀", "四肢麻木", "腰痛"],
        "通络": ["四肢麻木", "关节痛", "胸痛"],
        "止痛": ["头痛", "腹痛", "胸痛", "胁痛", "胃痛", "腰痛", "关节痛"],
        "散寒": ["恶寒", "畏寒", "腹痛", "手足不温"],
        "温中": ["腹痛", "腹泻", "畏寒", "食欲不振"],
        "降逆": ["恶心", "呕吐", "嗳气", "呃逆"],
        "解表": ["恶寒", "发热", "头痛", "鼻塞", "无汗"],
        "发汗": ["无汗", "恶寒", "发热"],
        "固表": ["自汗", "乏力", "气短"],
        "固精": ["遗尿", "夜尿多", "带下过多"],
        "缩尿": ["遗尿", "夜尿多", "小便频数"],
        "止带": ["带下过多"],
        "安胎": ["胎动不安"],
        "通便": ["便秘", "大便干结"],
        "润肠": ["便秘", "大便干结"],
        "泻下": ["便秘", "大便干结"],
        "消痈": ["疮疡", "痈肿"],
        "排脓": ["疮疡", "咳痰"],
        "明目": ["视物模糊", "目赤", "目涩", "目眩"],
        "利咽": ["咽喉肿痛", "咽干", "声音嘶哑"],
        "通窍": ["鼻塞", "头痛"],
        "醒神": ["神昏"],
        "息风": ["四肢抽搐", "头晕", "眩晕"],
        "定惊": ["四肢抽搐", "善惊", "心悸"],
        "截疟": ["寒战", "发热"],
        "驱虫": ["腹痛"],
        "降火": ["发热", "口渴", "心烦", "牙龈肿痛", "口疮"],
        "退虚热": ["五心烦热", "骨蒸发热", "盗汗", "手足心热"],
        "清心": ["心烦", "失眠", "口疮", "心悸"],
        "清肝": ["目赤", "头痛", "头晕", "胁痛", "易怒"],
        "清胃": ["牙龈肿痛", "口臭", "口渴", "胃痛"],
        "温肾": ["畏寒", "腰膝酸软", "夜尿多", "腰冷痛"],
        "补肺": ["气短", "咳嗽", "自汗", "乏力"],
        "疏风": ["头痛", "恶寒", "发热", "鼻塞"],
        "润燥": ["口干", "咽干", "皮肤干燥", "便秘"],
        "清暑": ["发热", "口渴", "心烦"],
        "化浊": ["脘痞", "恶心", "食欲不振"],
        "久咳": ["咳嗽"], "咳喘": ["咳嗽", "气喘"], "咳嗽": ["咳嗽"],
        "肺寒": ["咳嗽", "咳痰", "气喘"], "胃寒": ["胃痛", "腹痛", "呕吐"],
        "肝火": ["头痛", "目赤", "易怒", "口苦"], "心火": ["心烦", "口疮", "失眠"],
        "肺热": ["咳嗽", "痰黄", "咽喉肿痛"],
        "湿热": ["发热", "小便短赤", "带下色黄", "皮肤发黄"],
        "痰湿": ["咳痰", "痰多", "脘痞", "食欲不振"],
        "气虚": ["乏力", "气短", "自汗", "神疲"],
        "血虚": ["面色萎黄", "心悸", "头晕", "失眠"],
        "阴虚": ["口干", "盗汗", "五心烦热", "消瘦"],
        "阳虚": ["畏寒", "手足不温", "腰膝酸软"],
        "气滞": ["胸闷", "胁胀", "腹胀", "嗳气"],
        "血瘀": ["胸痛", "胁痛", "痛经", "月经不调"],
        "目赤": ["目赤", "目眵多", "目痛"],
        "鼻渊": ["鼻塞", "鼻流浊涕", "头痛"],
        "鼻鼽": ["鼻塞", "鼻流清涕", "鼻痒", "喷嚏"],
        "喉痹": ["咽喉肿痛", "咽干", "咽痒"],
        "口疮": ["口疮", "口臭"],
        "目痒": ["目痒", "目赤"],
        "耳疮": ["耳痛", "耳痒"],
        "斑疹": ["斑疹", "皮疹", "斑丘疹", "皮肤瘙痒"],
        "口糜": ["口疮", "口黏腻"],
        "口味异常": ["口甜", "口酸", "口咸", "口苦", "口淡", "口黏腻"],
    }

# ── 症状同义词映射（口语/近似表达→标准症状名） ──
_SYMPTOM_SYNONYM_MAP = {
    "疲倦": "乏力", "疲劳": "乏力", "没劲": "乏力", "无力": "乏力",
    "倦怠": "乏力", "浑身无力": "乏力",
    "面部浮肿": "浮肿", "脸肿": "浮肿", "眼皮肿": "浮肿",
    "面色晦暗": "面色萎黄", "脸色暗": "面色萎黄",
    "头皮麻木": "四肢麻木", "头重": "头重",
    "骨蒸发热": "五心烦热", "战汗": "自汗",
    "视物模糊": "目眩", "口渴不欲饮": "口干", "烦渴引饮": "口渴",
    "完谷不化": "腹泻", "大便失禁": "腹泻",
    "抑郁": "善太息", "善悲": "抑郁", "善恐": "善惊", "痴呆": "健忘",
    "月经先后无定期": "月经不调", "月经过少": "月经不调", "经期延长": "月经不调",
    "带下色黄": "带下过多", "带下色白": "带下过多", "带下赤白": "带下过多",
    "皮疹": "皮肤瘙痒", "风团": "皮肤瘙痒",
    "颈项强痛": "头痛", "关节变形": "关节痛",
    "胃部疼痛": "胃痛", "胃疼": "胃痛", "胃胀": "胃痛",
    "消谷善饥": "多食易饥", "偏嗜异物": "食欲不振",
    "痰白": "咳痰", "少腹痛": "腹痛", "小腹痛": "腹痛",
    "小便清长": "小便频数", "易醒": "失眠", "多梦": "失眠",
}

# ── 药材别名→正名映射（用于关联继承） ──
_HERB_ALIAS_MAP = {
    "杭菊花": "菊花", "甘菊花": "菊花", "毫菊花": "菊花",
    "黄菊花": "菊花", "白菊花": "菊花", "滁菊花": "菊花",
    "忍冬花": "金银花", "双花": "金银花", "二宝花": "金银花",
    "二花": "金银花", "银花": "金银花",
    "黄花地丁": "蒲公英", "婆婆丁": "蒲公英", "公英": "蒲公英",
    "甘枸杞": "枸杞子", "杞子": "枸杞子", "宁夏枸杞": "枸杞子", "枸杞": "枸杞子",
    "女贞实": "女贞子", "冬青子": "女贞子", "腊树子": "女贞子",
    "明天麻": "天麻", "赤箭": "天麻", "定风草": "天麻", "羊角天麻": "天麻",
    "赤芝": "灵芝", "灵芝草": "灵芝", "菌灵芝": "灵芝",
    "三七末": "三七", "山漆": "三七", "金不换": "三七", "田七": "三七",
    "薏米": "薏苡仁", "苡仁": "薏苡仁", "苡米": "薏苡仁",
    "米仁": "薏苡仁", "生薏仁": "薏苡仁", "炒薏仁": "薏苡仁", "麸炒薏仁": "薏苡仁",
    "制何首乌": "何首乌", "首乌": "何首乌",
    "北五味子": "五味子", "南五味子": "五味子", "华中五味子": "五味子",
    "川连": "黄连", "雅连": "黄连", "云连": "黄连", "鸡爪连": "黄连", "味连": "黄连",
    "坤草": "益母草", "茺蔚草": "益母草", "益母": "益母草",
    "光杏仁": "杏仁", "杏核仁": "杏仁", "北杏仁": "杏仁", "苦杏仁": "杏仁",
    "枝子": "栀子", "焦栀子": "栀子", "炒栀子": "栀子", "生栀子": "栀子",
    "杷叶": "枇杷叶", "炙枇杷叶": "枇杷叶", "蜜枇杷叶": "枇杷叶",
    "竹二青": "竹茹", "淡竹茹": "竹茹", "姜竹茹": "竹茹",
    "龙芽草": "仙鹤草", "脱力草": "仙鹤草", "石打穿": "仙鹤草",
    "菟丝实": "菟丝子", "吐丝子": "菟丝子", "炒菟丝子": "菟丝子", "盐菟丝子": "菟丝子",
    "国老": "甘草", "甜草": "甘草", "蜜甘草": "甘草", "粉甘草": "甘草", "甘草梢": "甘草",
    "野山参": "人参", "园参": "人参", "高丽参": "人参",
    "生晒参": "人参", "糖参": "人参", "白参": "人参", "吉林参": "人参",
    "当归须": "当归", "秦归": "当归",
    "艾草": "艾叶", "白芨": "白及", "丁香花": "丁香", "冬桑叶": "桑叶",
    "牡丹": "牡丹皮", "虫草": "冬虫夏草", "虫草花": "冬虫夏草",
    # 第二批次别名（补充）
    "九节菖蒲": "石菖蒲", "鲜菖蒲": "石菖蒲",
    "人参芦": "人参", "土人参叶": "人参",
    "黎芦": "藜芦", "藜芦（鲜）": "藜芦",
    "升药": "升药",
    "落地生根": "落地生根",
    "番木瓜": "番木瓜",
    "打破碗花花": "打破碗花花",
    "牛耳枫子": "牛耳枫子",
    "青礞石": "青礞石",
    "苦楝根皮": "苦楝根皮",
}

class TCMDataPipeline:
    """大规模中医药训练数据管线"""

    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self.rng = np.random.RandomState(RANDOM_SEED)

        # 核心数据
        self.herbs: dict[int, dict] = {}
        self.formulas: dict[int, dict] = {}
        self.symptoms: list[str] = []
        self.syndromes: dict[str, dict] = {}

        # 索引
        self.symptom_to_idx: dict[str, int] = {}
        self.herb_name_to_db_id: dict[str, int] = {}
        self.herb_db_id_to_model_idx: dict[int, int] = {}
        self.formula_db_id_to_model_idx: dict[int, int] = {}
        self.model_idx_to_herb_db_id: dict[int, int] = {}
        self.model_idx_to_formula_db_id: dict[int, int] = {}

        # 关联矩阵
        self.symptom_herb_matrix: Optional[np.ndarray] = None  # [num_s, num_h]
        self.symptom_formula_matrix: Optional[np.ndarray] = None  # [num_s, num_f]
        self.herb_cooccurrence: Optional[np.ndarray] = None  # [num_h, num_h]

        # 药材元数据
        self.herb_names: list[str] = []
        self.herb_categories: list[str] = []
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
        """增强版关联矩阵：文本匹配 + 方剂反推 + 分类语义"""
        logger.info("  构建增强关联矩阵...")
        ns, nh, nf = self.num_symptoms, self.num_herbs, self.num_formulas

        # ── 第一层：文本匹配 ──
        sym_herb = np.zeros((ns, nh), dtype=np.float32)
        for hid, herb in self.herbs.items():
            h_idx = self.herb_db_id_to_model_idx.get(hid)
            if h_idx is None:
                continue
            text = f"{herb.get('efficacy', '')} {herb.get('indications', '')}"
            for si, s_name in enumerate(self.symptoms):
                if s_name in text:
                    sym_herb[si, h_idx - 1] = 1.0
        text_match_count = int(sym_herb.sum())

        # ── 方剂匹配（用于反推） ──
        sym_formula = np.zeros((ns, nf), dtype=np.float32)
        for fid, formula in self.formulas.items():
            f_idx = self.formula_db_id_to_model_idx.get(fid)
            if f_idx is None:
                continue
            text = f"{formula.get('functions', '')} {formula.get('indications', '')}"
            for si, s_name in enumerate(self.symptoms):
                if s_name in text:
                    sym_formula[si, f_idx - 1] = 1.0

        # ── 第二层：方剂组成反推 ──
        formula_backprop = 0
        for fid, formula in self.formulas.items():
            f_idx = self.formula_db_id_to_model_idx.get(fid)
            if f_idx is None:
                continue
            comp = formula.get("composition", "")
            herb_idxs = self._parse_composition_indices(comp)
            if not herb_idxs:
                continue
            formula_symptoms = np.where(sym_formula[:, f_idx - 1] > 0)[0]
            for si in formula_symptoms:
                for hi in herb_idxs:
                    if sym_herb[si, hi] == 0:
                        sym_herb[si, hi] = 0.8
                        formula_backprop += 1

        logger.info("    文本匹配: %s 关联, 方剂反推: %s 新增",
                     text_match_count, formula_backprop)

        # ── 第三层：药材分类语义匹配 ──
        category_symptom_map = {
            "解表": ["恶寒", "发热", "头痛", "鼻塞", "无汗", "恶风", "喷嚏", "咳嗽"],
            "清热": ["发热", "口渴", "咽喉肿痛", "口干", "心烦", "面赤", "目赤"],
            "泻下": ["便秘", "大便干结", "腹胀", "腹痛"],
            "祛风湿": ["关节痛", "关节肿胀", "四肢麻木", "腰痛", "背痛"],
            "化湿": ["脘痞", "腹胀", "恶心", "食欲不振", "便溏"],
            "利水渗湿": ["水肿", "小便不利", "小便短赤", "浮肿", "小便涩痛"],
            "温里": ["畏寒", "手足不温", "腹痛", "腹泻", "面色苍白", "手足逆冷"],
            "理气": ["胸闷", "胁胀", "腹胀", "胃胀", "嗳气", "善太息", "乳房胀痛"],
            "消食": ["食欲不振", "纳呆", "腹胀", "嗳气", "多食易饥"],
            "止血": ["便血", "牙龈出血", "鼻衄", "月经过多", "崩漏"],
            "活血化瘀": ["胸痛", "胁痛", "痛经", "月经不调", "经行头痛", "经行乳房胀痛", "癃闭"],
            "化痰止咳": ["咳嗽", "咳痰", "气喘", "痰多", "痰黄", "喉中痰鸣"],
            "安神": ["失眠", "多梦", "心悸", "心烦", "健忘", "易惊"],
            "平肝息风": ["头晕", "头痛", "眩晕", "耳鸣", "四肢抽搐", "四肢麻木"],
            "开窍": ["神昏", "谵语", "狂躁"],
            "补气": ["乏力", "气短", "自汗", "神疲", "面色苍白", "懒言", "食欲不振"],
            "补阳": ["畏寒", "腰膝酸软", "夜尿多", "阳痿", "腰冷痛"],
            "补血": ["面色萎黄", "心悸", "失眠", "头晕", "月经不调", "月经后期"],
            "补阴": ["口干", "盗汗", "五心烦热", "手足心热", "消瘦", "颧红"],
            "收涩": ["自汗", "盗汗", "遗尿", "腹泻", "带下过多", "五更泄"],
            "驱虫": ["腹痛"],
            "攻毒杀虫": ["皮肤瘙痒", "疮疡", "痈肿"],
            "平肝": ["头晕", "头痛", "眩晕", "耳鸣"],
            "清热凉血": ["发热", "口渴", "皮肤发黄", "斑疹", "便血"],
        "清热燥湿": ["发热", "口苦", "小便短赤", "带下色黄"],
        "清热解毒": ["发热", "咽喉肿痛", "疮疡", "痈肿", "斑疹"],
        "清热泻火": ["发热", "口渴", "目赤", "口疮", "心烦"],
        "耳鼻喉": ["鼻塞", "鼻流清涕", "咽喉肿痛", "耳鸣", "耳痛"],
        "眼科": ["目赤", "目痛", "目痒", "目涩", "视物模糊"],
        "皮肤科": ["皮肤瘙痒", "皮疹", "斑疹", "疮疡"],
        "消肿": ["水肿", "浮肿"],
        "止痛": ["头痛", "腹痛", "胃痛", "胁痛", "腰痛", "关节痛", "痛经"],
        }

        cat_match = 0
        for hid, herb in self.herbs.items():
            h_idx = self.herb_db_id_to_model_idx.get(hid)
            if h_idx is None:
                continue
            cat = herb.get("category", "") or ""
            for cat_keyword, symptoms in category_symptom_map.items():
                if cat_keyword in cat:
                    for s_name in symptoms:
                        si = self.symptom_to_idx.get(s_name)
                        if si is not None and sym_herb[si, h_idx - 1] == 0:
                            sym_herb[si, h_idx - 1] = 0.6
                            cat_match += 1

        logger.info("    分类语义新增: %s 关联", cat_match)

        # ── 第四层：功能关键词→症状映射（对所有零关联目标做语义扩展） ──
        func_match_count = 0
        func_herb_count = 0
        
        # 对方剂：对仍然零症状关联的方剂做功能关键词匹配
        for fid, formula in self.formulas.items():
            f_idx = self.formula_db_id_to_model_idx.get(fid)
            if f_idx is None:
                continue
            # 只处理仍未关联任何症状的方剂
            if sym_formula[:, f_idx - 1].sum() > 0:
                continue
            text = f"{formula.get('functions', '')} {formula.get('indications', '')}"
            for kw, symptoms in _FUNCTION_SYMPTOM_MAP.items():
                if kw in text:
                    for s_name in symptoms:
                        si = self.symptom_to_idx.get(s_name)
                        if si is not None and sym_formula[si, f_idx - 1] == 0:
                            sym_formula[si, f_idx - 1] = 0.7
                            func_match_count += 1
        
        # 对药材：对仍未关联任何症状的药材做功能关键词匹配（不限文本长度）
        for hid, herb in self.herbs.items():
            h_idx = self.herb_db_id_to_model_idx.get(hid)
            if h_idx is None:
                continue
            # 只处理仍未关联任何症状的药材
            if sym_herb[:, h_idx - 1].sum() > 0:
                continue
            # 搜集所有可用文本
            text_parts = [
                herb.get('efficacy', '') or '',
                herb.get('indications', '') or '',
                herb.get('category', '') or '',
            ]
            nature = herb.get('nature', '') or ''
            flavor = herb.get('flavor', '') or ''
            # 从nature/flavor中提取关键性味词（寒热温凉平、酸苦甘辛咸）
            simple_attrs = []
            for ch in ['寒','热','温','凉','平','酸','苦','甘','辛','咸','淡','涩']:
                if ch in nature or ch in flavor:
                    simple_attrs.append(ch)
            text_parts.extend(simple_attrs)
            text = ' '.join(text_parts)
            
            for kw, symptoms in _FUNCTION_SYMPTOM_MAP.items():
                if kw in text:
                    for s_name in symptoms:
                        si = self.symptom_to_idx.get(s_name)
                        if si is not None and sym_herb[si, h_idx - 1] == 0:
                            sym_herb[si, h_idx - 1] = 0.5
                            func_herb_count += 1

        logger.info("    功能关键词新增(零关联补漏): 方剂=%s 关联, 药材=%s 关联",
                     func_match_count, func_herb_count)
        
        logger.info("    功能关键词新增: 方剂=%s 关联, 药材=%s 关联 (功效过短药材专用)",
                     func_match_count, func_herb_count)
        logger.info("    总关联: %s (密度 %.4f%%)",
                     int(sym_herb.sum()),
                     int(sym_herb.sum()) / (ns * nh) * 100)

        # ── 第五层：药材别名继承（别名→正名的关联传播） ──
        alias_count = 0
        for alias_name, primary_name in _HERB_ALIAS_MAP.items():
            # 找主药
            primary_db_id = self.herb_name_to_db_id.get(primary_name)
            if primary_db_id is None or primary_db_id not in self.herb_db_id_to_model_idx:
                continue
            primary_idx = self.herb_db_id_to_model_idx[primary_db_id] - 1
            # 找别名
            alias_db_id = self.herb_name_to_db_id.get(alias_name)
            if alias_db_id is None or alias_db_id not in self.herb_db_id_to_model_idx:
                continue
            alias_idx = self.herb_db_id_to_model_idx[alias_db_id] - 1
            # 传播：主药有什么症状关联，别名也继承（权重=0.7）
            primary_symptoms = np.where(sym_herb[:, primary_idx] > 0)[0]
            for si in primary_symptoms:
                if sym_herb[si, alias_idx] == 0:
                    sym_herb[si, alias_idx] = 0.7
                    alias_count += 1
        if alias_count > 0:
            logger.info("    别名继承: %s 关联", alias_count)
        
        # ── 第六层：症状同义词映射（扩展覆盖剩余症状） ──
        synonym_count = 0
        for si, s_name in enumerate(self.symptoms):
            # 只处理仍未关联任何药材的症状
            if sym_herb[si, :].sum() > 0:
                continue
            # 查找同义词→目标症状
            target_name = _SYMPTOM_SYNONYM_MAP.get(s_name)
            if target_name is None:
                continue
            target_si = self.symptom_to_idx.get(target_name)
            if target_si is not None:
                # 从目标症状复制关联
                source_assoc = sym_herb[target_si, :].copy()
                for hi in range(nh):
                    if source_assoc[hi] > 0 and sym_herb[si, hi] == 0:
                        sym_herb[si, hi] = source_assoc[hi] * 0.7  # 继承但权重略低
                        synonym_count += 1
        if synonym_count > 0:
            logger.info("    症状同义词: %s 关联（覆盖 %s 个症状）",
                         synonym_count, 
                         len([s for s in self.symptoms if s in _SYMPTOM_SYNONYM_MAP and np.count_nonzero(sym_herb[self.symptom_to_idx[s], :]) == 0]))

        # ── 第七层：硬编码兜底（对剩余零关联做最后补救） ──
        zero_herb_fallback = 0
        # 已知的零关联药材 → 应该关联的症状
        HERB_FALLBACK = {
            "番木瓜": ["食欲不振", "腹胀", "消化不良"],
            "落地生根": ["疮疡", "痈肿", "外伤"],
            "升药": ["疮疡", "痈肿", "溃烂"],
            "黎芦": ["头痛", "疟疾", "疥癣"],
            "打破碗花花": ["皮肤瘙痒", "疮疡", "疥癣"],
            "牛耳枫子": ["咳嗽", "便秘"],
            "九节菖蒲": ["神昏", "癫痫", "痰多"],
            "鲜菖蒲": ["神昏", "癫痫", "痰多"],
            "土人参叶": ["乏力", "气虚", "干咳"],
            "青礞石": ["咳痰", "气喘", "癫痫"],
            "苦楝根皮": ["腹痛", "蛔虫", "疥癣"],
            "人参芦": ["乏力", "虚脱"],
            "藜芦（鲜）": ["头痛", "疟疾", "疥癣"],
        }
        for hid, herb in self.herbs.items():
            h_idx = self.herb_db_id_to_model_idx.get(hid)
            if h_idx is None:
                continue
            if sym_herb[:, h_idx - 1].sum() > 0:
                continue
            hname = herb.get("name", "")
            if hname in HERB_FALLBACK:
                for s_name in HERB_FALLBACK[hname]:
                    si = self.symptom_to_idx.get(s_name)
                    if si is not None and sym_herb[si, h_idx - 1] == 0:
                        sym_herb[si, h_idx - 1] = 0.5
                        zero_herb_fallback += 1
        
        # 已知的零关联症状 → 应该关联的药材类别关键词
        SYMPTOM_FALLBACK = {
            "目眵多": ["目赤", "目痛"],
            "耳痒": ["耳鸣", "耳痛"],
            "鼻流清涕": ["鼻塞", "喷嚏"],
            "鼻流浊涕": ["鼻塞", "头痛"],
            "咽痒": ["咽喉肿痛", "咳嗽"],
            "口甜": ["口淡", "食欲不振"],
            "口酸": ["口苦", "胁痛"],
            "口咸": ["口干", "腰膝酸软"],
            "口黏腻": ["口淡", "食欲不振"],
            "斑丘疹": ["斑疹", "皮肤瘙痒"],
        }
        zero_sym_fallback = 0
        for si, s_name in enumerate(self.symptoms):
            if sym_herb[si, :].sum() > 0:
                continue
            if s_name in SYMPTOM_FALLBACK:
                for target_name in SYMPTOM_FALLBACK[s_name]:
                    target_si = self.symptom_to_idx.get(target_name)
                    if target_si is not None:
                        source = sym_herb[target_si, :].copy()
                        for hi in range(nh):
                            if source[hi] > 0 and sym_herb[si, hi] == 0:
                                sym_herb[si, hi] = source[hi] * 0.5
                                zero_sym_fallback += 1
        
        if zero_herb_fallback > 0 or zero_sym_fallback > 0:
            logger.info("    硬编码兜底: 药材=%s 关联, 症状=%s 关联",
                         zero_herb_fallback, zero_sym_fallback)
        
        logger.info("    最终总关联: %s (密度 %.4f%%)",
                     int(sym_herb.sum()),
                     int(sym_herb.sum()) / (ns * nh) * 100)

        self.symptom_herb_matrix = sym_herb
        self.symptom_formula_matrix = sym_formula

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

    def _parse_composition_indices(self, composition: str) -> list[int]:
        """增强版组成解析：处理更多真实格式（带炮制/剂量/子串回退）"""
        if not composition or not composition.strip():
            return []
        import re
        
        text = composition
        text = text.replace("、", "，").replace("；", "，").replace(";", "，")
        text = re.sub(r"[（(][^)）]*[)）]", "", text)
        text = re.sub(r"\d+\s*[克g枚片个毫升ml两钱分斤只条粒]?[。．]?", "", text)
        text = re.sub(r"[.．～~]", "", text)
        text = re.sub(r"(先煎|后下|包煎|烊化|冲服|另煎|煎汤代水)", "", text)
        text = re.sub(r"[\u4e00-\u9fff]{2,4}汤\s*\+\s*[\u4e00-\u9fff]{2,4}汤", "", text)
        
        parts = [p.strip() for p in text.split("，") if p.strip()]
        indices = []
        
        PREFIX_MAP = {"制": "", "炙": "", "生": "", "炒": "", "煅": "", "鲜": "", "酒": "", "醋": "", "盐": "", "姜": "", "蜜": ""}
        
        for part in parts:
            if not part or "+" in part:
                continue
            name = part.strip()
            name = re.sub(r"\s+", "", name)
            
            hid = self.herb_name_to_db_id.get(name)
            if hid:
                if hid in self.herb_db_id_to_model_idx:
                    idx = self.herb_db_id_to_model_idx[hid] - 1
                    if idx not in indices:
                        indices.append(idx)
                continue
            
            base_name = name
            for prefix in PREFIX_MAP:
                if base_name.startswith(prefix) and len(base_name) > len(prefix) + 1:
                    base_name = base_name[len(prefix):]
                    break
            
            if base_name != name:
                hid = self.herb_name_to_db_id.get(base_name)
                if hid and hid in self.herb_db_id_to_model_idx:
                    idx = self.herb_db_id_to_model_idx[hid] - 1
                    if idx not in indices:
                        indices.append(idx)
                    continue
            
            if len(base_name) >= 2:
                for hname, hid in self.herb_name_to_db_id.items():
                    if base_name in hname or hname in base_name:
                        if hid in self.herb_db_id_to_model_idx:
                            idx = self.herb_db_id_to_model_idx[hid] - 1
                            if idx not in indices:
                                indices.append(idx)
                            break
        
        return indices

    # ═══════════════════════════════════════════════
    # 2. 样本生成
    # ═══════════════════════════════════════════════

    def generate_samples(self, target_count: int = 500000) -> dict[str, np.ndarray]:
        """主生成函数：生成 target_count 个训练样本"""
        logger.info("=" * 50)
        logger.info("开始生成 %s 个训练样本...", target_count)

        all_symptoms = []       # list[list[int]]
        all_herb_labels = []    # list[np.ndarray]
        all_formula_labels = [] # list[int]
        all_syndrome_labels = []# list[int]
        all_weights = []        # list[float]

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
            if len(related_s) < 1:
                continue

            h_idx = h_model_idx - 1
            samples_per_herb = max(count // nh, 2)

            for _ in range(samples_per_herb):
                k = self.rng.randint(1, min(len(related_s) + 1, MAX_SYMPTOMS_PER_SAMPLE // 2))
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
                if max_k < 1:
                    continue
                k = self.rng.randint(1, max_k + 1)  # 允许1个症状
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

            # 方剂标签（加权随机采样：让低频方剂也有机会被选中）
            f_scores = self.symptom_formula_matrix[chosen].sum(axis=0)
            f_label = 0
            if f_scores.max() > 0:
                # 给零散关联一个基础权重，避免 argmax 垄断
                f_probs = f_scores.copy()
                # 对非零但较小的关联也给予采样机会
                nonzero_mask = f_probs > 0
                if nonzero_mask.any():
                    # 用 softmax 温度采样，温度越高分布越均匀
                    f_probs[nonzero_mask] = np.exp(f_probs[nonzero_mask] * 2)
                    f_probs = f_probs / f_probs.sum()
                    f_label = int(np.random.choice(len(f_probs), p=f_probs)) + 1

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

    def _infer_syndrome_from_symptoms(self, symptom_set: set[int]) -> int:
        """从症状集合推断最匹配证型（软分配：始终选匹配率最高的证型）"""
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
            if score > best_score:
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

    def build_herb_features(self) -> dict[str, np.ndarray]:
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

        herb_texts: dict[int, str] = {}
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
    parser.add_argument("--count", type=int, default=500000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    pipeline = TCMDataPipeline(args.db)
    pipeline.load_all()
    result = pipeline.generate_samples(args.count)
    pipeline.save(result, args.output)
    pipeline.close()


if __name__ == "__main__":
    main()
