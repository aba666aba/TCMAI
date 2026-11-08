# -*- coding: utf-8 -*-
"""组方引擎 + 方剂组成解析的功能验证（真实 DB，torch/numpy 桩注入）。"""
import sys
import os
import types
from unittest.mock import MagicMock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

# ---- torch / numpy 桩 ----
torch = MagicMock()


class _Module:
    def __init__(self, *a, **k):
        pass


torch.nn.Module = _Module
torch.nn.Module.__init__ = lambda self, *a, **k: None
sys.modules["torch"] = torch
sys.modules["torch.nn"] = torch.nn
sys.modules["torch.nn.functional"] = torch.nn.functional
sys.modules["torch.utils"] = MagicMock()
sys.modules["torch.utils.data"] = MagicMock()


class FakeArr:
    def __init__(self, shape):
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)
        if len(self.shape) == 1:
            self._d = [0.0] * self.shape[0]
        else:
            self._d = [[0.0] * self.shape[1] for _ in range(self.shape[0])]

    def __setitem__(self, key, val):
        if len(self.shape) == 1:
            self._d[key] = float(val)
        else:
            r, c = key
            self._d[r][c] = float(val)


np_mod = types.ModuleType("numpy")
np_mod.zeros = lambda shape, dtype=None: FakeArr(shape)
np_mod.float32 = "float32"
np_mod.int32 = "int32"
np_mod.int64 = "int64"
np_mod.float64 = "float64"
np_mod.ndarray = object          # 类型注解在类定义时求值
np_mod.array = lambda *a, **k: None
np_mod.asarray = lambda *a, **k: None
sys.modules["numpy"] = np_mod
sys.modules["tqdm"] = MagicMock()
sys.modules["tqdm.auto"] = MagicMock()

failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(name)


DB = os.path.join(ROOT, "data", "tcm_knowledge.db")

# ============ 1. 组方引擎（真实药材药性数据） ============
print("== 1) src/rx/generator.py 组方引擎 ==")
from rx.generator import FormulaGenerator
from tcm_recommender.extended_config import pair_conflicts

gen = FormulaGenerator(DB)

# 场景 A：心脾两虚（库内定义含治法），候选含毒性药 + 十九畏对
cands = [
    {"name": "川乌头", "score": 0.99},   # 毒性（库内实际名）→ 必须被剔除
    {"name": "砒石", "score": 0.98},     # 毒性 → 必须被剔除
    {"name": "五灵脂", "score": 0.85},   # 十九畏：与人参相反
    {"name": "黄芪", "score": 0.80},
    {"name": "酸枣仁", "score": 0.75},
    {"name": "当归", "score": 0.70},
    {"name": "白术", "score": 0.60},
    {"name": "远志", "score": 0.50},
    {"name": "龙眼肉", "score": 0.40},
    {"name": "人参", "score": 0.90},     # 先入选 → 五灵脂应被跳过（十九畏）
    {"name": "甘草", "score": 0.20},     # 惯用使药
]
res = gen.generate(
    symptoms=["失眠", "乏力", "心悸"],
    herb_candidates=cands,
    syndrome_name="心脾两虚",
    matched_syndromes=[{"name": "心脾两虚", "description": "心脾两脏气血虚衰"}],
    alternative_formulas=["归脾丸", "养血安神片"],
)
comp = res["composition"]
text = res["text"]
print("---- 场景A 生成文本 ----")
print(text)
print("------------------------")
check("川乌头/砒石被毒性剔除", set(res["dropped_toxic"]) >= {"川乌头", "砒石"},
      str(res["dropped_toxic"]))
check("毒性药不入组成", "川乌头" not in comp and "砒石" not in comp, str(comp))
check("十九畏规避: 五灵脂未入组成(与人参)", "五灵脂" not in comp, str(comp))
check("跳过记录含五灵脂", any("五灵脂" == s[0] for s in res["skipped"]), str(res["skipped"]))
check("组成规模 4-7 味", 4 <= len(comp) <= 7, str(comp))
check("君药=补气类治法候选(黄芪/人参)", comp[0] in ("黄芪", "人参"), str(comp))
# 组成内两两无禁忌（审计自证）
no_conflict = all(
    not pair_conflicts(a, [x for x in comp if x != a]) for a in comp
)
check("组成审计: 两两无十八反/十九畏", no_conflict, str(comp))
for key in ("【AI 组方草案", "拟方组成", "方解:", "拟方主治", "剂量: AI 不代定",
            "配伍审计", "备选已入库方剂", "执业中医师审核"):
    check(f"文本含「{key}」", key in text)

# 场景 B：库内定义证型（气血两虚 → 治法益气补血 + 经典方八珍汤）
res_b = gen.generate(
    symptoms=["乏力", "头晕", "心悸"],
    herb_candidates=[{"name": n, "score": s} for n, s in
                     [("黄芪", 0.9), ("当归", 0.8), ("党参", 0.75), ("熟地黄", 0.7),
                      ("白术", 0.6), ("茯苓", 0.55), ("甘草", 0.5)]],
    syndrome_name="气血两虚",
    alternative_formulas=["八珍颗粒"],
)
check("场景B 治法 grounding: 文本含「益气补血」", "益气补血" in res_b["text"])
check("场景B 经典方备选: 八珍汤", "八珍汤" in res_b["text"])

# 场景 C：空候选 → 优雅报错
res_c = gen.generate(symptoms=["失眠"], herb_candidates=[{"name": "不存在的药", "score": 1}])
check("场景C 无有效候选优雅返回", res_c.get("error") and res_c["text"] == "")

# ============ 2. 方剂组成解析（真实 DB 覆盖率） ============
print()
print("== 2) data_pipeline.build_formula_composition（真实 DB） ==")
import importlib.util
import sqlite3

spec = importlib.util.spec_from_file_location(
    "dp_test", os.path.join(ROOT, "tcm_recommender", "data_pipeline.py")
)
dp_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dp_mod)

pipe = object.__new__(dp_mod.TCMDataPipeline)
con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row
pipe.db = con
pipe.formulas = {
    r["id"]: dict(r)
    for r in con.execute("SELECT id, name, composition FROM formulas")
}
pipe.num_formulas = len(pipe.formulas)
pipe.herbs = {r["id"]: dict(r) for r in con.execute("SELECT id, name FROM herbs")}
pipe.num_herbs = len(pipe.herbs)
fid_order = [r["id"] for r in con.execute("SELECT id FROM formulas")]
hid_order = [r["id"] for r in con.execute("SELECT id FROM herbs")]
pipe.model_idx_to_formula_db_id = {i: fid for i, fid in enumerate(fid_order, 1)}
pipe.model_idx_to_herb_db_id = {i: hid for i, hid in enumerate(hid_order, 1)}

comp_m = pipe.build_formula_composition()
check("组成矩阵形状 (719, 2961)", comp_m.shape == (719, 2961), str(comp_m.shape))

# 抽验已知组成：四君子汤（人参、白术、茯苓、甘草）应 ≥4 个成分命中
name_to_row = {fid_order[i]: i for i in range(len(fid_order))}
n_zjz = 0
for r in con.execute(
    "SELECT id, name, composition FROM formulas WHERE name LIKE '%四君子%'"
):
    row = name_to_row[r["id"]]
    n_zjz = sum(1 for c in comp_m._d[row] if c > 0)
    print(f"    四君子汤[{r['name']}] 组成命中 {n_zjz} 味（原文: {r['composition'][:40]}）")
check("四君子汤组成命中 ≥4", n_zjz >= 4, str(n_zjz))

n_gzt = 0
for r in con.execute(
    "SELECT id, name, composition FROM formulas WHERE name='桂枝汤'"
):
    row = name_to_row[r["id"]]
    n_gzt = sum(1 for c in comp_m._d[row] if c > 0)
    print(f"    桂枝汤 组成命中 {n_gzt} 味（原文: {r['composition'][:40]}）")
check("桂枝汤组成命中 ≥5", n_gzt >= 5, str(n_gzt))

empty_rows = sum(1 for row in comp_m._d if sum(1 for c in row if c > 0) == 0)
print(f"    组成全空方剂行数: {empty_rows}/719（composition 字段缺失所致，可接受）")
check("空组成行占比 < 15%", empty_rows / 719 < 0.15, str(empty_rows))

# ============ 3. v2 药材特征（multi-hot: 性/味/归经） + 剂量剥离 ============
print()
print("== 3) build_herb_features v2 multi-hot + 剂量剥离 ==")
pipe2 = object.__new__(dp_mod.TCMDataPipeline)
con2 = sqlite3.connect(DB)
con2.row_factory = sqlite3.Row
pipe2.db = con2
pipe2.herbs = {r["id"]: dict(r) for r in con2.execute("SELECT * FROM herbs")}
pipe2.num_herbs = len(pipe2.herbs)
hid_order = [r["id"] for r in con2.execute("SELECT id FROM herbs")]
pipe2.model_idx_to_herb_db_id = {i: hid for i, hid in enumerate(hid_order, 1)}

feat = pipe2.build_herb_features()
check("nature_multihot 维度 (2961,5)",
      feat["nature_multihot"].shape == (2961, 5), str(feat["nature_multihot"].shape))
check("meridian_multihot 维度 (2961,12)",
      feat["meridian_multihot"].shape == (2961, 12), str(feat["meridian_multihot"].shape))

name_to_row = {hid_order[i]: i for i in range(len(hid_order))}
danggui_id = con2.execute(
    "SELECT id FROM herbs WHERE name='当归'").fetchone()[0]
dg_row = name_to_row[danggui_id]
dg_flavors = feat["flavor_multihot"]._d[dg_row]
check("当归多味 multi-hot（甘+辛同时=1，修复单索引压缩）",
      sum(1 for v in dg_flavors if v > 0) >= 2, str(dg_flavors))
huangqi_id = con2.execute(
    "SELECT id FROM herbs WHERE name='黄芪'").fetchone()[0]
hq_row = name_to_row[huangqi_id]
hq_mer = feat["meridian_multihot"]._d[hq_row]
check("黄芪归经 multi-hot（肺/脾 ≥2 经）",
      sum(1 for v in hq_mer if v > 0) >= 2, str(hq_mer))

# 剂量剥离：插入测试方剂重跑解析
fid_test = max(fid_order) + 1
pipe.formulas[fid_test] = {
    "id": fid_test, "name": "__测试方__",
    "composition": "黄芩三两、三七、半夏半升、当归(酒洗)二钱",
}
pipe.num_formulas += 1
pipe.model_idx_to_formula_db_id[pipe.num_formulas] = fid_test
comp_t = pipe.build_formula_composition()
trow = pipe.num_formulas - 1  # 新行（0-based）

def _col(name):
    hid = con2.execute("SELECT id FROM herbs WHERE name=?", (name,)).fetchone()[0]
    return hid_order.index(hid)

check("剂量剥离: 『黄芩三两』→黄芩", comp_t._d[trow][_col("黄芩")] == 1.0)
check("三七不误伤（无单位不剥）", comp_t._d[trow][_col("三七")] == 1.0)
check("剂量剥离: 『半夏半升』→半夏", comp_t._d[trow][_col("半夏")] == 1.0)
check("括号炮制: 『当归(酒洗)二钱』→当归", comp_t._d[trow][_col("当归")] == 1.0)
con2.close()

print()
if failures:
    print(f"结果: {len(failures)} 项失败 -> {failures}")
    sys.exit(1)
print("结果: 全部通过 ✅")
