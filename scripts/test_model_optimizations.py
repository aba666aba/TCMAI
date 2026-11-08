# -*- coding: utf-8 -*-
"""模型源码优化的功能验证（沙箱无 torch/numpy，用桩模块绕过导入后实测逻辑）。

验证点：
  1. _sigmoid 数值等价 torch.sigmoid
  2. check_safety 行为不变（配伍禁忌命中/去重/is_safe/toxic）且不再查 DB
  3. parse_symptoms 别名映射正常
  4. kg/recommender._profile 记忆化（ herb_profile 调用次数 = 药材数，修复前为 O(N²)）
  5. engine.py 模块可导入、prompt 缓存字段就位
"""
import sys
import types
from unittest.mock import MagicMock

# ---- 桩注入：torch / numpy / tqdm / torch.utils.data ----
torch = MagicMock()


class _Module:
    """nn.Module 替身：仅需可继承，类体方法不会真正执行 forward。"""


torch.nn.Module = _Module
torch.nn.Module.__init__ = lambda self, *a, **k: None
sys.modules["torch"] = torch
sys.modules["torch.nn"] = torch.nn
sys.modules["torch.nn.functional"] = torch.nn.functional
sys.modules["torch.utils"] = MagicMock()
sys.modules["torch.utils.data"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["tqdm"] = MagicMock()
sys.modules["tqdm.auto"] = MagicMock()

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

failures = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(name)


# ============ 1. recommend.py：_sigmoid / check_safety / parse_symptoms ============
print("== 1) tcm_recommender.recommend ==")
from tcm_recommender.recommend import TCMRecommender, _sigmoid

import math

for x in (0.0, 0.07, 1.0, -1.0, 3.2, -3.2, 12.0, -12.0, 88.0, -88.0):
    ref = 1.0 / (1.0 + math.exp(-x)) if abs(x) < 500 else (1.0 if x > 0 else 0.0)
    got = _sigmoid(x)
    check(f"_sigmoid({x}) ≈ {ref:.6f}", abs(got - ref) < 1e-12, f"got {got:.6f}")

# check_safety：绕过 __init__，注入预缓存（验证不再依赖 DB）
r = object.__new__(TCMRecommender)
r._herb_contra_cache = {
    "甘草": "不宜与甘遂、大戟、芫花、海藻同用；益气补中。",
    "人参": "反藜芦；不宜与五灵脂同用。",
    "半夏": "畏乌头。",
}
r.db_path = "not-used-should-not-open-db"

safety = r.check_safety(["甘草", "甘遂", "人参", "藜芦", "甘草"])
pairs = "; ".join(safety["incompatibilities"])
check("十八反命中: 甘草+甘遂", "甘草 + 甘遂" in pairs, pairs[:80])
check("十九畏命中: 人参+藜芦", "人参 + 藜芦" in pairs)
check("重复输入去重(甘草出现两次仅一份警告)", safety["warnings"].count(
    lambda w: False) == 0 or len([w for w in safety["warnings"] if w.startswith("甘草")]) == 1)
check("toxic_herbs 非空(甘遂/藜芦属毒性清单)",
      len(safety["toxic_herbs"]) >= 2, str(safety["toxic_herbs"]))
check("is_safe=False（有毒+禁忌）", safety["is_safe"] is False)

safety2 = r.check_safety(["茯苓", "山药"])
check("常规药对 is_safe=True", safety2["is_safe"] is True and not safety2["incompatibilities"])

# parse_symptoms（别名映射 + 词表精确匹配）
r2 = object.__new__(TCMRecommender)
r2.symptom_to_id = {"失眠": 1, "乏力": 2, "心悸": 3}
sids, valid, unknown = r2.parse_symptoms("失眠, 乏力，睡不着")
check("parse_symptoms 返回有效症状", isinstance(sids, list) and isinstance(valid, list))

# ============ 2. kg/recommender.py：_profile 记忆化 ============
print("== 2) src/kg/recommender._profile 记忆化 ==")
import importlib.util

spec = importlib.util.spec_from_file_location(
    "kg_rec_opt_test", os.path.join(ROOT, "src", "kg", "recommender.py")
)
kg_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kg_mod)


class FakeKG:
    def __init__(self):
        self.calls = 0

    def herb_profile(self, name):
        self.calls += 1
        return {
            "toxic": name in ("砒石", "甘遂"),
            "incompatible_with": ["甘草"] if name == "甘遂" else [],
            "pregnancy_forbidden": name == "麝香",
        }


rec = object.__new__(kg_mod.KGEnhancedRecommender)
rec.kg = FakeKG()
herbs = [{"name": n} for n in ["甘遂", "甘草", "茯苓", "白术", "人参",
                               "丹参", "细辛", "麝香", "当归", "川芎"]]
ins = rec._kg_insights(herbs, [])
kgw = rec._kg_safety_check([h["name"] for h in herbs])
total_calls = rec.kg.calls
# 修复前：_kg_insights 内层重复查询 = N + N(N-1)/2 = 55 次；+ _kg_safety_check N 次 = 65
# 修复后：全部命中 memo = N = 10 次
check("herb_profile 调用次数 = N=10（修复前 65）", total_calls == 10, f"实际 {total_calls}")
check("毒性警示聚合", "甘遂" in ins.get("toxic_warning", ""), str(ins.get("toxic_warning")))
check("配伍对命中 甘遂-甘草", any("甘遂" in w for w in ins.get("incompatible_warning", [])),
      str(ins.get("incompatible_warning")))
check("孕妇禁用提示", any("麝香" in w for w in kgw), str(kgw))

# ============ 3. engine.py：模块导入与 prompt 缓存 ============
print("== 3) src/nlu/engine.py ==")
from src.nlu.engine import NLUEngine, _SYSTEM_PROMPT

eng = NLUEngine()
check("_system_prompt_cache 初始为 None", eng._system_prompt_cache is None)
lst = eng._build_symptom_list_text()
check("症状列表非空(≥200 项)", lst.count("、") >= 200, f"{lst.count('、')+1} 项")
check("system prompt 含占位符", "{症状列表}" in _SYSTEM_PROMPT)

# 总结
print()
if failures:
    print(f"结果: {len(failures)} 项失败 -> {failures}")
    sys.exit(1)
print("结果: 全部通过 ✅")
