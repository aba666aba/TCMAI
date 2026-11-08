# -*- coding: utf-8 -*-
"""词表漂移自适应补丁的功能验证（沙箱无 torch，用实现相同算术语义的假张量）。"""
import sys
import os
import types
from unittest.mock import MagicMock

# torch 桩（recommend.py 模块级 import torch）
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
sys.modules["numpy"] = MagicMock()
sys.modules["tqdm"] = MagicMock()
sys.modules["tqdm.auto"] = MagicMock()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tcm_recommender.recommend import _adapt_state_dict_for_drift


# ---- 假张量：实现补丁用到的 torch 语义 ----
class Fake1D:
    def __init__(self, data):
        self._d = list(data)
        self.shape = (len(self._d),)

    def dim(self):
        return 1

    def unsqueeze(self, i):
        assert i == 0
        return FakeT([self._d])

    def __iter__(self):
        return iter(self._d)


class FakeT:
    """二维张量替身：shape/dim/切片/赋值/mean(dim=0)/unsqueeze/expand。"""

    def __init__(self, data):
        self._d = [list(r) for r in data]
        self.shape = (len(self._d), len(self._d[0]) if self._d else 0)

    def dim(self):
        return 2

    def new_zeros(self, rows, cols):
        return FakeT([[0.0] * cols for _ in range(rows)])

    def __getitem__(self, key):
        if isinstance(key, slice):
            s, e, step = key.indices(len(self._d))
            return FakeT(self._d[s:e:step])
        return FakeT([self._d[key]])

    def __setitem__(self, key, other):
        if isinstance(key, slice):
            s, e, step = key.indices(len(self._d))
            rows = other._d if isinstance(other, FakeT) else other
            for i, r in enumerate(range(s, e, step)):
                self._d[r] = list(rows[i])
        else:
            self._d[key] = list(other._d[0])

    def mean(self, dim=0):
        assert dim == 0
        cols = self.shape[1]
        return Fake1D(
            [sum(r[c] for r in self._d) / len(self._d) for c in range(cols)]
        )

    def expand(self, rows, neg1):
        assert neg1 == -1
        return FakeT([self._d[0]] * rows)


failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(name)


print("== 词表漂移自适应：场景测试 ==")

# 场景 1：方剂 713→720（模拟真实漂移：训练时 712 方剂 + padding = 713）
old = FakeT([[0.0] * 384] + [[1.0 + r * 0.001] * 384 for r in range(1, 713)])
ref = FakeT([[0.0] * 384 for _ in range(720)])
state = {"formula_embedding.weight": old, "other.weight": None}
adapted = _adapt_state_dict_for_drift(state, {"formula_embedding.weight": ref})
new = state["formula_embedding.weight"]
check("场景1 触发自适应", adapted == ["formula_embedding.weight"])
check("场景1 形状 720×384", new.shape == (720, 384), str(new.shape))
check("场景1 旧行原样保留", new._d[100] == old._d[100])
check("场景1 padding 行=0", all(v == 0.0 for v in new._d[0]))
mean_row = (new._d[713],)
exp_mean = sum(old._d[i][0] for i in range(1, 713)) / 712
check("场景1 新增行=已训练行均值", abs(new._d[713][0] - exp_mean) < 1e-9,
      f"{new._d[713][0]:.6f} vs {exp_mean:.6f}")
check("场景1 所有新增行相同", new._d[719] == new._d[713])

# 场景 2：无漂移（形状一致）→ 不动
old2 = FakeT([[0.0] * 8] * 230)
state2 = {"symptom_encoder.embedding.weight": old2}
adapted2 = _adapt_state_dict_for_drift(
    state2, {"symptom_encoder.embedding.weight": FakeT([[0.0] * 8] * 230)}
)
check("场景2 无漂移不动原值", adapted2 == [] and state2["symptom_encoder.embedding.weight"] is old2)

# 场景 3：症状词表 224→230 漂移（含扩展配置更新场景）
old3 = FakeT([[0.0] * 384] + [[2.0] * 384 for _ in range(224)])
ref3 = FakeT([[0.0] * 384 for _ in range(230)])
state3 = {"symptom_encoder.embedding.weight": old3}
_adapted3 = _adapt_state_dict_for_drift(state3, {"symptom_encoder.embedding.weight": ref3})
check("场景3 症状词表漂移 225→230", state3["symptom_encoder.embedding.weight"].shape == (230, 384))
check("场景3 新症状行=均值", abs(state3["symptom_encoder.embedding.weight"]._d[229][0] - 2.0) < 1e-9)

# 场景 4：不兼容（dim1 变化）→ 跳过并保留原值，交由 strict load 报错
old4 = FakeT([[0.0] * 256 for _ in range(10)])
ref4 = FakeT([[0.0] * 384 for _ in range(10)])
state4 = {"some.weight": old4}
adapted4 = _adapt_state_dict_for_drift(state4, {"some.weight": ref4})
check("场景4 dim1 不兼容跳过", adapted4 == [] and state4["some.weight"] is old4)

# 场景 5：收缩（720→713，知识库删减实体）→ 复制重叠行
old5 = FakeT([[0.0] * 384] + [[1.0] * 384 for _ in range(719)])
ref5 = FakeT([[0.0] * 384 for _ in range(713)])
state5 = {"formula_embedding.weight": old5}
_adapted5 = _adapt_state_dict_for_drift(state5, {"formula_embedding.weight": ref5})
check("场景5 收缩自适应", state5["formula_embedding.weight"].shape == (713, 384))

print()
if failures:
    print(f"结果: {len(failures)} 项失败 -> {failures}")
    sys.exit(1)
print("结果: 全部通过 ✅（漂移自适应补丁逻辑与 torch 张量语义一致）")
