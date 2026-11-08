# -*- coding: utf-8 -*-
"""v2 新模型加载 + 推理冒烟测试（D:\\Python，含 torch）。"""
import sys
import os

ROOT = "E:/AAATOM/NewLai/TCMAI"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from tcm_recommender.recommend import TCMRecommender

MODEL = os.path.join(ROOT, "tcm_recommender", "best_model_v2.pth")
DB = os.path.join(ROOT, "data", "tcm_knowledge.db")

failures = []
def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(name)

print("== 1) 加载 v2 checkpoint ==")
r = TCMRecommender(MODEL, DB, device="cpu")
comp = getattr(r.model, "comp_encoder", None)
check("组合式方剂编码器已启用", comp is not None)
check("症状位置编码已移除", "pos_encoder" not in dir(r.model.symptom_encoder))
check("v2 属性塔(multi-hot)", hasattr(r.model.herb_encoder, "meridian_proj"))

print()
print("== 2) 推理测试 ==")
cases = [
    ("失眠 心悸 乏力", "归脾汤"),
    ("胸痛 胸闷 心悸", "血府逐瘀汤"),
    ("眩晕 耳鸣", "天麻钩藤饮"),
    ("食欲不振 腹胀", "二陈汤"),
]
for symptoms, expect in cases:
    sids, valid, _ = r.parse_symptoms(symptoms)
    result = r.recommend(sids, top_k_herb=10, top_k_formula=5)
    herbs = [h["name"] for h in result["herbs"]]
    forms = [f["name"] for f in result["formulas"]]
    scores = [h["score"] for h in result["herbs"][:5]]
    syn = (result.get("syndrome") or {}).get("name", "无")
    print(f"\n  症状: {symptoms}")
    print(f"    证型: {syn}")
    print(f"    药材 top5: {herbs[:5]}")
    print(f"    药材分数: {[round(s,3) for s in scores]}")
    print(f"    方剂 top3: {forms[:3]}")
    check(f"[{symptoms}] 输出非空", len(herbs) > 0 and len(forms) > 0)
    check(f"[{symptoms}] 推荐含 {expect}", expect in herbs or expect in forms,
          f"herbs={herbs[:5]}, forms={forms[:3]}")
    # 分数饱和为已知遗留问题（hinge 对已分离 logits 无梯度），此处验证排序有效性
    check(f"[{symptoms}] 分数排序有效(降序)",
          all(scores[i] >= scores[i+1] - 1e-9 for i in range(len(scores)-1)),
          str([round(s, 3) for s in scores]))

print()
print("== 3) 组方引擎联测（v2 候选） ==")
from src.rx.generator import FormulaGenerator
gen = FormulaGenerator(DB)
res = gen.generate(
    symptoms=["失眠", "心悸", "乏力"],
    herb_candidates=[{"name": h["name"], "score": h["score"]} for h in r.recommend(
        r.parse_symptoms("失眠 心悸 乏力")[0], top_k_herb=10, top_k_formula=5)["herbs"]],
    syndrome_name=(r.recommend(r.parse_symptoms("失眠 心悸 乏力")[0],
                               top_k_herb=10, top_k_formula=5).get("syndrome") or {}).get("name"),
)
print(res["text"][:400])
check("组方草案生成", bool(res["text"]))
check("组成无配伍冲突", not res["conflicts"])

print()
if failures:
    print(f"结果: {len(failures)} 项失败 -> {failures}")
    sys.exit(1)
print("结果: 全部通过 ✅")
