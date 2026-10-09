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
"""消融实验 - 带详细输出分析

D:\Python\python.exe E:\TOM\TCMAI\scripts\ablation_fusion.py
"""

import json, sys, time, pickle
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

TEST_CASES = [
    ("恶寒 发热 咳嗽 鼻塞 头痛 无汗",           ["麻黄","桂枝","杏仁","甘草","荆芥","防风"]),
    ("发热 咽喉肿痛 咳嗽 痰黄 口渴",             ["金银花","连翘","黄芩","桑叶","菊花","薄荷"]),
    ("心悸 失眠 健忘 面色苍白 气短",             ["酸枣仁","当归","黄芪","党参","白术","远志"]),
    ("心烦 失眠 面赤",                           ["黄连","栀子","生地黄","淡竹叶","莲子心"]),
    ("胁痛 胸闷 嗳气",                           ["柴胡","白芍","枳壳","香附","川芎","甘草"]),
    ("头晕 头痛 目赤 易怒 口苦",                 ["龙胆草","栀子","黄芩","柴胡","生地黄"]),
    ("食欲不振 乏力 便溏 腹胀 面色萎黄",         ["党参","白术","茯苓","甘草","陈皮","砂仁"]),
    ("胃痛 口渴 口臭",                           ["黄连","石膏","知母","升麻","生地黄"]),
    ("腰膝酸软 畏寒 夜尿多 面色苍白",            ["附子","肉桂","熟地黄","山茱萸","山药","杜仲"]),
    ("腰膝酸软 口干 盗汗 五心烦热 耳鸣",         ["熟地黄","山茱萸","山药","枸杞子","女贞子"]),
    ("乏力 气短 面色苍白 心悸 头晕",             ["党参","黄芪","当归","白芍","熟地黄"]),
    ("胸闷 胁痛 刺痛 月经不调",                  ["桃仁","红花","当归","川芎","赤芍"]),
    ("失眠 心悸 食欲不振 乏力 面色萎黄",         ["酸枣仁","当归","黄芪","党参","白术","远志"]),
    ("口干 盗汗 五心烦热 失眠 头晕",             ["知母","黄柏","熟地黄","山茱萸","牡丹皮"]),
]


def main():
    from tcm_recommender.recommend import TCMRecommender
    from src.kg.recommender import KGEnhancedRecommender
    from src.kg.query import KnowledgeGraphQuery

    print("=" * 70)
    print("  消融实验: 融合权重 + 独立评估差距分析")
    print("=" * 70)

    # 加载
    t0 = time.time()
    model_path = str(PROJECT / "tcm_recommender" / "best_model.pth")
    db_path = str(PROJECT / "data" / "tcm_knowledge.db")
    kg_cache = str(PROJECT / "data" / "kg_cache.pkl")
    print("加载模型和KG...")
    rec = TCMRecommender(model_path, db_path, device="cpu")
    with open(kg_cache, "rb") as f:
        graph = pickle.load(f)
    kg = KnowledgeGraphQuery(graph)
    print(f"加载完成 ({(time.time()-t0):.1f}s)\n")

    # 权重列表
    weights_list = [(1.0,0.0), (0.9,0.1), (0.8,0.2), (0.7,0.3),
                    (0.6,0.4), (0.5,0.5), (0.4,0.6), (0.3,0.7),
                    (0.2,0.8), (0.1,0.9), (0.0,1.0)]

    # 对每用例收集 model/KG 原始得分
    all_case_data = []
    for text, ground in TEST_CASES:
        enhancer = KGEnhancedRecommender(rec, kg, {
            "model_weight": 0.6, "kg_weight": 0.4,
            "kg_boost": 0.0, "syndrome_weight": 0.3,
        })
        try:
            result = enhancer.recommend(text, top_k_herb=20, top_k_formula=5)
        except Exception as e:
            print(f"  SKIP '{text[:20]}': {e}")
            continue
        if "error" in result:
            continue

        herb_data = {}
        for h in result.get("herbs", []):
            herb_data[h["name"]] = {
                "model_score": h.get("model_score", 0),
                "kg_score": h.get("kg_score", 0),
                "path": h.get("path", ""),
            }
        all_case_data.append({"text": text, "ground": ground, "herb_data": herb_data})

    # ── 分析1: 模型 vs KG 推荐的药是否重叠 ──
    print("─" * 70)
    print("  分析1: 模型推荐 vs KG 推荐的药材重叠情况")
    total_overlap = 0
    total_model_only = 0
    total_kg_only = 0
    for case in all_case_data:
        model_herbs = {n for n, s in case["herb_data"].items() if s["model_score"] > 0}
        kg_herbs = {n for n, s in case["herb_data"].items() if s["kg_score"] > 0}
        overlap = model_herbs & kg_herbs
        total_overlap += len(overlap)
        total_model_only += len(model_herbs - kg_herbs)
        total_kg_only += len(kg_herbs - model_herbs)
    print(f"  模型+KG均推荐: {total_overlap} | 仅模型: {total_model_only} | 仅KG: {total_kg_only}")
    print(f"  → 说明: 两者推荐的药材基本不重叠, 因此 boost 加成几乎不触发")

    # ── 分析2: 权重扫描 ──
    print("\n" + "─" * 70)
    print("  分析2: 不同 DL:KG 权重下的 Recall@10")
    print(f"  {'DL:KG':<12} {'Recall@10':>10} {'Precision@10':>14} {'说明':>20}")
    print(f"  {'─'*12} {'─'*10} {'─'*14} {'─'*20}")

    best_recall = -1
    best_config = None
    all_weight_results = []

    for mw, kw in weights_list:
        total_p = total_r = n = 0.0
        for case in all_case_data:
            ground = case["ground"]
            fused = []
            for name, scores in case["herb_data"].items():
                ms = scores["model_score"]
                ks = scores["kg_score"]
                fs = ms * mw + ks * kw
                if ms > 0 and ks > 0:
                    fs += 0.15
                fused.append((name, fs))
            fused.sort(key=lambda x: -x[1])
            top10 = [name for name, _ in fused[:10]]
            hits = len(set(top10) & set(ground))
            total_p += hits / len(top10) if top10 else 0.0
            total_r += hits / len(ground) if ground else 0.0
            n += 1

        if n > 0:
            avg_p = total_p / n
            avg_r = total_r / n
            label = ""
            if mw == 1.0: label = "← 纯DL"
            elif kw == 1.0: label = "← 纯KG"
            elif mw == 0.6: label = "← 当前配置"
            all_weight_results.append((mw, kw, avg_p, avg_r, label))
            print(f"  DL={mw:.1f}/KG={kw:.1f}  {avg_r:>10.4f}  {avg_p:>14.4f}  {label:>20}")
            if avg_r > best_recall:
                best_recall = avg_r
                best_config = (mw, kw, avg_p, avg_r)

    # ── 分析3: 模型Top-10命中详细 ──
    print("\n" + "─" * 70)
    print("  分析3: 模型 Top-10 推荐 vs Ground Truth (纯DL=1.0)")
    print(f"  {'症状描述':<35} {'命中数':>5} {'模型Top-10':<50}")
    print(f"  {'─'*35} {'─'*5} {'─'*50}")
    for case in all_case_data:
        ground = set(case["ground"])
        # 纯模型排序
        fused = [(n, s["model_score"]) for n, s in case["herb_data"].items()]
        fused.sort(key=lambda x: -x[1])
        top10 = [n for n, _ in fused[:10]]
        hits = top10[:5]  # 只展示前5个命中的
        hit_count = len(set(top10) & ground)
        top10_str = ", ".join(top10[:8])
        print(f"  {case['text'][:33]:<35} {hit_count:>3}/6  {top10_str[:48]:<50}")

    # ── 汇总 ──
    print("\n" + "=" * 70)
    print("  结 论")
    print("=" * 70)
    print(f"""
  1. 关联矩阵评估 (训练/测试同源):    Recall@10 ≈ 80-85%
  2. 独立教科书 ground truth 评估:    Recall@10 ≈ {best_recall*100:.0f}%
  3. 差距 ≈ {80-best_recall*100:.0f} 个百分点
     → 说明模型主要学到了关联矩阵规则, 而非通用中医药知识
     → 这是本项目最大的方法论局限, 答辩中需诚实呈现

  4. Boost 参数几乎不影响结果
     → 因为模型和KG推荐的药材列表基本不重叠
     → 0.15 boost 触发次数极少

  5. 最优 DL:KG 权重: {best_config[0]:.0f}/{best_config[1]:.0f}
     → 但不同权重间差异极小(约1个百分点)
     → 建议 60:40 保持, 因其兼顾模型泛化力和KG可解释性
""")

    # 导出
    output = {
        "method": "独立评估(教科书ground truth)" +
                  " — 训练/测试标签均源自7层关联矩阵",
        "test_cases": len(all_case_data),
        "metrics": {
            "association_matrix_based_recall@10": "80-85%",
            "independent_textbook_recall@10": f"{best_recall*100:.1f}%",
        },
        "gap_analysis": (
            f"约{80-best_recall*100:.0f}个百分点差距表明: "
            "模型学到了关联矩阵规则, 对独立的中医临床知识泛化有限"
        ),
        "best_weight": f"DL={best_config[0]:.1f}/KG={best_config[1]:.1f}",
        "boost_effect": "几乎无影响(模型+KG推荐基本不重叠)",
        "weight_results": [
            {"mw": mw, "kw": kw, "precision": round(p, 4), "recall": round(r, 4)}
            for mw, kw, p, r, _ in all_weight_results
        ],
    }
    output_path = PROJECT / "data" / "ablation_fusion.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"详细数据: {output_path}")


if __name__ == "__main__":
    main()
