# Copyright (C) 2026 中医药智能问答系统 (TCMAI)
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
"""NLU 量化精度测量 — float16 vs 4-bit 症状提取一致率

运行方式:
  D:\Python\python.exe E:\TOM\TCMAI\scripts\ablation_nlu_quant.py

测试内容:
  1. 100条口语化症状描述 + 人工标注标准答案
  2. float16 全精度 vs 4-bit 量化的输出对比
  3. 精确匹配率 / 部分匹配率 / F1
"""

import json
import logging
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("nlu_ablation")

# ── 100 条测试用例：口语化表述 → 标准症状（人工标注） ──
TEST_CASES = [
    # 睡眠相关
    ("睡不着觉", ["失眠"]),
    ("晚上翻来覆去睡不着", ["失眠"]),
    ("入睡困难，好不容易睡着了又容易醒", ["失眠"]),
    ("最近老是失眠多梦", ["失眠", "多梦"]),
    ("梦特别多，一晚上做很多梦", ["多梦"]),
    ("睡不安稳，半夜总醒", ["失眠"]),
    ("躺床上很久才能睡着", ["失眠"]),
    ("天没亮就醒了再也睡不着", ["失眠"]),
    # 头部
    ("头疼得厉害", ["头痛"]),
    ("太阳穴跳着疼", ["头痛"]),
    ("整个脑袋昏昏沉沉的", ["头晕"]),
    ("站起来眼前发黑", ["头晕"]),
    ("天旋地转的感觉", ["眩晕"]),
    ("头胀，感觉脑袋要炸了", ["头痛"]),
    # 消化
    ("吃不下饭", ["食欲不振"]),
    ("没胃口，什么都不想吃", ["食欲不振"]),
    ("吃完饭肚子胀", ["腹胀"]),
    ("胃里反酸水", ["反酸"]),
    ("老打嗝，停不下来", ["嗳气"]),
    ("肚子咕噜咕噜响，拉稀", ["腹泻"]),
    ("大便不成形，稀稀拉拉的", ["便溏"]),
    ("好几天没解大便了", ["便秘"]),
    ("胃隐隐作痛", ["胃痛"]),
    ("吃一点就饱了", ["食欲不振"]),
    ("恶心想吐", ["恶心"]),
    ("吐了两次", ["呕吐"]),
    # 全身
    ("浑身没劲", ["乏力"]),
    ("走几步路就喘不上气", ["气短"]),
    ("总觉得累，不想动", ["乏力"]),
    ("稍微活动一下就出汗", ["自汗"]),
    ("晚上睡觉出一身汗", ["盗汗"]),
    ("手心脚心发热", ["五心烦热"]),
    ("身上一阵冷一阵热", ["寒热往来"]),
    ("瘦了很多", ["消瘦"]),
    # 呼吸/胸
    ("咳嗽，有黄痰", ["咳嗽", "咳痰"]),
    ("干咳，没痰", ["干咳"]),
    ("喘不上气，胸闷", ["胸闷", "气短"]),
    ("胸口闷得慌", ["胸闷"]),
    ("心慌，心跳得特别快", ["心悸"]),
    ("胸口刺痛", ["胸痛"]),
    # 冷热
    ("特别怕冷", ["畏寒"]),
    ("手脚冰凉", ["手足不温"]),
    ("发烧了，体温很高", ["发热"]),
    ("总觉得口干舌燥", ["口干"]),
    ("老想喝水，口特别干", ["口渴"]),
    ("嗓子疼，咽东西都疼", ["咽喉肿痛"]),
    # 腰/关节
    ("腰酸得直不起来", ["腰膝酸软"]),
    ("膝盖疼，上下楼费劲", ["腰膝酸软"]),
    ("腿肿了", ["水肿"]),
    ("浑身关节疼", ["关节痛"]),
    # 泌尿/生殖
    ("晚上老起来上厕所", ["夜尿多"]),
    ("小便黄，量少", ["小便黄"]),
    ("月经不调，有时候两个月才来", ["月经不调"]),
    ("来月经肚子疼", ["痛经"]),
    # 精神
    ("心烦意乱，坐立不安", ["心烦"]),
    ("脾气暴躁，一点就着", ["易怒"]),
    ("记性越来越差", ["健忘"]),
    ("总是闷闷不乐", ["抑郁"]),
    ("容易紧张害怕", ["惊悸"]),
    # 五官
    ("眼睛发红，很多眼屎", ["目赤"]),
    ("眼花，看东西模糊", ["视物模糊"]),
    ("耳朵嗡嗡响", ["耳鸣"]),
    ("鼻塞不通气", ["鼻塞"]),
    ("流清鼻涕", ["流涕"]),
    ("牙龈出血", ["牙龈出血"]),
    ("口苦，吃什么都没味道", ["口苦"]),
    ("口臭，自己都能闻到", ["口臭"]),
    # 皮肤
    ("身上起了很多红疹子", ["皮疹"]),
    ("皮肤特别痒", ["皮肤瘙痒"]),
    ("脸色发黄", ["面色萎黄"]),
    ("脸色苍白没血色", ["面色苍白"]),
    # 复合
    ("失眠多梦，白天头晕没精神", ["失眠", "多梦", "头晕", "乏力"]),
    ("胃胀不想吃东西还便秘", ["腹胀", "食欲不振", "便秘"]),
    ("咳嗽有黄痰还发烧", ["咳嗽", "咳痰", "发热"]),
    ("腰酸腿软怕冷", ["腰膝酸软", "畏寒"]),
    ("口干想喝水，手心脚心发热，晚上出汗", ["口干", "口渴", "五心烦热", "盗汗"]),
    ("心慌气短，走几步就喘，没什么力气", ["心悸", "气短", "乏力"]),
    ("头疼发烧怕冷没汗", ["头痛", "发热", "畏寒"]),
    ("头晕耳鸣腰酸", ["头晕", "耳鸣", "腰膝酸软"]),
    ("肚子胀不消化，老反酸水", ["腹胀", "反酸"]),
    ("嗓子疼发烧咳嗽", ["咽喉肿痛", "发热", "咳嗽"]),
    # 边界用例
    ("最近压力大", []),  # 不是症状，期望空输出
    ("感觉身体被掏空", ["乏力"]),  # 网络用语
    ("整个人都不好了", []),  # 极口语化，不应输出
    ("亚健康状态", []),  # 西医概念，不应输出
    ("湿气重", ["便溏", "水肿", "乏力"]),  # 中医口语概念
    ("上火了", ["发热", "咽喉肿痛", "口干"]),  # 模糊统称
    ("昨天打球扭到脚了", ["关节痛"]),
    ("熬夜后第二天头昏脑胀", ["头晕", "乏力"]),
    ("喝多了酒胃难受", ["胃痛", "恶心"]),
    ("吃了不干净东西拉肚子", ["腹泻"]),
]


def test_nlu_engine(engine, test_cases, label=""):
    """用指定的 NLU 引擎跑全部测试用例"""
    results = []
    for text, expected in test_cases:
        try:
            output = engine.extract_symptoms(text)
            if isinstance(output, list):
                predicted = output
            else:
                predicted = []
        except Exception:
            predicted = []
        results.append({"text": text, "expected": expected, "predicted": predicted})
    return results


def compute_metrics(results):
    """计算精确匹配率 / 部分匹配率 / F1 / 召回"""
    exact_match = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    at_least_one_match = 0
    n_non_empty_exp = 0  # 期望非空的样本数

    for r in results:
        expected = set(r["expected"])
        predicted = set(r["predicted"])

        # 精确匹配
        if expected == predicted:
            exact_match += 1

        # 至少命中一个
        if expected:
            n_non_empty_exp += 1
            if expected & predicted:
                at_least_one_match += 1

        # TP/FP/FN
        tp = len(expected & predicted)
        fp = len(predicted - expected)
        fn = len(expected - predicted)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "exact_match_rate": round(exact_match / len(results), 4),
        "at_least_one_hit_rate": round(at_least_one_match / n_non_empty_exp, 4) if n_non_empty_exp else 0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total_samples": len(results),
    }


def main():
    from src.nlu.engine import NLUEngine

    print("NLU 量化精度测量")
    print(f"测试用例: {len(TEST_CASES)} 条")
    print()

    # ── float16 全精度 ──
    print("[1/2] 加载 float16 全精度 NLU 引擎...")
    try:
        engine_fp16 = NLUEngine(device="cpu", use_4bit=False)
        engine_fp16.load()
        print("  float16 引擎加载完成")
        fp16_available = True
    except Exception as e:
        print(f"  float16 引擎加载失败: {e}")
        fp16_available = False

    # ── 4-bit 量化 ──
    print("\n[2/2] 加载 4-bit 量化 NLU 引擎...")
    try:
        import torch
        if torch.cuda.is_available():
            engine_4bit = NLUEngine(device="cuda", use_4bit=True)
            engine_4bit.load()
            print("  4-bit (CUDA) 引擎加载完成")
            quant_available = True
            quant_mode = "cuda_4bit"
        else:
            print("  CUDA 不可用, 无法测试 4-bit 量化版本")
            quant_available = False
            quant_mode = "n/a"
    except Exception as e:
        print(f"  4-bit 引擎加载失败: {e}")
        quant_available = False
        quant_mode = "n/a"

    # ── 运行测试 ──
    print("\n" + "=" * 60)
    if fp16_available:
        print("运行 float16 测试...")
        results_fp16 = test_nlu_engine(engine_fp16, TEST_CASES, "float16")
        metrics_fp16 = compute_metrics(results_fp16)
    else:
        results_fp16 = []
        metrics_fp16 = {}

    if quant_available:
        print("运行 4-bit 测试...")
        results_4bit = test_nlu_engine(engine_4bit, TEST_CASES, "4bit")
        metrics_4bit = compute_metrics(results_4bit)
    else:
        results_4bit = []
        metrics_4bit = {}

    # ── 输出 ──
    print("\n" + "=" * 60)
    print("  结果对比")
    print("=" * 60)
    print(f"\n{'指标':<20} {'float16':>12} {'4-bit':>12} {'差异':>10}")
    print(f"{'─'*20} {'─'*12} {'─'*12} {'─'*10}")
    for key in ["exact_match_rate", "at_least_one_hit_rate", "precision", "recall", "f1"]:
        v16 = metrics_fp16.get(key, 0)
        v4 = metrics_4bit.get(key, 0)
        diff = v4 - v16
        sign = "+" if diff > 0 else ""
        print(f"{key:<20} {v16:>12.4f} {v4:>12.4f} {sign}{diff:>9.4f}")

    # ── 详细错误分析（两版不一致的用例） ──
    if fp16_available and quant_available:
        print(f"\n{'─'*60}")
        print("  输出不一致的用例:")
        print(f"  ({'差异' if len(results_fp16) == len(results_4bit) else 'FAIL: 结果数量不一致'})")
        diff_count = 0
        for i, (r16, r4) in enumerate(zip(results_fp16, results_4bit)):
            if set(r16["predicted"]) != set(r4["predicted"]):
                diff_count += 1
                if diff_count <= 15:
                    print(f"\n  [{i+1}] {r16['text']}")
                    print(f"    标注: {r16['expected']}")
                    print(f"    fp16: {r16['predicted']}")
                    print(f"    4bit: {r4['predicted']}")
        print(f"\n  共 {diff_count}/{len(results_fp16)} 条输出不一致")

    # ── 导出 JSON ──
    output = {
        "test_count": len(TEST_CASES),
        "quant_mode": quant_mode,
        "float16_metrics": metrics_fp16,
        "4bit_metrics": metrics_4bit,
        "diff_count": diff_count if (fp16_available and quant_available) else None,
    }
    output_path = PROJECT / "data" / "ablation_nlu_quant.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
