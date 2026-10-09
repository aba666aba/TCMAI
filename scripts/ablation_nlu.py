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
"""NLU 精度测试 — CPU float32
D:\Python\python.exe E:\TOM\TCMAI\scripts\ablation_nlu.py
"""
import sys, json, time
sys.path.insert(0, r'E:\TOM\TCMAI')
from src.nlu.engine import NLUEngine

tests = [
    ('睡不着觉', ['失眠']),
    ('晚上翻来覆去睡不着', ['失眠']),
    ('头疼得厉害', ['头痛']),
    ('吃不下饭', ['食欲不振']),
    ('没胃口什么都不想吃', ['食欲不振']),
    ('肚子咕噜咕噜响拉稀', ['腹泻']),
    ('浑身没劲', ['乏力']),
    ('走几步路就喘不上气', ['气短']),
    ('咳嗽有黄痰', ['咳嗽','咳痰']),
    ('胸口闷得慌', ['胸闷']),
    ('手脚冰凉', ['手足不温']),
    ('腰酸得直不起来', ['腰膝酸软']),
    ('晚上老起来上厕所', ['夜尿多']),
    ('心烦意乱坐立不安', ['心烦']),
    ('眼睛发红', ['目赤']),
    ('耳鸣', ['耳鸣']),
    ('鼻塞不通气', ['鼻塞']),
    ('口苦', ['口苦']),
    ('胃胀不想吃东西还便秘', ['腹胀','食欲不振','便秘']),
    ('腰酸腿软怕冷', ['腰膝酸软','畏寒']),
    ('口干手心发热晚上出汗', ['口干','五心烦热','盗汗']),
    ('心慌气短走几步就喘', ['心悸','气短','乏力']),
    ('头晕耳鸣腰酸', ['头晕','耳鸣','腰膝酸软']),
    ('失眠多梦白天头晕没精神', ['失眠','多梦','头晕','乏力']),
    ('最近压力大', []),
    ('整个人都不好了', []),
    ('上火了', ['发热','咽喉肿痛','口干']),
    ('湿气重', ['便溏','水肿','乏力']),
    ('拉肚子', ['腹泻']),
    ('吃一点就饱了', ['食欲不振']),
]

print('加载 NLU 引擎...')
t0 = time.time()
engine = NLUEngine(device='cpu', use_4bit=False)
engine.load()
print(f'加载完成 ({(time.time()-t0):.0f}s)')

results = []
for text, expected in tests:
    t1 = time.time()
    try:
        output = engine.extract_symptoms(text)
    except Exception as e:
        output = [f"ERROR: {e}"]
    elapsed = time.time() - t1
    predicted = output if isinstance(output, list) else []
    results.append({
        'text': text, 'expected': expected, 'predicted': predicted,
        'time': round(elapsed, 1),
    })
    status = 'OK' if set(predicted) == set(expected) else '!!'
    print(f'  [{status}] {text[:30]:<30} → {predicted}')

# 指标
exact = sum(1 for r in results if set(r['predicted']) == set(r['expected']))
total_tp = sum(len(set(r['predicted']) & set(r['expected'])) for r in results)
total_fp = sum(len(set(r['predicted']) - set(r['expected'])) for r in results)
total_fn = sum(len(set(r['expected']) - set(r['predicted'])) for r in results)
p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
n_exp = sum(1 for r in results if r['expected'])
at_least_one = sum(1 for r in results if r['expected'] and (set(r['predicted']) & set(r['expected'])))

print(f'\n===== NLU 精度 (CPU float32, {len(tests)}条) =====')
print(f'  精确匹配率:     {exact}/{len(tests)} = {exact/len(tests):.1%}')
print(f'  至少命中一个率: {at_least_one}/{n_exp} = {at_least_one/n_exp:.1%}')
print(f'  Precision:      {p:.1%}')
print(f'  Recall:         {r:.1%}')
print(f'  F1:             {f1:.1%}')
print(f'  平均推理时间:   {sum(r["time"] for r in results)/len(results):.1f}s')

# 保存
out = {
    'test_count': len(results),
    'precision_type': 'cpu_float32',
    'exact_match_rate': exact / len(results),
    'at_least_one_hit_rate': at_least_one / n_exp if n_exp else 0,
    'precision_micro': p, 'recall_micro': r, 'f1_micro': f1,
    'avg_inference_time_s': sum(r["time"] for r in results) / len(results),
    'details': results,
}
with open(r'E:\TOM\TCMAI\data\ablation_nlu.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f'\n结果: E:/TOM/TCMAI/data/ablation_nlu.json')
