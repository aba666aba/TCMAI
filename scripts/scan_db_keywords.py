"""扫描知识库提取所有症状/功效/主治关键词"""
import sqlite3, re
from collections import Counter

db_path = 'E:/Tom/TCMAI/data/tcm_knowledge.db'
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

def extract_terms(texts, min_len=2):
    all_words = []
    for t in texts:
        if not t:
            continue
        parts = re.split(r'[；;，,、。.\\s（）()]', t)
        for p in parts:
            p = p.strip()
            if len(p) >= min_len and not any(c.isascii() and not c.isalpha() for c in p):
                all_words.append(p)
    return Counter(all_words)

# 1. 方剂
print('='*60)
print('方剂主治高频词(前80)')
print('='*60)
cursor = conn.execute('SELECT name, indications, functions FROM formulas')
all_ind = []
all_func = []
for row in cursor:
    if row['indications']: all_ind.append(row['indications'])
    if row['functions']: all_func.append(row['functions'])

ind_cnt = extract_terms(all_ind)
func_cnt = extract_terms(all_func)
for w, c in ind_cnt.most_common(80):
    print(f'  {w}: {c}')

print()
print('='*60)
print('方剂功效高频词(前80)')
print('='*60)
for w, c in func_cnt.most_common(80):
    print(f'  {w}: {c}')

# 2. 药材
print()
print('='*60)
print('药材主治高频词(前80)')
print('='*60)
cursor = conn.execute('SELECT indications, efficacy FROM herbs')
h_ind = []
h_eff = []
for row in cursor:
    if row['indications']: h_ind.append(row['indications'])
    if row['efficacy']: h_eff.append(row['efficacy'])

h_ind_cnt = extract_terms(h_ind)
h_eff_cnt = extract_terms(h_eff)
for w, c in h_ind_cnt.most_common(80):
    print(f'  {w}: {c}')

print()
print('='*60)
print('药材功效高频词(前80)')
print('='*60)
for w, c in h_eff_cnt.most_common(80):
    print(f'  {w}: {c}')

# 3. 民间验方疾病
print()
print('='*60)
print('民间验方疾病(前40)')
print('='*60)
cursor = conn.execute('SELECT disease FROM folk_remedies')
diseases = [r['disease'] for r in cursor if r['disease']]
d_cnt = Counter(diseases)
for d, c in d_cnt.most_common(40):
    print(f'  {d}: {c}')

# 4. 中成药
print()
print('='*60)
print('中成药主治高频词(前40)')
print('='*60)
cursor = conn.execute('SELECT name, indications FROM patent_medicines')
p_ind = [r['indications'] for r in cursor if r['indications']]
p_cnt = extract_terms(p_ind)
for w, c in p_cnt.most_common(40):
    print(f'  {w}: {c}')

# 5. 把所有的症状名输出成一列表
print()
print('='*60)
print('所有方剂中提到的症状/证型汇总')
print('='*60)
# 提取可能是症状的短词(2-6字)
all_terms = set()
for counter in [ind_cnt, func_cnt, h_ind_cnt, h_eff_cnt, p_cnt]:
    for word in counter:
        word = word.strip()
        if 2 <= len(word) <= 6 and counter[word] >= 2:
            all_terms.add(word)

# 过滤掉非症状词
symptom_like = sorted([t for t in all_terms if any(kw in t for kw in 
    ['痛', '热', '寒', '虚', '湿', '肿', '咳', '痰', '泻', '秘', '喘', 
     '胀', '闷', '酸', '软', '麻', '痒', '晕', '悸', '烦', '渴',
     '汗', '带', '经', '食', '眠', '乏', '弱', '亏', '郁', '火',
     '风', '燥', '积', '滞', '瘀', '毒', '淋', '痹'])])
for t in symptom_like:
    print(f'  {t}')

conn.close()
