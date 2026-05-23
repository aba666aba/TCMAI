"""找出训练中漏掉的方剂"""
import numpy as np
from collections import Counter
from tcm_recommender.data_pipeline import TCMDataPipeline

db_path = 'data/tcm_knowledge.db'
pipeline = TCMDataPipeline(db_path)
pipeline.load_all()

data = np.load('data/train_data.npz', allow_pickle=True)
fl = data['formula_labels']
fc = Counter(fl.tolist())
in_train = {l for l in fc if l > 0}

# 检查每个方剂有没有进入训练集
missing = []
for fid, f in pipeline.formulas.items():
    fi = pipeline.formula_db_id_to_model_idx.get(fid)
    if fi and fi not in in_train:
        # 看它的增强关联矩阵
        missing.append((fid, f['name'], f.get('category','')))

print(f'漏掉的方剂 ({len(missing)}):')
for fid, name, cat in missing:
    print(f'  {fid:>4}. {name:<30} {cat or ""}')
pipeline.close()
