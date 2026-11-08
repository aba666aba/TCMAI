# -*- coding: utf-8 -*-
"""打包 TCMNet 服务器训练包（最小依赖，上传租用 GPU 服务器直接训练）。

产物：dist/tcmnet_training_<日期>.zip
内容：tcm_recommender 源码（5 个 .py）+ data/tcm_knowledge.db + SERVER_README.md
依赖（服务器）：torch(CUDA) / numpy / tqdm —— 其余全是标准库。
"""
import os
import shutil
import zipfile
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "dist", "tcmnet_training")
STAMP = datetime.now().strftime("%Y%m%d")

FILES = [
    "tcm_recommender/__init__.py",
    "tcm_recommender/model.py",
    "tcm_recommender/train.py",
    "tcm_recommender/data_pipeline.py",
    "tcm_recommender/recommend.py",
    "tcm_recommender/extended_config.py",
    "data/tcm_knowledge.db",
]

README = """# TCMNet 服务器训练包（3090 / 24GB）

## 1. 环境（一次性）

```bash
# Python 3.10~3.12
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy tqdm
# 验证 GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. 训练（3090 推荐参数）

```bash
cd tcmnet_training
nohup python -m tcm_recommender.train \\
  --db data/tcm_knowledge.db \\
  --epochs 50 --device cuda --batch-size 256 \\
  > train.log 2>&1 &
tail -f train.log        # 看进度
```

说明：
- 首次运行自动按当前数据库生成训练集（约 56 万样本，几分钟）；
- batch 256 在 24GB 上余量充足（峰值约 4-6GB），可试 512 进一步提速；
- 50 epochs 全量预计 **1.5~3 小时**（3090 vs 笔记本 3050 快约 20-50 倍）；
- 数据/库一致性有守卫：npz 与 db 实体数不匹配会拒绝训练（防错位模型）。

## 3. 训练完成后回传

产物：`tcm_recommender/best_model.pth`（新架构 checkpoint，组合式方剂表征）

```bash
# 服务器 -> 本机（在本机执行）
scp user@<服务器IP>:<路径>/tcmnet_training/tcm_recommender/best_model.pth \\
    E:/AAATOM/NewLai/TCMAI/tcm_recommender/best_model.pth
```

本地桌面版会**自动识别新架构**（checkpoint 含 comp_encoder 键），
方剂表征 = 组成药材药性聚合：新增方剂免重训、方剂-药材共享表征空间。

## 4. 健康指标

- syn_loss（证型）应从 ~4.5 降至 <1.5；
- total_loss 从 ~10 持续下降；验证 loss 回升即为过拟合（可减 epochs 重训）；
- 结束时有最终测试（Recall@K 等）输出在 train.log 末尾。
"""


def main():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    for rel in FILES:
        src = os.path.join(ROOT, rel.replace("/", os.sep))
        dst = os.path.join(OUT_DIR, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  + {rel}")

    readme_path = os.path.join(OUT_DIR, "SERVER_README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(README)
    print("  + SERVER_README.md")

    zip_path = os.path.join(ROOT, "dist", f"tcmnet_training_{STAMP}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for base, _, files in os.walk(OUT_DIR):
            for fn in files:
                full = os.path.join(base, fn)
                rel = os.path.relpath(full, OUT_DIR)
                z.write(full, f"tcmnet_training/{rel.replace(os.sep, '/')}")

    size_mb = os.path.getsize(zip_path) / 1024**2
    print(f"\n打包完成: {zip_path}  ({size_mb:.2f} MB)")
    print(f"目录版（可直接 scp -r）: {OUT_DIR}")


if __name__ == "__main__":
    main()
