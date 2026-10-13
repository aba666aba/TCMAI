---
license: agpl-3.0
library_name: pytorch
pipeline_tag: text-classification
tags:
- traditional-chinese-medicine
- tcm
- recommendation
- pytorch
- checkpoint
language:
- zh
---

# TCMAI 推荐模型权重（TCMNet）

本模型库是 **TCMAI 中医药智能方剂生成系统** 的深度学习推荐组件权重。
对应 GitHub 仓库：[aba666aba/TCMAI](https://github.com/aba666aba/TCMAI)

> ⚠️ **科研原型 / 软件演示，不是医疗器械，不构成诊断、处方或用药建议。**
> 详见下方"免责声明"。

---

## 1. 这个权重是做什么的

`TCMNet` 负责把**标准化症状集合**映射为**证型、药材、方剂**三类候选，供下游与知识图谱推理结果融合：

```
症状（229 标准症状） ──► TCMNet ──► 证型 logits / 药材多标签得分 / 方剂得分
                                        │
                                        ▼
                          与 KG 推理结果融合（model×0.6 + kg×0.4）
```

NLU 症状抽取由 Qwen2.5-0.5B-Instruct 承担（公开模型，不在本库内）。

## 2. 文件说明

| 文件 | 说明 |
|------|------|
| `best_model.pth` | PyTorch 检查点（`torch.save` 字典），约 127 MB |

检查点内部结构：

| 键 | 说明 |
|----|------|
| `model_state_dict` | TCMNet 权重（115 个张量，**11.08M 参数**） |
| `optimizer_state_dict` | AdamW 优化器状态（仅供继续训练） |
| `epoch` | 最优轮次（7） |
| `metrics` | 该轮验证指标 |
| `history` | 训练/验证损失曲线 |
| `config` | 模型结构参数：`{"d_model": 384, "num_layers": 4, "version": "v2"}` |

## 3. 训练配置与指标

指标取自**检查点自带记录**（`metrics` / `history` / `config` 字段），与权重严格对应：

| 项 | 值 |
|----|-----|
| 模型结构 | `d_model=384`，`num_layers=4`，`nhead=8` |
| 训练轮次 | 7（`val_loss` 1.603 → 1.411 单调下降） |
| device | CPU |
| **证型准确率 `syn_acc`** | **0.979** |
| **药材召回 `herb_recall10`** | **0.398** |
| 验证损失 `val_loss` | 1.408 |
| 损失构成 | 证型 CE + 方剂 CE×0.5 + 药材加权 BCE×0.3 + 对比学习×0.2 |

> 训练数据由 `tcm_recommender/data_pipeline.py` 按 5 阶段流水线合成（知识库 + 关联矩阵 + 噪声注入 + 负采样），
> **并非真实临床标注数据**。上述指标是在同分布合成验证集上的结果，不代表临床有效性。
>
> ⚠️ 主仓库中的 `tcm_recommender/best_model.json` 记录的是**另一次训练**
> （`d_model=256, num_layers=2, output=best_model_v4.pth`），与本权重不匹配，请勿据此核对指标。

## 4. 使用方法

```bash
# 1. 下载（脚本会自动放到正确路径）
python scripts/download_weights.py --only ckpt

# 2. 命令行推理
python -m tcm_recommender.recommend "失眠 乏力 头晕" \
    --model tcm_recommender/best_model.pth \
    --db data/tcm_knowledge.db \
    --top-herb 10 --top-formula 5
```

模型结构参数从检查点的 `config` 字段读取（`recommend.py` 会自动适配），
因此换用不同 `d_model` / `num_layers` 的权重时无需改动代码。

**环境依赖**：`torch>=2.0`，Python 3.12。CPU 即可推理。

## 5. 已知限制

- 验证集为**合成数据**，指标偏高，只反映拟合能力，不构成有效性证据
- `herb_recall10 ≈ 0.40`，药材推荐召回仍有限；系统中由知识图谱推理与规则层兜底
- 输出**不经过**临床验证；系统中的安全筛查（十八反十九畏、毒性药材）是独立规则层，不在本权重内
- 训练脚本缺少全局随机种子，且训练集 `data/train_data*.npz` 未随仓库发布，**权重无法精确复现**
- 使用前请阅读主仓库 README 的完整医疗免责声明

## 6. 许可与引用

本权重与 TCMAI 主仓库一致，采用 **AGPL-3.0-or-later**。
AGPL 第 13 条：以网络服务方式提供本程序的修改版本，必须向所有用户提供对应源代码。

```bibtex
@misc{tcmai2026,
  title  = {TCMAI: 中医药智能方剂生成系统},
  author = {aba666aba},
  year   = {2026},
  url    = {https://github.com/aba666aba/TCMAI}
}
```

## 7. 免责声明

本模型是面向中医药知识学习与算法研究的**科研原型**，不是医疗器械，未取得任何医疗资质。
其输出**全部由算法自动生成**，不构成诊断结论、处方或用药建议，不得作为购药、煎服、
调整剂量或停药等任何医疗决定的依据。中医诊疗须由执业中医师四诊合参。
如出现急危重症表现，请立即就医。**禁止依据本模型输出自行用药。**
