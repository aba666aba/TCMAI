<div align="center">
  <br>
  <img src="https://img.shields.io/badge/Python-3.9+-256a4b?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.6-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Qwen2.5-1.5B-8c5a2b?style=for-the-badge&logo=data:image/svg%2bml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgdmlld0JveD0iMCAwIDMyIDMyIj48cGF0aCBkPSJNMTYgMkM4LjI2OCAyIDIgOC4yNjggMiAxNnM2LjI2OCAxNCAxNCAxNCAxNC02LjI2OCAxNC0xNFMyMy43MzIgMiAxNiAyem0wIDI2QzkuMzcyIDI4IDQgMjIuNjI4IDQgMTZTOS4zNzIgNCAxNiA0czEyIDUuMzcyIDEyIDEyLTUuMzcyIDEyLTEyIDEyeiIgZmlsbD0iI2M4OTUzZSIvPjwvc3ZnPg==&logoColor=white" alt="NLU">
  <img src="https://img.shields.io/badge/许可证-Apache%202.0-8aae96?style=for-the-badge" alt="License">

  <br><br>
  
  # 🌿 中医药智能方剂生成系统
  
  **TCMAI — Traditional Chinese Medicine Intelligent Formula Generation System**
  
  <p align="center">
    <i>让中医诊疗更智能 · 从自然语言到精准方剂 · 知识图谱驱动</i>
  </p>

  <br>
  
  <table align="center">
    <tr>
      <td align="center">🧠 <b>NLU 自然语言理解</b><br><small>Qwen2.5 大模型驱动</small></td>
      <td align="center">⚕️ <b>证型推断</b><br><small>30 种证型分类</small></td>
      <td align="center">🌾 <b>药材推荐</b><br><small>2,961 味药材</small></td>
      <td align="center">📜 <b>方剂匹配</b><br><small>717 首方剂</small></td>
      <td align="center">🔗 <b>知识图谱</b><br><small>可视化关联网络</small></td>
    </tr>
  </table>

  <br>
</div>

---

## 📋 目录

- [🌟 项目概述](#-项目概述)
- [✨ 系统能力](#-系统能力)
- [🏗️ 系统架构](#️-系统架构)
- [⚡ 核心流程](#-核心流程)
- [🧩 功能模块](#-功能模块)
- [📊 数据资源](#-数据资源)
- [🔧 安装与运行](#-安装与运行)
- [🤖 AI 模型](#-ai-模型)
- [☁️ 云端训练](#️-云端训练)

---

## 🌟 项目概述

> **从医生的一句「胃部隐痛两天，吃不下饭」到完整的方剂推荐——全程由 AI 理解、推理、输出。**

本系统是一个基于 **Python + Tkinter + PyTorch** 的桌面应用程序，采用**两阶段推理架构**：

```
🗣️ 自然语言 → 🤖 NLU 大模型 → 🎯 标准化症状 → 🧠 TCMNet → 📋 证型/药材/方剂
```

| 阶段 | 模型 | 能力 | 显存 |
|:----|:----|:----|:----:|
| **NLU** 🧠 | Qwen2.5-1.5B (4-bit) | 理解任意中文病情描述 → 提取标准症状名 | ~1.2GB |
| **推荐** 🎯 | TCMNet Transformer | 症状 → 证型分类 / 药材推荐 / 方剂匹配 | ~2GB |

> ⚠️ **设计原则**：生成的药方需由专业中医师判断后使用，系统只做基础安全筛选。

---

## ✨ 系统能力

<div align="center">

| 🏆 能力 | 🛠️ 实现方式 | 📈 指标 |
|:--------|:------------|:-------:|
| **自然语言理解** | Qwen2.5-1.5B-Instruct 4-bit 量化 | 任意中文输入 → 标准症状名 |
| **证型推断** | TCMNet Transformer + CrossAttention | 30 证型分类，Top-1 准确率 >83% |
| **药材推荐** | CrossAttentionRanker + Sigmoid 评分 | 2,961 味药材，召回率 Top-10 >70% |
| **方剂匹配** | CrossAttentionRanker + Softmax | 717 首方剂 |
| **禁忌筛查** | 6 层安全体系 | 十八反/十九畏/孕妇禁忌/毒性检测 |
| **知识图谱** | NetworkX 可视化 + 12 种关系边 | ~15,000 节点，~60,000 边 |
| **证型展开** | 自动匹配 | 输入"肾虚"→展开为肾阳虚/肾阴虚症状 |

</div>

---

## 🏗️ 系统架构

```
E:\Tom\TCMAI/
│
├── 🚀 launcher.py / .pyw           # 桌面启动器
├── 📦 pyproject.toml               # 项目配置
├── 📜 requirements.txt             # 依赖清单
│
├── 🧠 src/nlu/                     # ★ NLU 自然语言理解引擎
│   ├── engine.py                   #   Qwen2.5-1.5B 加载与推理
│   └── __init__.py
│
├── ⚙️ src/core/                    # 核心业务逻辑
│   ├── tcm_config.py               #   配置中心（症状/证型/禁忌）
│   ├── symptom_pipeline.py         #   症状处理管线
│   ├── symptom_analysis.py         #   症状分析器
│   ├── knowledge_base.py           #   SQLite 查询
│   └── intelligent_search.py       #   智能搜索引擎
│
├── 🖥️ src/desktop/                 # 桌面应用层
│   ├── app.py                      #   Tkinter 界面
│   ├── service.py                  #   业务服务层
│   ├── kg_view.py                  #   知识图谱可视化
│   ├── splash.py                   #   启动画面
│   └── runtime.py                  #   运行时工具
│
├── 🔗 src/kg/                      # 知识图谱
│   ├── builder.py                  #   图谱构建器（12 种关系）
│   ├── query.py                    #   图谱查询接口
│   └── recommender.py              #   KG+模型融合推荐
│
├── 🤖 tcm_recommender/             # ★ 深度学习推荐子系统
│   ├── model.py                    #   TCMNet Transformer
│   ├── train.py                    #   训练器
│   ├── recommend.py                #   推理接口
│   ├── data_pipeline.py            #   数据管线（7层关联+5阶段）
│   └── extended_config.py          #   229症状/30证型
│
├── 📦 models/nlu/qwen/             # Qwen2.5-1.5B 模型文件
│
└── 📁 data/                        # 数据目录
    ├── tcm_knowledge.db            #   2,961药材 / 717方剂
    └── train_data.npz              #   563K 训练样本
```

---

## ⚡ 核心流程

### 🔄 用户查询

```
┌──────────────────────────────────────────────────────────────┐
│ 🧑‍⚕️ 医生输入：「胃部隐痛两天，吃不下饭，反酸嗳气」              │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 🧠 NLU 引擎 (Qwen2.5-1.5B 4-bit)                            │
│                                                              │
│  Prompt 引导 → 理解自然语言 → 映射到 229 种标准症状名         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ ["胃痛", "食欲不振", "嗳气"]                           │  │
│  └────────────────────────────────────────────────────────┘  │
│  ⚡ 失败时自动降级到关键词匹配                               │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 🤖 TCMNet 模型推理                                           │
│                                                              │
│  症状ID序列 → Embedding → 4层Transformer → AttentivePooling │
│  → CrossAttentionRanker → 多头输出                            │
│                                                              │
│  ┌────────────────────┬──────────────────┬─────────────────┐ │
│  │ 证型: 胃热炽盛     │ 药材 Top-5       │ 方剂 Top-3      │ │
│  │ (概率 97.2%)       │ 鸡爪莲/猫毛草... │ 清带汤/茵陈...  │ │
│  └────────────────────┴──────────────────┴─────────────────┘ │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 📋 输出整合                                                 │
│                                                              │
│  ✅ 格式化回答（症状/证型/药材/方剂/安全提示）                │
│  ✅ 更新知识图谱子图（以症状为中心展示关联网络）              │
│  ✅ 保存历史记录 + 脱敏审查                                  │
└──────────────────────────────────────────────────────────────┘
```

### 🔁 训练数据管线

```
┌─────────── DB ───────────┐
│ 2,961 药材 · 717 方剂    │
│ 229 症状 · 30 证型       │
└──────┬───────────────────┘
       │
       ▼
┌── 7 层关联矩阵构建 ──────────────────────────────────────────┐
│                                                              │
│  ① 📝 文本子串匹配 (weight=1.0)     →   5,726 关联          │
│  ② 🔄 方剂反推 (weight=0.8)        →  +4,959 关联          │
│  ③ 📂 分类语义 (weight=0.6)        → +12,007 关联          │
│  ④ 🔑 功能关键词 (weight=0.7/0.5)  →  +3,667 关联          │
│  ⑤ 🔗 别名继承 (weight=0.7)        →  +2,879 关联          │
│  ⑥ 🔀 症状同义词 (×0.7)            →  +6,536 关联          │
│  ⑦ 🛟 硬编码兜底 (weight=0.5)      →  +5,895 关联          │
│                                                              │
│  密度: 0.89% ──→ 3.68%  ▲                                   │
│  总计: 24,968 关联                                           │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌── 5 阶段样本生成 ────────────────────────────────────────────┐
│                                                              │
│  ① 🏥 证型驱动 (50%, w=1.0)   ‖ ② 🌿 药材驱动 (20%, w=0.8) │
│  ③ 📜 方剂驱动 (12%, w=0.9)   ‖ ④ 🔀 组合增强 (25%, w=0.5) │
│  ⑤ ⛔ 负样本 (17%, w=0.3)                                    │
│                                                              │
│  总样本: 563,611 条                                          │
│  平均症状: 4.2 · 平均正药材: 8.3 · 有方剂标签: 75.2%        │
└──────────────────────────────────────────────────────────────┘
```

---

## 🧩 功能模块

### 🧠 NLU 自然语言理解 (`src/nlu/`)

| 组件 | 说明 |
|:-----|:-----|
| **模型** | Qwen2.5-1.5B-Instruct，4-bit 量化 |
| **显存** | ~1.2GB |
| **加载** | 本地 `models/nlu/qwen/`，即时加载 |
| **Prompt** | COT 引导 + Few-shot 示例 → 输出 JSON 症状数组 |
| **降级** | 模型不可用时自动回退关键词匹配 |

### 🩺 证型展开

直接输入证型名即可自动展开为对应症状集：

| 输入 | 自动展开为 |
|:----|:-----------|
| `肾虚` | 畏寒、手足不温、腰膝酸软、夜尿多、头晕、耳鸣、盗汗…… |
| `脾虚` | 食欲不振、腹胀、便溏、面色萎黄、乏力、水肿…… |
| `肝气郁结` | 胁痛、胁胀、善太息、胸闷、抑郁、易怒、失眠…… |

### 🔗 知识图谱 (`src/kg/`)

| 实体类型 (10 种) | 关系类型 (12 种) |
|:-----------------|:-----------------|
| Symptom, Syndrome, Herb, Formula | belongs_to, recommends_herb/formula |
| PatentMedicine, FolkRemedy | associated_with_herb/formula |
| Category, Meridian, Nature, Flavor | composes, has_nature/flavor/meridian |
| | incompatible_with, co_occurs_with |

### 🛡️ 禁忌安全体系 (6 层)

```
┌─ ① 配置定义 ───────────────────────┐
│  tcm_config.INCOMPATIBLE_HERBS      │
│  十八反 10对 · 十九畏 9对 · 孕妇 13味│
└──────────┬──────────────────────────┘
           ▼
┌─ ② KG 图注入 ──────────────────────┐
│  _add_incompatibilities()           │
│  注入 incompatible_with 边          │
└──────────┬──────────────────────────┘
           ▼
┌─ ③ KG 查询 ───────────────────────┐
│  herb_profile() → 返回相畏列表      │
└──────────┬──────────────────────────┘
           ▼
┌─ ④ 症状分析 ──────────────────────┐
│  _check_contraindications()        │
│  证型禁忌规则                      │
└──────────┬──────────────────────────┘
           ▼
┌─ ⑤ 模型安全检查 ──────────────────┐
│  check_safety()                    │
│  herbs.contraindications 交叉检测   │
└──────────┬──────────────────────────┘
           ▼
┌─ ⑥ 桌面 UI ────────────────────┐
│  ⚠️ 提示 + 图谱红色显示         │
└─────────────────────────────────┘
```

---

## 📊 数据资源

<div align="center">

| 🏛️ 资源 | 📏 规模 | 📝 说明 |
|:---------|:-------:|:--------|
| 🌿 药材 | **2,961** 种 | 含功效/主治/用法/禁忌/性味归经 |
| 📜 方剂 | **717** 首 | 含组成/功效/主治/加减 |
| 🩺 症状 | **229** 种 (AI) / **33** 种 (桌面快捷) | 涵盖脏腑、气血、经络辨证 |
| 🏥 证型 | **30** 种 (AI) / **9** 种 (桌面) | 按肺/心/肝胆/脾胃/肾/气血津液组织 |
| 💊 中成药 | **125** 种 | 常见中成药信息 |
| 📖 民间验方 | **216** 个 | 民间单方验方 |
| 🎯 训练样本 | **563,611** 条 | 7 层关联 + 5 阶段管线生成 |
| 🕸️ 知识图谱 | **~15,000 节点 / ~60,000 边** | 12 种关系类型 |

</div>

### 📁 数据库表结构

```sql
🌿 herbs             → id, name, category, nature, flavor, meridian, 
                        efficacy, indications, usage, contraindications

📜 formulas          → id, name, composition, dosage, functions, 
                        indications, usage, modification

💊 patent_medicines  → id, name, composition, functions, indications, 
                        usage_dosage, contraindications

📖 folk_remedies     → id, disease, remedy, ingredients, method
```

---

## 🔧 安装与运行

### 📋 环境要求

| 组件 | 最低 | 推荐 |
|:----|:----|:----|
| 🐍 Python | ≥ 3.9 | 3.10+ |
| 🪟 系统 | Windows 10 | Windows 11 |
| 🎮 GPU | — | NVIDIA 4GB+ VRAM |
| 💾 内存 | 8GB | 16GB |
| 💿 磁盘 | 5GB | 10GB+（含模型） |

### 🚀 快速启动

```bash
# 1. 📥 克隆
git clone https://github.com/aba666aba/SRT.git
cd TCMAI

# 2. 📦 安装依赖
pip install -r requirements.txt

# 3. 🔥 安装深度学习（AI 模式必需）
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install bitsandbytes transformers accelerate

# 4. 🧠 准备 NLU 模型 → 放入 models/nlu/qwen/
#    从魔搭 ModelScope 下载 Qwen2.5-1.5B-Instruct

# 5. 🎬 启动
python launcher.py
```

**或者直接双击** `知识图谱中药方生成系统.pyw` ✨

---

## 🤖 AI 模型

### 🧠 NLU 引擎 (Qwen2.5-1.5B)

| 配置 | 值 |
|:----|:----|
| 模型 | `Qwen/Qwen2.5-1.5B-Instruct` |
| 量化 | 4-bit (bitsandbytes) |
| 显存 | ~1.2 GB |
| 加载 | 本地路径 `models/nlu/qwen/` |
| 输出 | JSON 症状名数组 |

### 🎯 TCMNet 推荐模型

| 配置 | 值 |
|:----|:----|
| 架构 | Transformer Encoder + Dual Tower + CrossAttention |
| 参数量 | ~11M (d_model=384, 4层) / ~30M (d_model=512, 6层) |
| 显存 | ~2 GB (384) |
| 多头输出 | 证型 Softmax + 药材 BCE + 方剂 Softmax |
| 对比学习 | InfoNCE，拉近症状-药材匹配对 |

### 🔗 KG 融合推荐

```
fused_score = model_score × 0.6 + kg_score × 0.4
if 双方一致: fused_score += 0.15 （额外加分）
```

---

## ☁️ 云端训练

### 📤 上传

```
tcm_recommender/       # 全部代码
data/tcm_knowledge.db  # 数据库
```

### 🏋️ 训练命令

```bash
# 生成数据
python tcm_recommender/data_pipeline.py \
  --db data/tcm_knowledge.db \
  --output data/train_data.npz \
  --count 500000

# 训练模型
python tcm_recommender/train.py \
  --data data/train_data.npz \
  --db data/tcm_knowledge.db \
  --epochs 30 --batch-size 256 \
  --d-model 384 --num-layers 4 \
  --lr 2e-4 --device cuda \
  --output tcm_recommender/best_model.pth
```

### 📥 下载

| 文件 | 大小 | 说明 |
|:----|:----:|:-----|
| `best_model.pth` | ~127 MB | 模型权重，含优化器状态 |
| `best_model.json` | ~2 KB | 训练指标和配置 |

---

<div align="center">
  <br>
  <p>
    <sub>🌿 基于中医药理论与深度学习 · 辅助诊疗 · 仅供参考</sub>
  </p>
  <p>
    <sub>Generated formulas require professional TCM practitioner verification.</sub>
  </p>
  <br>
  <p>
    <a href="#-目录">⬆ 回到顶部</a>
  </p>
  <br>
</div>
