# 中医药智能方剂生成系统 (TCMAI)

> Traditional Chinese Medicine Intelligent Formula Generation System
> 许可证: Apache-2.0

---

## 目录

- [项目概述](#项目概述)
- [系统能力](#系统能力)
- [系统架构](#系统架构)
- [核心流程](#核心流程)
- [功能模块](#功能模块)
- [数据资源](#数据资源)
- [安装与运行](#安装与运行)
- [AI 模型](#ai-模型)
- [云端训练](#云端训练)

---

## 项目概述

本系统是一个基于 **Python + Tkinter + PyTorch** 的桌面应用程序，核心目标是构建**智能方剂生成系统**——医生输入自然语言病情描述，系统自动理解并推荐药材（含用量）、匹配传统方剂，结合知识图谱可视化展示，并通过十八反十九畏等禁忌规则筛查，辅助医生提高诊断效率。

> **设计原则**：生成的药方不需要确保真实可用，医生做最终判断。系统做好基础安全筛选即可。

### 两阶段推理架构

```
医生自然语言 → [NLU 大模型] → 标准化症状 → [TCMNet] → 证型/药材/方剂 → 输出
                                  ↓ (失败)
                            关键词降级匹配
```

---

## 系统能力

| 能力 | 实现方式 |
|------|----------|
| **自然语言理解** | Qwen2.5-1.5B-Instruct 4-bit 量化（~1.2GB 显存），理解任意中文病情描述 |
| **证型推断** | TCMNet Transformer 模型（30 证型分类），准确率 >83% |
| **药材推荐** | 从 2961 种药材中推荐最匹配的 10 味（含置信度） |
| **方剂匹配** | 从 717 首方剂中推荐最匹配的 5 首 |
| **禁忌筛查** | 6 层安全体系：十八反/十九畏/孕妇禁忌/毒性药材 |
| **知识图谱** | NetworkX 图谱可视化，展示证型/药材/方剂的关联网络 |
| **证型展开** | 直接输入"肾虚""脾虚"等证型名，自动展开为典型症状集 |

---

## 系统架构

```
E:\Tom\TCMAI/
├── launcher.py/pyw               # 桌面启动器（Splash + 主窗口）
├── pyproject.toml                 # 项目元数据
├── requirements.txt               # 全部依赖
├── ruff.toml                      # 代码风格配置
├── tcmai.spec                     # PyInstaller 打包配置
│
├── src/                           # 核心源码
│   ├── nlu/                       # ★ NLU 自然语言理解引擎
│   │   ├── engine.py              #   Qwen2.5-1.5B 4-bit 加载与推理
│   │   └── __init__.py
│   │
│   ├── core/                      # 核心业务逻辑
│   │   ├── tcm_config.py           # 配置中心（症状/证型/禁忌/快捷标签）
│   │   ├── symptom_pipeline.py     # 症状处理管线
│   │   ├── symptom_analysis.py     # 症状分析器（标准化→证型推断→禁忌）
│   │   ├── fallback.py             # 降级回答引擎（关键词规则）
│   │   ├── knowledge_base.py       # SQLite 数据库查询
│   │   └── intelligent_search.py   # 智能搜索引擎
│   │
│   ├── desktop/                    # 桌面应用层
│   │   ├── app.py                  # Tkinter 桌面界面
│   │   ├── service.py              # 业务服务层（NLU + TCMNet 协调）
│   │   ├── kg_view.py              # 知识图谱可视化（matplotlib+networkx）
│   │   ├── splash.py               # 全屏启动画面
│   │   └── runtime.py              # 运行时工具
│   │
│   └── kg/                         # 知识图谱构建与查询
│       ├── builder.py              # TCMKnowledgeGraph 构建器（12 种关系边）
│       ├── query.py                # KnowledgeGraphQuery 查询接口
│       └── recommender.py          # KG+模型融合推荐器
│
├── tcm_recommender/                # ★ 深度学习推荐子系统
│   ├── model.py                    # TCMNet: Transformer+双塔+交叉注意力
│   ├── train.py                    # 训练器（AMP+多任务+对比学习）
│   ├── recommend.py                # TCMRecommender 推理接口
│   ├── data_pipeline.py            # 训练数据管线（7 层关联+5 阶段生成）
│   ├── extended_config.py          # 229 症状 / 30 证型定义（AI 端）
│   └── best_model.pth              # 最佳模型权重
│
├── models/                         # 大模型文件
│   └── nlu/qwen/                   # Qwen2.5-1.5B-Instruct 4-bit
│       ├── model.safetensors       #   ~2.88GB 权重文件
│       ├── config.json
│       └── tokenizer.json
│
└── data/                           # 数据目录
    ├── tcm_knowledge.db            # 2961 药材 / 717 方剂 / 30 证型
    └── train_data.npz              # 563K 训练样本（按需生成）
```

---

## 核心流程

### 用户查询流程

```
用户输入: "胃部隐痛两天，吃不下饭，反酸嗳气"
                    │
                    ▼
┌── 阶段1: NLU (Qwen2.5-1.5B, GPU) ─────────────┐
│  Prompt: 自然语言 → 229 标准症状名               │
│  输出: ["胃痛", "食欲不振", "嗳气"]             │
│  (失败时自动降级到关键词匹配)                    │
└────────────────────────────────────────────────┘
                    │
                    ▼
┌── 阶段2: TCMNet (GPU) ─────────────────────────┐
│  症状 ID 序列 → Transformer → AttentivePooling  │
│  → CrossAttentionRanker → 多头输出               │
│  输出: {"syndrome":"胃热炽盛","herbs":[...],"formulas":[...]}│
└────────────────────────────────────────────────┘
                    │
                    ▼
┌── 阶段3: 输出整合 ─────────────────────────────┐
│  · 格式化回答（症状/证型/药材/方剂/安全提示）    │
│  · 更新知识图谱子图（以症状为中心展示）          │
│  · 保存历史记录                                 │
└────────────────────────────────────────────────┘
```

### 训练数据生成流程

```
DB (2961 药材 / 717 方剂)
  │
  ├── 7 层关联矩阵构建 ───────────┐
  │   1. 文本子串匹配 (1.0)        │ 密度: 0.89% → 3.68%
  │   2. 方剂组成反推 (0.8)        │ 关联数: 5,726 → 24,968
  │   3. 分类语义匹配 (0.6)        │
  │   4. 功能关键词 (0.7/0.5)      │
  │   5. 药材别名继承 (0.7)        │
  │   6. 症状同义词映射 (×0.7)     │
  │   7. 硬编码兜底 (0.5)          │
  └────────────────────────────────┘
        │
        ▼
  ┌── 5 阶段样本生成 ──────────────┐
  │ 1. 证型驱动 (50%)    weight=1.0│
  │ 2. 药材驱动 (20%)    weight=0.8│
  │ 3. 方剂驱动 (12%)    weight=0.9│
  │ 4. 组合增强 (25%)    weight=0.5│
  │ 5. 负样本 (17%)      weight=0.3│
  └────────────────────────────────┘
        │
        ▼
  563,611 样本, 229 症状, 2961 药材, 30 证型
```

---

## 数据资源

| 资源 | 规模 | 说明 |
|------|------|------|
| 药材 | **2,961 种** | SQLite 数据库，含功效/主治/用法/禁忌 |
| 方剂 | **717 首** | 含组成/功效/主治/加减 |
| 症状 | **229 种**（AI 端）/ **33 种**（桌面端快捷） | 涵盖脏腑/气血/经络辨证 |
| 证型 | **30 种**（AI 端）/ **9 种**（桌面端） | 按肺/心/肝胆/脾胃/肾/气血津液/复合组织 |
| 中成药 | **125 种** | 常见中成药信息 |
| 民间验方 | **216 个** | 民间单方验方 |
| 训练样本 | **563,611 条** | 7 层关联矩阵 + 5 阶段管线生成 |
| 知识图谱 | **~15,000 节点 / ~60,000 边** | 12 种关系类型 |

### 数据库表结构

```sql
herbs:            id, name, category, nature, flavor, meridian, efficacy, indications, usage, contraindications
formulas:         id, name, category, composition, dosage, functions, indications, usage, modification
patent_medicines: id, name, composition, functions, indications, usage_dosage, contraindications
folk_remedies:    id, disease, remedy, ingredients, method
```

---

## 安装与运行

### 环境要求

| 组件 | 最低要求 | 推荐 |
|------|----------|------|
| Python | ≥ 3.9 | ≥ 3.10 |
| 系统 | Windows 10+ | Windows 11 |
| GPU | — | NVIDIA 4GB+ VRAM (RTX 3050) |
| 内存 | 8GB | 16GB |
| 磁盘 | 5GB 可用 | 10GB + (含模型文件) |

### 快速启动

```bash
# 1. 克隆仓库
git clone https://github.com/aba666aba/SRT.git
cd TCMAI

# 2. 安装基础依赖
pip install -r requirements.txt

# 3. 安装深度学习依赖（AI 模式必需）
pip install torch==2.6.0+cu124 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install bitsandbytes transformers accelerate

# 4. 下载/准备 NLU 模型到 models/nlu/qwen/
# 从魔搭(ModelScope)下载 Qwen2.5-1.5B-Instruct
# 或直接从 data_expansion.py 生成训练数据

# 5. 启动
python launcher.py
```

### 一键启动

双击 `知识图谱中药方生成系统.pyw` 或 `scripts\快速启动.bat`。

### 打包 exe

```bash
scripts\打包exe.bat
```

---

## AI 模型

### NLU 引擎 (Qwen2.5-1.5B)

| 参数 | 值 |
|------|-----|
| 模型 | Qwen/Qwen2.5-1.5B-Instruct |
| 量化 | 4-bit (bitsandbytes) |
| 显存 | ~1.2GB |
| 加载方式 | 本地 `models/nlu/qwen/` |
| 功能 | 自然语言描述 → 标准症状名 JSON 数组 |

### TCMNet 推荐模型

| 参数 | 值 |
|------|-----|
| 架构 | Transformer Encoder + Dual Tower + CrossAttention |
| 参数量 | ~11M (d_model=384, 4层) / ~30M (d_model=512, 6层) |
| 显存 | ~2GB (384) |
| 输入 | 症状 ID 序列 (1-based, padding=0) |
| 输出 | 证型(30类) + 药材(2961种) + 方剂(717首) |
| 损失 | CE(label_smoothing) + 加权BCE + InfoNCE |

### 知识图谱推理 (KG Enhanced)

KG 增强推荐器将 TCMNet 模型预测与知识图谱推理融合：

```
fused_score = model_score × 0.6 + kg_score × 0.4
if 双方一致推荐: fused_score += 0.15
```

三条 KG 推理路径：
- **路径 A**：症状→证型→推荐药材/方剂 (辨证论治，权重最高)
- **路径 B**：症状→直接关联药材/方剂 (权重 0.2)
- **路径 C**：多症状综合征型分析

### 禁忌安全体系 (6层)

| 层级 | 模块 | 检测内容 |
|------|------|----------|
| 1-配置定义 | `tcm_config.INCOMPATIBLE_HERBS` | 十八反10对、十九畏9对、孕妇禁用13味 |
| 2-KG图注入 | `kg/builder._add_incompatibilities()` | 注入图结构，添加 `incompatible_with` 边 |
| 3-KG查询 | `kg/query.herb_profile()` | 返回药材的相反相畏列表 |
| 4-症状分析 | `symptom_analysis._check_contraindications()` | 证型禁忌规则 |
| 5-模型安全 | `recommend.check_safety()` | herbs.contraindications 字段交叉检测 |
| 6-桌面UI | `service.py` 输出 | ⚠️ 提示 + 图谱红色显示 |

---

## 云端训练

### 上传到云端的文件

```
tcm_recommender/          # 全部代码
data/tcm_knowledge.db     # 数据库
```

### 训练命令

```bash
# 生成训练数据
python tcm_recommender/data_pipeline.py --db data/tcm_knowledge.db \
  --output data/train_data.npz --count 500000

# 训练模型
python tcm_recommender/train.py --data data/train_data.npz \
  --db data/tcm_knowledge.db --epochs 30 --batch-size 256 \
  --d-model 384 --num-layers 4 --lr 2e-4 --grad-accum 1 \
  --contrastive-weight 0.3 --device cuda \
  --output tcm_recommender/best_model.pth
```

### 下载回本地的文件

| 文件 | 说明 |
|------|------|
| `best_model.pth` | 模型权重 (~127MB) |
| `best_model.json` | 训练指标 |

---

> **免责声明**: 本系统仅提供中医药知识科普和参考信息，不构成医疗诊断或治疗建议。生成的药方需由专业中医师判断后使用。
