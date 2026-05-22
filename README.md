# 中医药智能问答系统 (TCMAI)

> Traditional Chinese Medicine Intelligent Q&A System  
> 版本: 2.1.0 | 许可证: Apache-2.0

---

## 目录

- [项目概述](#项目概述)
- [系统架构](#系统架构)
- [功能模块](#功能模块)
- [数据资源](#数据资源)
- [安装与运行](#安装与运行)
- [AI 推荐模型](#ai-推荐模型)
- [开发指南](#开发指南)
- [项目状态](#项目状态)

---

## 项目概述

本系统是一个基于 **Python + Tkinter** 的桌面应用程序，提供中医药领域的智能问答与推荐服务。支持两种运行模式：

| 模式 | 说明 | 依赖 |
|------|------|------|
| **快速模式** | 基于规则引擎的本地问答（症状检测 → 证型推断 → 方剂推荐） | 零额外依赖 |
| **AI 模式** | 基于深度学习的智能推荐（TCMNet Transformer 模型） | PyTorch + CUDA (可选) |

系统内置 33 种常见症状、9 种中医证型、3000+ 药材和 1000+ 方剂数据，支持自然语言症状输入、证型诊断推理和药材/方剂推荐。

---

## 系统架构

```
tcmai/
│
├── launcher.py / launcher.pyw    # 桌面启动器（入口）
├── src/
│   ├── main.py                   # 命令行入口
│   ├── core/                     # 核心业务逻辑（原创，软著保护）
│   │   ├── tcm_config.py         # ★ 统一配置中心（症状/证型/药材/方剂定义）
│   │   ├── symptom_pipeline.py   # ★ 统一症状处理管线
│   │   ├── symptom_analysis.py   # 症状分析器（标准化→证型推断→禁忌检查）
│   │   ├── fallback.py           # 降级回答引擎（基于规则）
│   │   ├── knowledge_base.py     # 数据库查询接口
│   │   ├── result_verify.py      # 结果审核与过滤
│   │   └── data_desensitize.py   # 数据脱敏处理
│   │
│   ├── desktop/                  # 桌面应用层
│   │   ├── app.py                # Tkinter 桌面主界面（1280×840）
│   │   └── service.py            # 桌面业务服务
│   │
│   └── ...                       # 其他辅助模块
│
├── tcm_recommender/              # ★ 深度学习推荐子系统
│   ├── model.py                  # TCMNet Transformer 模型（11.2M 参数）
│   ├── train.py                  # 训练脚本（多任务损失 + 对比学习）
│   ├── recommend.py              # TCMRecommender 推理接口
│   ├── data_pipeline.py          # 训练数据生成管线（5阶段生成）
│   ├── data_expansion.py         # 数据库扩展脚本（基于教材）
│   ├── best_model.pth            # 当前模型权重（41.6 MB）
│   └── best_model.json           # 训练配置记录
│
├── data/                         # 数据目录
│   ├── tcm_knowledge.db          # SQLite 数据库（3497 药材, 1019 方剂）
│   ├── train_data.npz            # 训练数据（94333 样本）
│   ├── history.json              # 问答历史记录
│   └── logs/                     # 运行日志
│
├── docs/                         # 文档
│   ├── 操作手册.md               # 用户操作手册
│   ├── 设计说明.md               # 软件设计说明
│   └── 项目优化方案.md           # 优化方案文档
│
├── pyproject.toml                # 项目元数据
├── requirements.txt              # 基础依赖
├── requirements-recommender.txt  # AI 模型依赖
├── tcm_desktop.spec              # PyInstaller 打包配置
├── 打包桌面版.bat                # 桌面版打包脚本
└── 快速启动.bat                  # 快速启动脚本
```

---

## 功能模块

### 1. 统一配置中心 (`tcm_config.py`)

所有业务模块的单一数据源，包含：

- **33 种症状**：乏力、失眠、头痛、咳嗽、发热等
- **症状别名映射**：33 条口语→标准术语映射（如"肚子疼"→"腹痛"）
- **33 条症状详细信息**：类别、描述、相关证型
- **9 种证型定义**：脾胃虚弱、气血两虚、肝郁气滞、肾阳虚等
- **15 条内置药材**（人参、黄芪、当归等）+ **14 条经典方剂**
- **配伍禁忌**：十八反、十九畏、孕妇禁用
- **8 种毒性药材**清单
- 15 种 UI 快捷症状标签

### 2. 症状处理管线 (`symptom_pipeline.py`)

统一入口，四步处理流程：

```
用户输入 → 关键词检测 → 症状标准化 → 证型推断 → 禁忌检查 → 调理建议
```

### 3. 桌面应用 (`desktop/app.py`)

基于 Tkinter 的 1280×840 桌面界面，特点：

- 深绿色主题配色（#1f4f3c/#2d6a4f）
- 快捷症状标签选择
- 自由文本输入
- 历史记录管理
- 异步查询处理（线程安全队列）

### 4. 降级回答引擎 (`fallback.py`)

当 AI 模型不可用时，基于规则的中医药知识回答：症状检测 → 药材/方剂搜索 → 养生建议。

### 5. 症状分析器 (`symptom_analysis.py`)

基于中医辨证论治理论：
- 症状标准化（口语→专业术语）
- 证型匹配（规则匹配 + 置信度评分）
- 禁忌检查
- 调理建议生成

### 6. 数据库查询 (`knowledge_base.py`)

SQLite 数据库操作接口，支持 SQL 注入防护。

---

## 数据资源

| 资源 | 规模 | 来源 |
|------|------|------|
| 药材 | **3,497 种** | SQLite 数据库（《中药学》教材扩展） |
| 方剂 | **1,019 种** | SQLite 数据库（《方剂学》教材扩展） |
| 症状 | **200 种**（深度学习）/ **33 种**（快速模式） | 扩展配置 + 统一配置 |
| 证型 | **17 种**（深度学习）/ **9 种**（快速模式） | 扩展配置 + 统一配置 |
| 训练样本 | **94,333 条** | 5 阶段数据管线自动生成 |
| 配伍禁忌 | 十八反、十九畏、孕妇禁用 | 中药学标准 |

数据库可通过 `data_expansion.py` 脚本从《中药学》《方剂学》教材数据扩展生成。

---

## 安装与运行

### 环境要求

- **Python**: ≥ 3.10（推荐 3.9+，实际兼容）
- **操作系统**: Windows（Tkinter 桌面应用）

### 快速启动

```bash
# 1. 安装基础依赖
pip install -r requirements.txt

# 2. 直接运行（双击即可）
快速启动.bat

# 3. 或命令行启动
python launcher.py
```

### AI 模式（深度学习推荐）

```bash
# 安装 AI 推理依赖
pip install torch>=2.0.0 numpy>=1.24.0 tqdm>=4.65.0

# GPU 加速（可选，需 NVIDIA GPU）
# CUDA 12.4 版：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 桌面版打包

```bash
打包桌面版.bat
# 或
pyinstaller tcm_desktop.spec
```

---

## AI 推荐模型

### TCMNet 架构

当前模型为 **TCMNet**，基于 Transformer 的多任务深度学习模型。

| 参数 | 完整配置 | 当前模型 |
|------|---------|---------|
| 参数总量 | **11,192,726** | 11,192,726 |
| 模型维度 (d_model) | 384 | 256 |
| Transformer 层数 | 4 | 2 |
| 注意力头数 | 8 | 8 |
| 训练 epoch | 50 | 1 |
| 训练数据 | 94,333 样本 | `train_data_v4_small.npz` |
| 证型分类准确率 | — | **96.4%** |
| 药材召回率@10 | — | **25.7%** |

**当前模型 (`best_model.pth`)** 是用小配置（d_model=256, 2层, 1 epoch）快速验证的。完整配置（d_model=384, 4层, 50 epoch）训练后可显著提升药材召回率。

### 模型组件

- **SymptomEncoder**: 4 层 Transformer (PreLN) + AttentivePooling
- **HerbEncoder**: 双塔结构（CharCNN 文本塔 + 属性嵌入塔）→ 门控融合
- **CrossAttentionRanker**: 症状→药材/方剂交叉注意力评分
- **SyndromeHead**: MLP 分类头（17 证型）
- **ContrastiveLoss**: InfoNCE 对比学习

### 训练数据管线

5 阶段自动生成，覆盖证型驱动、药材驱动、方剂驱动、组合增强、负采样：

| 阶段 | 占比 | 说明 |
|------|------|------|
| 证型驱动 | ~50% | 为每个证型生成症状子集变体 |
| 药材驱动 | ~20% | 为每味药材匹配关联症状 |
| 方剂驱动 | ~12.5% | 为每个方剂解析药材+症状 |
| 组合增强 | ~25% | 随机症状组合 + 关联矩阵标签 |
| 负采样 | ~10% | 零关联负样本 |

### 完整训练命令

```bash
python tcm_recommender/train.py ^
  --data data/train_data.npz ^
  --db data/tcm_knowledge.db ^
  --epochs 50 ^
  --batch-size 64 ^
  --d-model 384 ^
  --num-layers 4 ^
  --lr 1e-4 ^
  --device cuda ^
  --grad-accum 2 ^
  --contrastive-weight 0.2 ^
  --output tcm_recommender/best_model.pth
```

---

## 开发指南

### 依赖管理

```bash
# 基础依赖
pip install -r requirements.txt

# AI 推荐依赖
pip install -r requirements-recommender.txt

# 全部依赖
pip install -e .[recommender,dev]

# 代码检查
ruff check .
```

### 项目约定

- **编码**: UTF-8
- **Python**: 3.9+（兼容 3.9/3.10/3.11）
- **代码风格**: 遵循 ruff 配置 (`ruff.toml`)
- **版权**: Apache-2.0，所有原创代码保留软著权益

### 数据扩展

```bash
# 从《中药学》《方剂学》教材数据扩展数据库
python tcm_recommender/data_expansion.py
```

---

## 项目状态

### 版本历史

| 提交 | 日期 | 说明 |
|------|------|------|
| `HEAD` | 2026-05-22 | V4 重构：模型重命名、清理旧文件、新增核心模块和项目配置 |
| `a1de053` | 2026-05-22 | V4 升级：100K 级数据管线 + 最新 Transformer 架构 |
| `aa5a3fe` | 2026-05-22 | V2.1→V3：数据库扩展到 1024 药材/377 方剂/200 症状/17 证型 |
| `6498f11` | 2026-04-07 | 重构：Web→Desktop 桌面模块迁移 |

### 模型训练纪要

- `best_model.pth` 当前为小配置快速验证版本（1 epoch）
- 完整 50 epoch 训练预计: GPU (RTX 3050) 约 15-30 分钟
- 训练需修复 `-1e9` 溢出问题（已在 `model.py` 中修复）

### 待办清单

- [ ] 用完整配置（d_model=384, 4层, 50 epoch）重新训练模型
- [ ] 恢复测试用例并适配 V4 架构
- [ ] 添加 `.gitignore` 排除 `__pycache__`/`dist`/`*.bak`

---

> **免责声明**: 本系统仅提供中医药知识科普和参考信息，不构成医疗诊断或治疗建议。用户如有健康问题，请咨询专业医疗机构或执业医师。
