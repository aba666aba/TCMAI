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
├── launcher.pyw                 # 桌面启动器（唯一入口）
├── launcher.py                  # 启动模块（被入口导入）
├── pyproject.toml               # 项目元数据（v2.1.0）
├── requirements.txt             # 全部依赖（含可选标记）
├── ruff.toml                    # 代码风格配置
├── tcmai.spec                   # PyInstaller 打包配置
│
├── scripts/                     # 工具脚本
│   ├── 快速启动.bat             # 快速启动脚本
│   ├── 打包exe.bat              # PyInstaller 打包脚本
│   ├── import_knowledge.py/v2.py  # 知识库导入脚本
│   └── ...                      # 其他辅助脚本
│
├── src/                         # 源码目录
│   ├── core/                    # 核心业务逻辑（原创，软著保护）
│   │   ├── tcm_config.py         # ★ 统一配置中心（33症状/9证型/15药材/14方剂）
│   │   ├── symptom_pipeline.py   # 统一症状处理管线
│   │   ├── symptom_analysis.py   # 症状分析器（标准化→证型推断→禁忌检查）
│   │   ├── fallback.py           # 降级回答引擎（基于规则）
│   │   ├── knowledge_base.py     # SQLite 数据库查询
│   │   ├── result_verify.py      # 结果审核与过滤
│   │   └── data_desensitize.py   # 数据脱敏（中医药白名单保护）
│   │
│   ├── desktop/                  # 桌面应用层
│   │   ├── app.py                # Tkinter 桌面界面（1280×840）
│   │   ├── service.py            # 桌面业务服务层
│   │   └── runtime.py            # 运行时工具
│   │
│   └── ...                       # 其他辅助模块
│
├── tcm_recommender/              # ★ 深度学习推荐子系统
│   ├── model.py                  # TCMNet Transformer（~11M 参数，4GB GPU）
│   ├── train.py                  # 训练器（AMP+多任务+对比学习）
│   ├── recommend.py              # TCMRecommender 推理接口
│   ├── data_pipeline.py          # 训练数据管线（5阶段 150K 样本）
│   ├── data_expansion.py         # 数据库扩展（3000+药材）
│   └── extended_config.py        # 200症状/17证型定义（AI端）
│
├── data/                         # 数据目录
│   ├── tcm_knowledge.db          # 3010药材 / 1095方剂 / 125中成药 / 222验方
│   └── train_data.npz            # 训练数据（~150K 样本）
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

- **Python**: ≥ 3.10
- **操作系统**: Windows（Tkinter 桌面应用）

### 快速启动

```bash
# 1. 安装基础依赖
pip install -r requirements.txt

# 2. 直接运行（双击即可）
scripts\快速启动.bat

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
scripts\打包exe.bat
# 或
pyinstaller tcmai.spec
```

---

## AI 推荐模型

### TCMNet 架构

当前模型为 **TCMNet**，基于 Transformer 的多任务深度学习模型（11,192,726 参数）。

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

### 训练命令

```bash
python tcm_recommender/train.py --data data/train_data.npz --db data/tcm_knowledge.db --epochs 50 --batch-size 64 --d-model 384 --num-layers 4 --lr 1e-4 --device cuda --grad-accum 2 --contrastive-weight 0.2 --output tcm_recommender/best_model.pth
```

---

## 开发指南

### 依赖管理

```bash
# 基础依赖（含可选标记）
pip install -r requirements.txt

# 全部依赖（含开发工具）
pip install -e .[recommender,dev]

# 代码检查
ruff check .
```

### 项目约定

- **编码**: UTF-8
- **Python**: ≥ 3.10
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
| `HEAD` | 2026-05-22 | V4 三轮优化：DRY养生建议/正则改进/特征缓存/npz校验/死代码清理 |
| `7952494` | 2026-05-22 | V4 重构：负样本修复/跨包依赖清理/UI同步/死代码清理 |
| `a1de053` | 2026-05-22 | V4 升级：100K 级数据管线 + Transformer 架构 |
| `aa5a3fe` | 2026-05-22 | V2.1→V3：数据库扩展到 1024 药材/377 方剂/200 症状/17 证型 |
| `6498f11` | 2026-04-07 | 重构：Web→Desktop 桌面模块迁移 |

---

> **免责声明**: 本系统仅提供中医药知识科普和参考信息，不构成医疗诊断或治疗建议。用户如有健康问题，请咨询专业医疗机构或执业医师。
