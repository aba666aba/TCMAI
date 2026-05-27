<div align="center">
  <br>
  <img src="https://img.shields.io/badge/Python-3.12-256a4b?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.12-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CustomTkinter-5.2-2d7a4f?style=for-the-badge&logo=python&logoColor=white" alt="UI">
  <img src="https://img.shields.io/badge/Qwen2.5-0.5B-8c5a2b?style=for-the-badge&logo=data:image/svg%2bml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgdmlld0JveD0iMCAwIDMyIDMyIj48cGF0aCBkPSJNMTYgMkM4LjI2OCAyIDIgOC4yNjggMiAxNnM2LjI2OCAxNCAxNCAxNCAxNC02LjI2OCAxNC0xNFMyMy43MzIgMiAxNiAyem0wIDI2QzkuMzcyIDI4IDQgMjIuNjI4IDQgMTZTOS4zNzIgNCAxNiA0czEyIDUuMzczIDEyIDEyLTUuMzcyIDEyLTEyIDEyeiIgZmlsbD0iI2M4OTUzZSIvPjwvc3ZnPg==&logoColor=white" alt="NLU">
  <img src="https://img.shields.io/badge/许可证-Apache%202.0-8aae96?style=for-the-badge" alt="License">

  <br><br>

  # 🌿 中医药智能问答系统

  **TCMAI — Traditional Chinese Medicine Intelligent Q&A System**

  <p align="center">
    <i>基于深度学习的智能辨证与方剂推荐 · 自然语言理解 · 知识图谱可视化</i>
  </p>

  <br>

  <table align="center">
    <tr>
      <td align="center">🧠 <b>NLU 自然语言理解</b><br><small>Qwen2.5 大模型驱动</small></td>
      <td align="center">⚕️ <b>证型推断</b><br><small>30 种证型分类体系</small></td>
      <td align="center">🌾 <b>药材推荐</b><br><small>2,961 味药材</small></td>
      <td align="center">📜 <b>方剂匹配</b><br><small>717 首经典方剂</small></td>
      <td align="center">🔗 <b>知识图谱</b><br><small>交互式可视化网络</small></td>
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
- [🖥️ UI 特性](#️-ui-特性)
- [🔧 安装与运行](#-安装与运行)
- [🧠 AI 模型](#-ai-模型)
- [📦 打包部署](#-打包部署)
- [📜 更新日志](#-更新日志)

---

## 🌟 项目概述

> **从「胃部隐痛两天，吃不下饭」到完整的方剂推荐——全程由 AI 理解、推理、输出。**

本系统是一个基于 **Python + CustomTkinter + PyTorch** 的桌面应用程序，采用**两阶段推理架构**：

```
🗣️ 自然语言 → 🤖 NLU 大模型 → 🎯 标准化症状 → 🧠 TCMNet → 📋 证型/药材/方剂
```

| 阶段 | 模型 | 能力 | 显存 |
|:----|:----|:----|:----:|
| **NLU** 🧠 | Qwen2.5-0.5B (4-bit) | 理解任意中文病情描述 → 提取标准症状名 | ~1.2GB |
| **推荐** 🎯 | TCMNet Transformer | 症状 → 证型分类 / 药材推荐 / 方剂匹配 | ~2GB |

> ⚠️ **设计原则**：生成的药方需由专业中医师判断后使用，系统只做基础安全筛选（十八反十九畏等）。

---

## ✨ 系统能力

<div align="center">

| 🏆 能力 | 🛠️ 实现方式 | 📈 指标 |
|:--------|:------------|:-------:|
| **自然语言理解** | Qwen2.5-0.5B-Instruct 4-bit 量化 | 任意中文输入 → 标准症状名 |
| **证型推断** | TCMNet Transformer + CrossAttention | 30 证型分类体系 |
| **药材推荐** | CrossAttentionRanker + Sigmoid 评分 | 2,961 味药材 |
| **方剂匹配** | CrossAttentionRanker + Softmax | 717 首经典方剂 |
| **禁忌筛查** | 6 层安全体系 | 十八反/十九畏/孕妇禁忌/毒性检测 |
| **知识图谱** | NetworkX 交互式 Canvas | ~4,300 节点，~63,000 边 |
| **智能搜索** | 多策略融合 + 同义词扩展 | 272 条口语→标准映射 |
| **历史管理** | 搜索/导出/删除 | 保存最近 500 条记录 |

</div>

---

## 🏗️ 系统架构

```
E:\Tom\TCMAI/
│
├── 🚀 launcher.py                  # 开发模式启动器
├── 🚀 启动中医药智能问答系统.pyw     # 双击启动（自动使用 venv）
├── 📦 pyproject.toml               # 项目元数据
├── 📜 requirements.txt             # 全量依赖清单
├── 🔧 tcmai.spec                   # PyInstaller 打包配置
│
├── 🧠 src/nlu/                     # ★ NLU 自然语言理解引擎
│   └── engine.py                   #   Qwen2.5 加载与推理
│
├── ⚙️ src/core/                    # 核心业务逻辑
│   ├── knowledge_base.py           #   SQLite 数据库查询
│   ├── intelligent_search.py       #   多策略智能搜索引擎
│   ├── symptom_pipeline.py         #   症状处理管线
│   ├── symptom_analysis.py         #   症状标准化与证型判断
│   ├── data_desensitize.py         #   数据脱敏（隐私保护）
│   └── result_verify.py            #   结果验证与格式化
│
├── 🖥️ src/desktop/                 # ★ 桌面应用层（CustomTkinter）
│   ├── app.py                      #   主窗口 UI（深色主题）
│   ├── service.py                  #   业务服务层（历史/推荐/NLU）
│   ├── kg_view.py                  #   交互式知识图谱可视化
│   ├── splash.py                   #   两段式启动动画
│   └── runtime.py                  #   运行时工具（控制台隐藏等）
│
├── 🔗 src/kg/                      # 知识图谱
│   ├── builder.py                  #   图谱构建器
│   ├── query.py                    #   图谱查询接口
│   └── recommender.py              #   KG+模型融合推荐
│
├── 🤖 tcm_recommender/             # ★ 深度学习推荐子系统
│   ├── model.py                    #   TCMNet Transformer
│   ├── train.py                    #   训练器
│   ├── recommend.py                #   推理与安全检查
│   ├── data_pipeline.py            #   数据管线
│   └── extended_config.py          #   229症状/30证型配置
│
├── 🧠 models/nlu/qwen/             # Qwen2.5 离线模型文件
│
├── 📁 data/                        # 数据目录
│   ├── tcm_knowledge.db            #   核心数据库
│   ├── kg_cache.pkl                #   知识图谱缓存
│   └── logs/                       #   运行日志
│
├── 📁 venv/                        # ★ 项目内建虚拟环境
│   └── Scripts/python.exe          #   Python 3.12.10
│
├── 📁 python/                      # ★ 便携 Python 3.12.10（绿色自包含）
│   └── python.exe                  #   解压即用，拷到任意机器可运行
│
└── 📁 build/ dist/                 # PyInstaller 打包产出
```

---

## ⚡ 核心流程

### 🔄 用户查询

```
┌──────────────────────────────────────────────────────────────┐
│ 🧑‍⚕️ 输入：「胃部隐痛两天，吃不下饭，反酸嗳气」               │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 🧠 NLU 引擎 (Qwen2.5-0.5B 4-bit)                            │
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
│ 🔗 智能搜索引擎                                               │
│                                                              │
│  ① 原文搜索 → ② 同义词扩展 → ③ 症状搜索 → ④ 证型推理       │
│  ⑤ 去重排序 → 智能摘要                                       │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 🛡️ 禁忌安全检查                                              │
│                                                              │
│  十八反 → 十九畏 → 孕妇禁忌 → 毒性检测 → 证型冲突           │
│  (KG边 + DB字段双路交叉验证)                                 │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 📋 输出整合                                                  │
│                                                              │
│  ✅ 格式化回答（症状/证型/药材/方剂/安全提示）                │
│  ✅ 更新知识图谱子图（交互式可视化）                          │
│  ✅ 保存历史记录（JSON, 最近500条）                           │
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
│  ①  文本子串匹配 (w=1.0)    ②  方剂反推 (w=0.8)            │
│  ③  分类语义 (w=0.6)        ④  功能关键词 (w=0.7/0.5)      │
│  ⑤  别名继承 (w=0.7)        ⑥  症状同义词 (×0.7)           │
│  ⑦  硬编码兜底 (w=0.5)                                      │
│  总计: ~25,000 关联，密度 0.89% → 3.68%                     │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌── 5 阶段样本生成 ────────────────────────────────────────────┐
│  ① 证型驱动 (50%, w=1.0)   ‖ ② 药材驱动 (20%, w=0.8)       │
│  ③ 方剂驱动 (12%, w=0.9)   ‖ ④ 组合增强 (25%, w=0.5)       │
│  ⑤ 负样本 (17%, w=0.3)                                      │
│  总计: ~563,000 条，平均症状 4.2，有方剂标签 75.2%           │
└──────────────────────────────────────────────────────────────┘
```

---

## 🧩 功能模块

### 🖥️ 桌面应用 (`src/desktop/`)

采用 **CustomTkinter** 深色主题 UI，包含四个标签页：

| 标签页 | 功能 |
|:-------|:-----|
| **🏠 首页** | 症状输入框 + 快捷标签 + 智能问答 + 推荐结果展示 |
| **🔗 知识图谱** | 交互式 Canvas 可视化（拖拽/缩放/单击详情/双击展开） |
| **📋 历史记录** | 搜索/查看/导出/删除历史查询记录 |
| **ℹ️ 关于** | 系统信息与使用说明 |

**UI 亮点：**
- 深色专业主题（Photoshop 风格）
- 字体整体放大（基础 17px），widget_scaling 1.25
- 两段式启动动画（文字先隐 → 窗口淡出，60帧丝滑）
- 知识图谱：单击显示节点详情，双击展开邻域，右键菜单

### 🧠 NLU 自然语言理解 (`src/nlu/`)

| 组件 | 说明 |
|:-----|:-----|
| **模型** | Qwen2.5-0.5B-Instruct，4-bit 量化 |
| **显存** | ~1.2GB（GPU 可用时自动启用） |
| **加载** | 本地 `models/nlu/qwen/`，无网络回退 |
| **设备** | 自动检测 CUDA / CPU（无 GPU 时自动切 CPU） |
| **Prompt** | COT 引导 + Few-shot 示例 → JSON 症状数组 |

### 🔗 知识图谱 (`src/kg/`)

| 实体类型 (10 种) | 关系类型 (12 种) |
|:-----------------|:-----------------|
| Symptom, Syndrome, Herb, Formula | belongs_to, recommends_herb/formula |
| PatentMedicine, FolkRemedy | associated_with_herb/formula |
| Category, Meridian, Nature, Flavor | composes, has_nature/flavor/meridian |
| | incompatible_with, co_occurs_with |

**可视交互：**
- 🖱️ **单击**：显示节点详情（含关联数/配伍禁忌/推荐药材等）
- 🖱️ **双击**：展开邻域全部节点
- 🔄 **拖拽**：平移图谱
- 🔍 **滚轮**：缩放视图
- 📋 **右键**：详情/展开/设为中心/复制名称
- 🎨 **边过滤**：按关系类型筛选（全部/证型/药材/方剂/禁忌）
- 📐 **布局切换**：spring / circular / kamada_kawai / shell

### 🛡️ 禁忌安全体系 (6 层)

```
① 配置定义 → ② KG 图注入 → ③ KG 查询 → ④ 症状分析
 → ⑤ 模型安全检查 → ⑥ 桌面 UI 提示（红色标注）
```

---

## 📊 数据资源

<div align="center">

| 🏛️ 资源 | 📏 规模 | 📝 说明 |
|:---------|:-------:|:--------|
| 🌿 药材 | **2,961** 种 | 功效/主治/用法/禁忌/性味归经 |
| 📜 方剂 | **717** 首 | 组成/功效/主治/加减/用法用量 |
| 🩺 症状 | **229** 种（AI）/ **33** 种（桌面快捷） | 脏腑+气血+经络辨证 |
| 🏥 证型 | **30** 种（AI）/ **9** 种（桌面） | 肺/心/肝胆/脾胃/肾/气血津液/复合 |
| 💊 中成药 | **125** 种 | 常见中成药信息 |
| 📖 民间验方 | **216** 个 | 民间单方验方 |
| 🎯 训练样本 | **~563,000** 条 | 7 层关联 + 5 阶段管线 |
| 🕸️ 知识图谱 | **~4,300 节点 / ~63,000 边** | 12 种关系类型 |
| 🔤 口语映射 | **272 条** | 口语→标准症状名 |

</div>

---

## 🖥️ UI 特性

### 启动动画
- 全屏深色背景，金色 `TCMAI` 标题
- **两段式淡出**：文字逐渐融入背景色 → 窗口整体透明度淡出
- 60 帧丝滑动画，3 秒时长

### 主界面布局
```
┌──────────────────────────────────────────────────┐
│ 🌿 中医药智能问答系统                             │
│ 基于深度学习的中医药知识推荐 · 支持症状辨证        │
├──────────────────────────────────────────────────┤
│ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐             │
│ │ 首页  │ │知识图谱│ │历史记录│ │ 关于  │         │
│ └──────┘ └──────┘ └──────┘ └──────┘             │
├──────────────────────────────────────────────────┤
│                                                  │
│  📝 输入症状描述...                    [发送]    │
│                                                  │
│  🔥 [胃痛] [头痛] [发热] [咳嗽] [失眠]           │
│                                                  │
│  ┌──────────────────┐ ┌─────────────────────┐   │
│  │ 📖 回答区域       │ │ 🔗 知识图谱子图      │   │
│  │                   │ │                     │   │
│  │ 证型推断结果       │ │ 交互式 Canvas       │   │
│  │ 药材推荐列表       │ │ 拖拽/缩放/点击      │   │
│  │ 方剂匹配结果       │ │                     │   │
│  │ 安全提示           │ │                     │   │
│  └──────────────────┘ └─────────────────────┘   │
└──────────────────────────────────────────────────┘
```

---

## 🔧 安装与运行

### 📋 环境要求

| 组件 | 最低 | 推荐 |
|:----|:----|:----|
| 🐍 Python | ≥ 3.10 | **3.12** |
| 🪟 系统 | Windows 10 | Windows 11 |
| 🎮 GPU | — | NVIDIA 4GB+ VRAM |
| 💾 内存 | 8GB | 16GB |
| 💿 磁盘 | 5GB | 10GB+（含模型） |

### 🚀 快速启动（推荐：使用项目内建虚拟环境）

项目已预建 `venv/` 目录，所有依赖已安装，直接启动即可：

```bash
# 双击启动
双击「启动中医药智能问答系统.pyw」

# 或命令行启动
venv\Scripts\python.exe launcher.py
```

### 💚 绿色便携部署（U 盘即插即用）

整个项目目录内置了 Python 3.12.10（`python/`），拷到任意 Windows 机器即可运行：

```bash
# 1. 将整个 TCMAI 目录拷贝到目标电脑（U 盘 / 硬盘任意位置）
# 2. 双击启动
双击「启动中医药智能问答系统.pyw」

# 或命令行启动
venv\Scripts\python.exe launcher.py
```

> **首次启动时**：启动器自动检测 `pyvenv.cfg` 的 `home` 路径是否匹配当前位置，
> 若不匹配则自动修正并重启，整个过程约 0.5 秒，无需用户干预。

### 📦 从零搭建

```bash
# 1. 克隆
git clone https://github.com/aba666aba/SRT.git
cd TCMAI

# 2. 创建虚拟环境
python -m venv venv

# 3. 安装依赖
venv\Scripts\python.exe -m pip install -r requirements.txt

# 4. 安装 PyTorch（CPU 版约 122MB，CUDA 版约 2.5GB）
venv\Scripts\python.exe -m pip install torch==2.12.0 ^
  --index-url https://download.pytorch.org/whl/cpu

# 5. 准备 NLU 模型 → 放入 models/nlu/qwen/
#    从魔搭 ModelScope 下载 Qwen2.5-0.5B-Instruct：
#    from modelscope import snapshot_download
#    snapshot_download('Qwen/Qwen2.5-0.5B-Instruct',
#                       local_dir='models/nlu/qwen')

# 6. 启动
venv\Scripts\python.exe launcher.py
```

### 📦 依赖清单

```
torch>=2.0.0              # 深度学习框架
transformers>=4.40.0      # HuggingFace 模型加载
accelerate>=1.0.0         # 分布式加速
numpy>=1.24.0             # 科学计算
scipy>=1.10.0             # 稀疏矩阵（networkx 依赖）
Pillow>=10.0.0            # 图像处理
tqdm>=4.65.0              # 进度条
customtkinter>=5.2.0      # 现代 UI 框架
networkx>=3.0             # 知识图谱
modelscope>=1.30.0        # 模型下载（魔搭）
bitsandbytes>=0.43.0      # 4-bit 量化
pytest>=7.0.0             # 测试（可选）
ruff>=0.8.0               # 代码检查（可选）
```

---

## 🧠 AI 模型

### NLU 引擎 (Qwen2.5-0.5B)

| 配置 | 值 |
|:----|:----|
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct` |
| 来源 | 魔搭 ModelScope |
| 量化 | 4-bit (bitsandbytes)，CPU 回退 float16 |
| 显存 | ~1.2 GB（GPU）|
| 加载 | 本地路径 `models/nlu/qwen/` |
| 输出 | 标准症状名 JSON 数组 |

### TCMNet 推荐模型

| 配置 | 值 |
|:----|:----|
| 架构 | Transformer Encoder + Dual Tower + CrossAttention |
| 参数量 | ~11M (d_model=384, 4层) |
| 显存 | ~2 GB |
| 多头输出 | 证型 Softmax + 药材 BCE + 方剂 Softmax |
| 对比学习 | InfoNCE 拉近症状-药材匹配对 |

### KG 融合推荐

```
fused_score = model_score × 0.6 + kg_score × 0.4
if 双方一致: fused_score += 0.15 （额外加分）
```

---

## 📦 打包部署

使用 PyInstaller 打包为独立桌面应用：

```bash
# 完整离线打包（约 4.7 GB，含 Qwen2.5 模型）
python -m PyInstaller --clean tcmai.spec
```

输出结构：
```
dist/TCMAI_Desktop/
├── 中医药智能问答系统.exe     # 25 MB 启动器
├── _internal/
│   ├── torch/                # PyTorch CPU
│   ├── transformers/         # HuggingFace
│   ├── bitsandbytes/         # 量化
│   ├── data/                 # 数据库与缓存
│   ├── tcm_recommender/      # 推理模型
│   └── models/nlu/qwen/      # Qwen2.5 离线模型 (~2.9 GB)
```

**安装包**：`dist/中医药智能问答系统_Setup.exe`（约 1.37 GB）
- 支持安装到指定目录
- 开始菜单快捷方式
- 三种卸载方式（再次运行安装包 / 卸载程序 / 开始菜单）

---

## 📜 更新日志

### 2026-05-27

- **绿色便携部署**：Python 3.12.10 嵌入项目 `python/` 目录，`venv` 指向本地路径
- **自修复路径**：`launcher.py` 新增 `_fix_venv_home()`，拷到任意位置自动修正 `pyvenv.cfg`
- **强退按钮**：启动画面右下角增加「强制中止」按钮，防止卡死无法退出
- **UI 现代化**：CustomTkinter 深色主题，字体整体放大 +2px，widget_scaling 1.25
- **启动画面**：两段式动画（文字融入背景 → 窗口淡出），60 帧丝滑
- **知识图谱**：单击显示详情 / 双击展开邻域，修复全节点冒泡 bug
- **加载顺序**：启动画面先于主界面构建，覆盖初始化耗时
- **NLU 引擎**：自动检测 CUDA/CPU，无 GPU 时自动回退
- **离线环境**：项目内建 `venv/`，双击 .pyw 自动使用
- **依赖补全**：requirements.txt 补充 accelerate/scipy/Pillow/networkx/modelscope
- **代码清理**：移除 3 个未使用的 import，优化 `_fix_venv_home` 文件读取
- **代码风格**：ruff 自动修复 42 处代码风格问题
- **远程推送**：推送到 GitHub main 分支

### 2026-05-26

- 路径修复：frozen 模式下 `__file__` 统一降级链（TCMAI_ROOT → frozen → __file__）
- 桌面安装包：Setup.exe + ZIP 嵌入 + 三方式卸载
- 打包方案：base 环境打包 + SRT 运行时环境分离
- 证型扩展：17 → 30 证型分类体系
- 智能查询：口语化症状识别 + 多策略搜索融合 + 272 条别名映射

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
