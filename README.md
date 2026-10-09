# TCMAI — 中医药智能方剂生成系统

基于 **NLU + 知识图谱 + 深度学习** 三引擎协同的中医药方剂推荐桌面应用。

用户输入自然语言症状描述 → 系统输出推荐证型、药材（含用法用量）、方剂，附带推理路径和知识图谱可视化。

---

## 目录

- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [目录结构](#目录结构)
- [核心模块详解](#核心模块详解)
  - [NLU 引擎](#1-nlu-引擎-nluengine)
  - [智能搜索引擎](#2-智能搜索引擎-intelligentsearch)
  - [知识图谱](#3-知识图谱-tcmknowledgegraph)
  - [推荐模型 TCMNet](#4-推荐模型-tcmnet)
  - [KG 增强推荐器](#5-kg-增强推荐器-kgenhancedrecommender)
  - [症状分析管线](#6-症状分析管线-symptompipeline)
  - [安全与合规](#7-安全与合规)
- [数据管线与训练流程](#数据管线与训练流程)
- [算法对比与选型](#算法对比与选型)
- [快速开始](#快速开始)
- [构建安装包](#构建安装包)
- [许可证](#许可证)

---

## ⚕️ 医疗免责声明（请务必完整阅读）

> **本节为强制阅读内容。使用本系统前，请完整阅读并理解以下全部条款。**

1. **非医疗器械、非诊疗服务**：本系统是面向中医药知识学习与算法研究的**科研原型 / 软件演示**，不是医疗器械，不提供医疗服务，未取得药品、医疗器械或互联网诊疗相关资质。
2. **不构成诊断、处方或用药建议**：系统输出的"证型""方剂""药材""用量""用法"等**全部由算法自动生成**，不构成任何形式的诊断结论、处方或用药建议，不得作为处方、购药、煎服、调整剂量、停药或其他任何医疗决定的依据。
3. **不可替代执业中医师**：中医诊疗须由**执业中医师**四诊合参（望、闻、问、切），并结合个体体质、病史、合并疾病、正在服用的药物及中西药相互作用综合判断。AI 无法完成上述过程，本系统不能、也不应被用于替代执业医师的面诊、检查与治疗。
4. **紧急情况请立即就医**：如出现胸痛、呼吸困难、意识障碍、抽搐、大出血、高热不退、剧烈腹痛、肢体无力或言语不清等急危重症表现，**请立即拨打 120 或前往就近医院急诊**，切勿依赖本系统。**孕产妇、婴幼儿、老年人、肝肾功能不全者、正在服用处方药或抗凝药者，以及肿瘤、自身免疫病等慢性病患者，禁止依据本系统输出自行用药。**
5. **数据与模型的局限性**：知识库由项目组依据公开中医药资料**人工整理并程序化扩充**，**未逐条标注页码与版本出处**，可能存在错漏、过时或与现行药典不一致之处；界面展示的药材剂量/煎服法仅为知识库文本回显，**未参与任何推荐计算**（建模时剂量已被正则剔除，见 `tcm_recommender/data_pipeline.py`），不代表推荐用量。推荐模型基于**算法合成样本**训练，训练标签与推理期关联规则同源，评估存在乐观偏差：独立教科书口径的 Recall@10 约 **22.4%**（`data/ablation_fusion.json`），即**多数推荐结果可能与权威辨证结果不符**；自然语言症状抽取在 CPU 环境下精确匹配率约 **43.3%**、单条推理约 **18.8 秒**（`data/ablation_nlu.json`），存在误抽、漏抽。
6. **AI 会出错（幻觉风险）**：大语言模型与神经网络均可能输出**看似合理但实际错误**的内容，包括不存在的药材/方剂名、错误配伍、错误剂量、错误出处。系统内置安全筛查为**有限规则检查**，**不能保证查全**；历史版本曾以 1.00 置信度将剧毒药「砒石」列入推荐药材（见"开发进度与已知不足"）。**任何输出均须经执业中医师复核后方可讨论使用。**
7. **无人工审核**：本系统**没有**任何执业医师在结果返回前进行审核、复核或签发。
8. **数据与隐私**：本系统**完全离线运行**，不联网、不上传任何数据。为便于回顾，您输入的症状与系统生成的回答会**以明文形式保存在本机** `data/history.json`，运行日志 `data/logs/tcmai.log` 亦可能记录症状文本。**请勿输入姓名、身份证号、手机号、住址等可识别个人信息**；如需清除，请在"历史记录"页删除或直接删除相应文件。
9. **责任限制**：使用本系统即表示您已理解并接受上述全部内容，**因依赖本系统输出而作出的任何医疗决定及其后果，由使用者自行承担**。
10. **许可与用途**：本项目以 AGPL-3.0-or-later 发布（见 [LICENSE](LICENSE)、[NOTICE](NOTICE)），仅供学习、研究与技术交流；**禁止用于临床诊疗、药品销售、健康咨询等任何面向真实患者的商业或医疗用途。**

## 系统架构

```
                          ┌──────────────────────────┐
                          │     CustomTkinter GUI     │
                          │   5标签页（首页/图谱/      │
                          │   知识管理/历史/关于）     │
                          └──────────┬───────────────┘
                                     │ query(text)
                                     ▼
                          ┌──────────────────────────┐
                          │    TCMDesktopService      │
                          │   (service.py)            │
                          │   - 查询路由              │
                          │   - 历史CRUD              │
                          │   - NLU 状态管理          │
                          └──────────┬───────────────┘
                                     │
                     ┌───────────────┼───────────────┐
                     ▼               ▼               ▼
              ┌────────────┐ ┌────────────┐ ┌────────────────┐
              │ NLU Engine │ │ Recommender│ │ Intelligent    │
              │ Qwen2.5    │ │ TCMNet + KG│ │ Search         │
              │ 0.5B 4-bit │ │ 融合推理   │ │ 多策略搜索     │
              └─────┬──────┘ └─────┬──────┘ └──────┬─────────┘
                    │              │                │
                    ▼              ▼                ▼
              ┌─────────────────────────────────────────────┐
              │              数据层                          │
              │  ┌──────────────┐  ┌──────────────────────┐ │
              │  │ SQLite 知识库 │  │ NetworkX MultiDiGraph │ │
              │  │ (1010药材/    │  │ (10节点类型/14边类型) │ │
              │  │  347方剂/     │  │                        │ │
              │  │  63中成药/    │  │                        │ │
              │  │  221验方)     │  │                        │ │
              │  └──────────────┘  └──────────────────────┘ │
              └─────────────────────────────────────────────┘
```

**数据流详述**：

```
用户输入自由文本
    │
    ├─[1]─► NLU Engine (Qwen2.5): 提取结构化症状列表 → ["失眠","乏力","头晕"]
    │         证型展开增强: 若检测到证型名→附加对应症状
    │
    ├─[2]─► TCMRecommender.parse_symptoms(): 别名映射 + 229标准症状匹配
    │
    ├─[3]─► 三路并行:
    │         ├─ TCMNet.predict(): 6层Transformer编码症状 → CrossAttention排序药材/方剂
    │         ├─ KG推理: syndrome_analysis() + symptom_to_recommendation()
    │         └─ IntelligentSearch: 口语扩展 + 证型反推 + 多策略融合搜索
    │
    ├─[4]─► KGEnhancedRecommender.fusion():
    │         model_score × 0.6 + kg_score × 0.4 + boost(0.15 共现加分)
    │
    ├─[5]─► 安全筛查:
    │         十八反/十九畏 → INCOMPATIBLE 边检测
    │         孕妇禁用 → pregnancy_forbidden 属性检查
    │         毒性标记 → toxic 属性检查
    │
    └─[6]─► 输出: 证型 + 药材(含 dosage) + 方剂 + 推理路径 + 安全警告
```

---

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **UI** | CustomTkinter | 深色主题，扁平化控件，1.25x DPI 缩放 |
| **NLU** | Qwen2.5-0.5B (4-bit 量化) | 本地推理，bitsandbytes NF4 量化，封闭词表约束 |
| **推荐模型** | TCMNet (PyTorch) | 6层 Pre-LN Transformer + 双塔 CNN 药材编码 + Cross-Attention 排序 |
| **知识图谱** | NetworkX MultiDiGraph | 10 节点类型，14 边类型，有向多重图 |
| **数据库** | SQLite | 药材/方剂/中成药/验方 四表，LIKE 查询 |
| **训练** | PyTorch + AdamW | 混合精度(AMP)，CosineAnnealingWarmRestarts，监督对比学习 |
| **打包** | PyInstaller + Inno Setup 6 | onedir → Windows 安装包 |
| **可视化** | Tkinter Canvas | 交互式图谱：缩放/拖拽/展开/布局切换/PNG 导出 |

---

## 目录结构

```
TCMAI/
├── 启动中医药智能问答系统.pyw    # 入口（.pyw，无控制台窗口）
├── 启动.bat                      # 开发启动器
├── installer.iss                 # Inno Setup 安装配置
│
├── src/
│   ├── main.py                   # 备用命令行入口
│   │
│   ├── core/                     # 核心业务层
│   │   ├── tcm_config.py         #   硬编码中医药知识（症状/证型/方剂/禁忌/别名映射）
│   │   ├── knowledge_config.py   #   SQLite+硬编码双层配置（读优先DB，写落DB）
│   │   ├── knowledge_base.py     #   知识库访问层（线程安全 SQLite 封装）
│   │   ├── intelligent_search.py #   多策略融合搜索引擎（3步流水线）
│   │   ├── symptom_analysis.py   #   辨证论治分析器（9证型匹配评分）
│   │   ├── symptom_pipeline.py   #   统一症状处理管线（检测→分析→汇总）
│   │   ├── result_verify.py      #   结果合规校验（敏感词过滤+溯源+完整性验证）
│   │   └── data_desensitize.py   #   数据脱敏（含中医药白名单保护）
│   │
│   ├── desktop/                  # 桌面 GUI 层
│   │   ├── app.py                #   主窗口 DesktopApp（5标签页，CTkTabview）
│   │   ├── kg_view.py            #   知识图谱可视化 KnowledgeGraphView（Canvas）
│   │   ├── knowledge_manager.py  #   知识管理面板 KnowledgeManagerView（5配置表 CRUD）
│   │   ├── service.py            #   业务服务层 TCMDesktopService（NLU预加载+推荐推理+历史）
│   │   ├── splash.py             #   启动动画 SplashScreen（渐隐+色彩过渡）
│   │   └── runtime.py            #   Win32 辅助（隐藏控制台+错误弹窗）
│   │
│   ├── kg/                       # 知识图谱层
│   │   ├── __init__.py           #   公开 API：TCMKnowledgeGraph, KnowledgeGraphQuery
│   │   ├── builder.py            #   图谱构建器（SQLite+config→MultiDiGraph）
│   │   ├── query.py              #   图谱查询（实体/关系/推理/最短路径/药材画像）
│   │   └── recommender.py        #   KG增强推荐器 KGEnhancedRecommender（模型+KG融合）
│   │
│   └── nlu/                      # NLU 层
│       ├── __init__.py           #   公开 API：NLUEngine, extract_symptoms, get_engine
│       └── engine.py             #   Qwen2.5 NLU 引擎（本地推理+JSON输出解析）
│
├── tcm_recommender/              # 深度学习推荐模型
│   ├── __init__.py               #   包入口，导出 TCMRecommender
│   ├── extended_config.py        #   229症状 + 30证型定义 + 别名映射 + 毒性药材表
│   ├── data_pipeline.py          #   数据管线 TCMDataPipeline（7层关联矩阵+5阶段采样）
│   ├── data_expansion.py         #   知识库扩充脚本（按中药学分类定义药材/方剂）
│   ├── model.py                  #   TCMNet 模型（SymptomEncoder+HerbEncoder双塔+CrossAttention）
│   ├── train.py                  #   训练脚本（TCMDataset+Trainer+早停+混合精度）
│   └── recommend.py              #   推理接口 TCMRecommender（parse→predict→check_safety→format）
│
├── data/
│   ├── tcm_knowledge.db          # SQLite 知识库（1010药材/347方剂/63中成药/221验方）
│   ├── create_tcm_db.sql         # 数据库建表 DDL
│   ├── enhance_data.py           # 训练数据增强工具
│   └── fill_indications.py       # 方剂主治字段补充工具
│
└── models/nlu/qwen/              # Qwen2.5-0.5B 模型文件
    ├── config.json               #   模型配置（hidden=896, layers=24, heads=14）
    ├── model.safetensors         #   权重文件（~988MB, bfloat16）
    └── chat_template.jinja       #   Qwen2 对话模板
```

---

## 核心模块详解

### 1. NLU 引擎 (`NLUEngine`)

**职责**：将中文自由文本患者描述转化为 229 种标准中医药症状名称的结构化列表。

**模型**：Qwen2.5-0.5B（`hidden_size=896, num_layers=24, num_heads=14, vocab=151936`）。

**量化策略**：
| 运行模式 | 精度 | 显存/内存 |
|----------|------|----------|
| CUDA + 4-bit (bitsandbytes NF4) | 4-bit | ~0.5-1 GB |
| CUDA (fallback) | float16 | ~1-2 GB |
| CPU | float32 | ~2-4 GB |

**提示词工程**：封闭词表约束——系统提示词包含完整 229 症状列表，强制模型"只输出列表中存在的症状名"，将开放生成任务降级为多标签分类。4 个 few-shot 示例覆盖消化/心肺/寒虚/肠胃领域。

**输出解析**：两步策略
1. 主策略：在原始输出中定位 `[...]` 边界，`json.loads()` 解析
2. 回退策略：逐行清洗引号/标点，收集长度 ≥2 的非空行

**证型展开增强**（在 `service.py` 的 `detect_symptoms_nlu()` 中）：检测用户输入中的证型名（如"肾虚""脾虚"），从 `EXTENDED_SYNDROME_DEFINITIONS` 追加对应症状描述，提升症状召回率。

**关键参数**：
| 参数 | 值 |
|------|-----|
| `max_new_tokens` | 256 |
| `temperature` | 0.1 |
| `do_sample` | False（确定性输出） |
| `max_length`（输入） | 2048 tokens |

---

### 2. 智能搜索引擎 (`IntelligentSearch`)

**职责**：多策略融合搜索，提升对口语化查询的匹配能力。三步流水线：

```
Step 1: 查询扩展
  ├── 同义词标准化: 匹配 SYMPTOM_ALIAS_MAP（272条）→ 映射到标准症状
  ├── 直接检测: 匹配 SYMPTOM_LIST 标准症状
  ├── 证型同义扩展: 匹配 TCM_SYNONYM_MAP（53类）
  └── 构建扩展词集: 原文+标准化+别名反向+证型同义+方剂别名+药材别名→去重

Step 2: 证型推理 → 方剂推荐
  └── infer_syndromes(): 评分 = min(1.0, 0.4 + n_symptoms × 0.3)

Step 3: 多策略搜索 + 融合排序
  ├── 原文搜索:  权重 1.0
  ├── 同义词扩展: 权重 0.9
  ├── 症状搜索:  权重 0.85
  └── 证型搜索:  权重 0.8
  └── 按 item_id 去重合并
```

**配置规模**：
| 映射表 | 条目数 |
|--------|--------|
| `SYMPTOM_ALIAS_MAP` | 272 条（口语→标准） |
| `SYMPTOM_TO_FORMULA_MAP` | 58 种症状→方剂推理链 |
| `TCM_SYNONYM_MAP` | 53 类证型/症状同义词 |
| `FORMULA_ALIAS_MAP` | 56 首方剂别名 |
| `HERB_ALIAS_MAP` | 96 味药材别名 |

---

### 3. 知识图谱 (`TCMKnowledgeGraph`)

**图结构**：`networkx.MultiDiGraph`（有向多重图），节点 ID 格式 `{type}:{name}`。

**10 种实体类型**：
`Symptom` | `Syndrome` | `Herb` | `Formula` | `PatentMedicine` | `FolkRemedy` | `Category` | `Meridian` | `Nature` | `Flavor`

**14 种关系类型**：
| 关系 | 方向 | 来源 |
|------|------|------|
| `belongs_to` | symptom → syndrome | extended_config |
| `recommends_herb` | syndrome → herb | extended_config |
| `recommends_formula` | syndrome → formula | extended_config |
| `associated_with_herb` | symptom → herb | 关联矩阵 (weight) |
| `associated_with_formula` | symptom → formula | 关联矩阵 (weight) |
| `composes` | herb → formula | 方剂组成解析 |
| `has_nature` | herb → nature | 药性（9种：寒/热/温/凉/平/微寒/微温/大寒/大热） |
| `has_flavor` | herb → flavor | 药味（7种：酸/苦/甘/辛/咸/淡/涩） |
| `has_meridian` | herb → meridian | 归经（12经络） |
| `belongs_to_category` | herb → category | 药材分类 |
| `incompatible_with` | herb ↔ herb (双向) | 十八反/十九畏 |
| `co_occurs_with` | herb → herb | 共现矩阵（阈值 >0.1） |
| `toxic` | 节点属性标记 | TOXIC_HERBS |
| `pregnancy_forbidden` | 节点属性标记 | 孕妇禁用 |

**查询能力**（`KnowledgeGraphQuery`）：
- `find_nodes(name, type)`: 模糊匹配 + 类型过滤
- `symptom_to_recommendation()`: 单症状→证型→药材/方剂（按 weight 降序）
- `syndrome_analysis(symptoms)`: 多症状证型投票统计
- `herb_profile(name)`: 药材完整画像（性味归经+组成方剂+共现+禁忌+关联症状）
- `recommend_by_symptoms()`: 多症状聚合推荐（证型加权 + 直接关联 0.3 折扣）
- `shortest_path(a, b)`: 无权最短路径

**可视化**（`KnowledgeGraphView`）：
- 交互式 Canvas：缩放（滚轮）、拖拽平移、节点详情（单击）、邻域展开（双击）、右键菜单
- 4 种 NetworkX 布局：spring / circular / kamada_kawai / shell
- 边类型过滤（SegmentedButton 切换）
- 搜索 + 高亮
- 展开历史栈（后退按钮）
- PNG 导出

---

### 4. 推荐模型 TCMNet

**架构**：

```
                     SymptomEncoder
症状输入 [B, L] ──► Embedding + LearnedPosEncoding
                         │
                    6× Pre-LN Transformer Encoder
                    (d_model=384/512, nhead=8, FFN=4×d_model, GELU)
                         │
                    AttentivePooling ──► symptom_vec [B, D]
                         │
          ┌──────────────┼──────────────┬──────────────┐
          ▼              ▼              ▼              ▼
   SyndromeHead    HerbRanker      FormulaRanker   Projection
   (3层MLP)        (CrossAttn)     (CrossAttn)     (→256 dims)
          │              │              │              │
   syndrome_logits  herb_scores   formula_scores  contrastive_loss
   [B, 30]           [B, H]         [B, F]
                         │
                    HerbEncoder (双塔)
                    ┌────文本塔────┐  ┌──属性塔──┐
                    │ CharCNN 2层  │  │ cat_emb   │
                    │ text_dim=256 │  │ nature_emb│
                    │ → d_model    │  │ flavor_emb│
                    └──────┬───────┘  └────┬──────┘
                           └── 融合门 ────┘
                    gate = σ(W·[text;meta])
                    output = gate*text + (1-gate)*meta
                              │
                         herb_vec [H, D]
```

**核心子模块**：

| 模块 | 配置 | 说明 |
|------|------|------|
| `SymptomEncoder` | 6层 Pre-LN Transformer, d_model=384/512, nhead=8 | Pre-LN 将 LayerNorm 放在 Attention/FFN 之前，缓解深层梯度消失 |
| `HerbEncoder`（文本塔） | 2层 MLP + GELU, text_dim=256 | 输入：药材功效+主治文本的中文字符多热编码 |
| `HerbEncoder`（属性塔） | category_emb(64) + nature_emb(64) + flavor_emb(64) | 药材分类+药性+药味三元嵌入 |
| `HerbEncoder`（融合门） | sigmoid gate, output_dim=d_model | 学习文本特征与属性特征的动态融合比例 |
| `CrossAttentionRanker` | MultiheadAttention + 残差 + score_proj | 症状向量作为 Query，药材/方剂向量作为 Key/Value |
| `SyndromeHead` | 3层 MLP: d_model→d_model→d_model/2→30 | 30 证型多分类 |

**损失函数**（`Trainer`）：

| 损失 | 类型 | 权重 | 说明 |
|------|------|------|------|
| 证型分类 | CrossEntropyLoss(label_smoothing=0.05) | 1.0 | ignore_index=0（padding） |
| 方剂排序 | CrossEntropyLoss(label_smoothing=0.05) | 0.5 | 正类为标签方剂 |
| 药材多标签 | 加权 BCE | 0.3 | 负样本权重上限 50，平衡正负样本 |
| 对比学习 | 监督对比损失（多正类） | 0.2 | 余弦相似度 / temperature → logsumexp |

**训练配置**：

| 超参数 | 默认值 |
|--------|--------|
| 优化器 | AdamW (lr=1e-4, β=(0.9, 0.98), weight_decay=1e-5) |
| 调度器 | CosineAnnealingWarmRestarts (T_0=10, T_mult=2) |
| Batch size | 64 |
| Epochs | 50（早停 patience=15） |
| 梯度累积 | 2 |
| 梯度裁剪 | max_norm=2.0 |
| 混合精度 | torch.amp.GradScaler (CUDA only) |
| 温度参数 | 0.07（可学习 nn.Parameter） |

---

### 5. KG 增强推荐器 (`KGEnhancedRecommender`)

**职责**：融合 TCMNet 模型预测与知识图谱推理，生成最终推荐结果。

**融合公式**：
```
fused_score = model_score × 0.6 + kg_score × 0.4
if model_score > 0 AND kg_score > 0:
    fused_score += 0.15  # 模型+KG共同推荐加分
```

**三路 KG 推理**：
| 路径 | 方法 | 得分计算 |
|------|------|---------|
| A. 辨证论治 | `syndrome_analysis()` → 证型推荐药材/方剂 | 匹配症状数/总症状数 加权，归一化到 [0,1] |
| B. 症状直接关联 | `symptom_to_recommendation()` 逐症状 | 原始 score × 0.2 折扣 |
| C. 综合分析 | A 的证型匹配部分 | 提供推理可视化数据 |

**输出结构**：
```python
{
    "symptoms": ["失眠", "心悸", ...],
    "syndrome": {"name": "心脾两虚", "confidence": 0.87},
    "herbs": [{"name": "酸枣仁", "score": 0.92, "model_score": 0.88, "kg_score": 0.95, "path": "模型+证型推荐"}],
    "formulas": [...],
    "reasoning_paths": [{"type": "辨证", "title": "辨证论治路径", "detail": "...", "path": [...]}],
    "kg_insights": {"toxic_warning": [...], "incompatible_warning": [...]},
    "safety": {"warnings": [...], "kg_warnings": [...]}
}
```

---

### 6. 症状分析管线 (`SymptomPipeline`)

**管线**：`detect(关键词匹配) → analyze(标准化+证型推断+禁忌检查+建议生成)`

**证型匹配算法**：
- 匹配率 = `len(matched_symptoms) / len(required_symptoms)`
- 置信度阈值：≥ 0.4 命中
- 按置信度降序排列

**严重程度推断**：基于 `SYMPTOM_INFO` 的 description 关键词（"出血"/"高热"/"神昏"/"剧痛"/"休克" → severity=5，否则=3）。

**9 证型定义**（`SYNDROME_DEFINITIONS`）：用于轻量级症状分析，与推荐模型的 30 证型互补。

---

### 7. 安全与合规

**配伍禁忌检查**（`INCOMPATIBLE_HERBS`）：
- **十八反**：甘草反甘遂/大戟/海藻/芫花；乌头反贝母/瓜蒌/半夏/白蔹/白及；藜芦反人参/沙参/丹参/玄参/细辛/芍药
- **十九畏**：硫黄畏朴硝，水银畏砒霜，狼毒畏密陀僧，巴豆畏牵牛，丁香畏郁金，川乌/草乌畏犀角，牙硝畏三棱，官桂畏赤石脂，人参畏五灵脂
- **孕妇禁用**：节点属性 `pregnancy_forbidden=True`
- **毒性标记**：7 种毒性药材（生川乌/生草乌/附子/巴豆/甘遂/大戟/芫花）

**结果校验**（`ResultVerifier`）：
- 敏感词正则替换（保证治愈/根治/无效退款/绝对有效 等绝对化断言）
- 典籍引用来源标注
- 合规免责声明自动生成
- 回答完整性验证（长度<50判不完整）

**数据脱敏**（`DataDesensitizer`）：
- 五类 PII 检测：身份证号（优先）、手机号、邮箱、地址、姓名
- 中医药白名单保护：~80+ 药材/方剂名防止姓名正则误伤
- SHA256 匿名化 ID 生成

---

## 数据管线与训练流程

### 关联矩阵构建（7层流水线）

`TCMDataPipeline._build_association_matrices()` 构建 `symptom_herb_matrix [229, H]` 和 `symptom_formula_matrix [229, F]`：

| 层 | 策略 | 权重 | 方法 |
|----|------|------|------|
| 1 | 文本精确匹配 | 1.0 | 药材功效/主治文本中查找症状名 |
| 2 | 方剂组成反推 | 0.8 | 方剂含某药材 + 方剂关联某症状 → 药材关联该症状 |
| 3 | 药材分类语义匹配 | 0.6 | 25类药材→对应症状组 |
| 4 | 功能关键词映射 | 0.5-0.7 | ~80个功能关键词（清热/补气/活血…）→ 症状 |
| 5 | 药材别名继承 | 0.7 | 别名药材继承正名药材的关联 |
| 6 | 症状同义词映射 | 源×0.7 | 同义词症状继承源症状关联 |
| 7 | 硬编码兜底 | 0.5 | 对零关联药材/症状手动映射 |

### 训练样本生成（5阶段，500K 样本）

| 阶段 | 占比 | 策略 | 权重 |
|------|------|------|------|
| 1. 证型驱动 | 50% | 从 30 证型定义提取症状+药材+方剂，随机选 2~k 症状，15% 噪声注入 | 1.0 |
| 2. 药材驱动 | 20% | 每种药材选相关症状，补同症状下的其他药材，12% 噪声 | 0.8 |
| 3. 方剂驱动 | 12.5% | 解析方剂组成，匹配症状 | 0.9 |
| 4. 组合增强 | 25% | 随机 2-6 症状，基于关联矩阵加权随机选药材/方剂 | 0.5 |
| 5. 负样本 | ~16.7% | 选零关联药材作负标签 | 0.3 |

**标签约定**：`1.0`=正样本 | `-1.0`=负样本 | `0.0`=未标注（训练时忽略）

### 训练命令

```bash
# 1. 扩充知识库（首次或数据更新后执行）
python tcm_recommender/data_expansion.py

# 2. 生成训练样本
python -m tcm_recommender.data_pipeline \
    --db data/tcm_knowledge.db \
    --output data/train_data.npz \
    --count 500000

# 3. 训练模型
python -m tcm_recommender.train \
    --data data/train_data.npz \
    --output tcm_recommender \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-4
```

---

## 算法对比与选型

中医药方剂推荐是一个典型的**多模态小样本知识推理**问题：症状空间稀疏（229 维组合），先验知识密集（十八反十九畏、性味归经），且对可解释性有刚性需求（医生需要理解推荐依据）。

### 对比方案

| 维度 | 纯规则系统 | 纯知识图谱 | 纯深度学习 | **TCMAI 混合方案** |
|------|-----------|-----------|-----------|-------------------|
| **原理** | 预定义 if-then 规则 | 图遍历+最短路径 | 端到端神经网络 | NLU→KG推理+DL排序→融合 |
| **覆盖率** | 低（规则有限） | 中（依赖图连通性） | 高（泛化学习） | **高**（三引擎互补） |
| **可解释性** | 强（规则透明） | 强（路径可视化） | 弱（黑盒） | **强**（推理路径+图谱可视化） |
| **冷启动抗性** | 需要人工编规则 | 需要构建图 | 需要标注数据 | **中**（KG+规则兜底DL） |
| **模糊症状处理** | 差 | 差 | 好 | **好**（NLU 语义理解） |
| **安全约束** | 易集成 | 易集成（边属性） | 需额外设计 | **内置**（KG边+规则双重筛查） |
| **维护成本** | 高（规则膨胀） | 中 | 低（数据驱动） | **中**（配置与数据分离） |

### 为什么不选其他方案

**纯规则系统**（如传统专家系统）：
- 229 症状 × 30 证型 × 300+ 药材的组合规则无法穷举
- 口语化症状（"吃不下饭""睡不好"）需要维护大量别名规则
- 方剂配伍中的剂量推理无法用简单 if-then 表达

**纯知识图谱**：
- `symptom_herb_matrix` 稀疏性严重：大部分药材与大部分症状零关联（冷启动）
- 图游走无法学习到证型→药材的隐含模式（如"滋阴"类药材的共性）
- 多症状组合查询时缺乏有效的评分融合机制

**纯深度学习**（如直接用 BERT/LLM 端到端推荐）：
- 中医药领域缺乏大规模标注数据（无 ImageNet 级别的药材-症状对齐语料）
- 黑盒输出无法向医生提供诊断依据（"为什么推荐黄连？"）
- LLM 幻觉风险：可能生成不存在的药材名或不安全的配伍
- 无法利用已结构化的中医先验知识（性味归经、十八反十九畏）

### TCMAI 混合方案的优势

1. **NLU 解耦症状提取与推荐**：Qwen2.5 将口语化文本标准化为 229 维症状向量，避免 LLM 直接生成药方的幻觉风险
2. **KG 提供结构化先验**：将中医药典籍知识（性味归经、方剂组成、配伍禁忌）编码为图结构，DL 模型在 KG 约束下学习，而非从零开始
3. **三引擎融合**：模型泛化 + KG 可解释 + 规则安全，各自做最擅长的事
4. **Pre-LN Transformer + 双塔 CNN + Cross-Attention** 的架构设计针对症状稀疏、药材特征异构（文本+属性）的特点
5. **监督对比学习** 显式拉近症状与适配药材在嵌入空间的距离，优于单纯的多标签分类
6. **自然语言控制**（NLU 的封闭词表提示词设计）：将开放式生成约束为分类任务，兼顾灵活性与安全性

---

## 开发进度与已知不足

### 已实现（代码可验证）

- 桌面 GUI 五标签页（首页 / 知识图谱 / 知识管理 / 历史记录 / 关于）与启动动画
- **NLU 为真实小模型推理**：本地 Qwen2.5-0.5B（`models/nlu/qwen/`，权重实测存在），非规则匹配；模型缺失时显式报错而非静默降级
- 知识图谱为真实 `networkx.MultiDiGraph`，10 类节点、12 种边实际写出；十八反/十九畏与孕妇禁用从配置建边
- 图谱查询族：模糊查点 / 邻居 / 证型投票 / 药材画像 / 多症状聚合 / 最短路径；可视化支持缩放、拖拽、PNG 导出
- 推荐模型为完整 Transformer 工程（Pre-LN、注意力池化、双塔、CrossAttention 排序、SupCon 对比损失），训练侧含 AMP / 梯度累积 / 梯度裁剪 / 余弦热重启 / 早停
- 7 层关联矩阵 + 5 阶段合成样本管线；三处 CLI（训练 / 数据 / 推理）；消融实验脚本与结果留档
- 历史记录：文件锁 + 原子替换 + 损坏文件转存；浏览 / 搜索 / 删除 / 清空 / 导出
- 离线打包链路曾有产物（`dist/TCMAI/TCMAI.exe`）

### 未实现 / 与文档不符（如实列出）

- **"三引擎融合"在主链路不成立**：桌面端只调用 `TCMRecommender`，`KGEnhancedRecommender` 仅出现在消融脚本中
- **多策略搜索的权重是装饰性的**：权重被传入但合并时未使用，实际只按 id 去重保留先到者
- **训练超参与随仓库权重严重不符**：README 描述为 6 层 / d_model 384-512 / batch 64 / epochs 50，而 `best_model.json` 记录的是 `d_model=256, num_layers=2, batch=32, epochs=1, samples=100000`，`herb_recall10` 仅 **0.2573**
- **权重不可复现**：`data/train_data.npz`（及 `train_data_v4_small.npz`、`best_model_v4.pth`）在仓库中**不存在**
- 药材编码器文档写"CharCNN"，代码中**无任何 Conv 层**，实为 MLP
- 证型头实际为 **31 类**，文档写 30 类
- **`ResultVerifier`（合规校验 / 绝对化断言过滤）与 `DataDesensitizer`（五类 PII 脱敏）均为死代码**，全库零实例化/零调用
- 历史记录无 Update 操作（仅增删清导），文档称"CRUD"不准确
- 方剂 347 条的 `dosage` 列在 INSERT 时被整列省略，实际为空；`source` 字段被程序统一写成"系统扩充"，缺乏权威出处
- `is_toxic` 曾硬编码为 `False`、毒性清单仅 7 味，导致剧毒药「砒石」以 1.00 置信度进入推荐结果（**已于本次修复，见下**）
- 打包流程开箱即失败：`启动.bat` 不存在，`build_installer.bat` 在 `scripts/` 下且依赖不存在的 `TCMAI.spec`；脚本写死 `D:\Python\python.exe`，实际运行环境是 `D:\Anaconda3\envs\SRT\python39`
- 消融脚本写死他人机器的路径 `E:\TOM\TCMAI`，在本仓库直接运行会 import 错位置
- **无任何单元测试**（无 tests 目录、无 pytest 结构）
- 多个模块为死代码或恒假分支：`symptom_pipeline.py`、`symptom_analysis.py`、`herb_toxic_mask`、`count_herbs()` 等

### 已修复的安全问题（2026-10）

| 问题 | 修复 |
|---|---|
| `is_toxic` 硬编码为 `False`，用户永远看不到毒性标识 | 接上真实毒性判定 `is_toxic_herb()` |
| 毒性清单仅 7 味，不含砒石/朱砂/雄黄等矿物毒药 | 扩充为 **52 味**（矿物类 / 动物类 / 植物类） |
| 剧毒药「砒石」以 1.00 置信度进入推荐结果 | 推荐结果对毒性药材**硬拦截**，被拦截项通过 `blocked_toxic_herbs` 明确告知用户 |
| 含毒性药材的方剂（朱砂安神丸、桂枝附子汤等）无任何提示 | 保留方剂但标注 ☠️ 警示，提示须医师指导 |
| `check_safety` 不返回 `is_safe`，导致每次校验都误报"检测到配伍禁忌" | 补齐 `is_safe`；改用**权威药对表（十八反/十九畏）+ 明确"不宜与…同用"句式**判定 |
| 校验异常时反向输出"未检测到常见配伍禁忌"，形成假安全 | 校验不可用时输出明确警告，不再给出"安全"结论 |

> 药对判定精度实测 10/10；复现实盘查询「腹痛」「乏力，头痛」不再输出砒石。

## 快速开始

### 环境要求

- **Python**: 3.12（推荐 D:\Python\3.12.10）
- **OS**: Windows 10/11
- **GPU**: NVIDIA GPU（可选，推荐 RTX 3050+，4GB+ VRAM 用于 NLU 加速）
- **磁盘**: ~3GB（含 Qwen2.5 模型 ~1GB + SQLite 知识库 + 依赖）

### 安装依赖

```bash
pip install customtkinter networkx torch transformers bitsandbytes numpy tqdm
```

### 启动

```bash
# 推荐：双击启动
启动.bat

# 或直接运行 .pyw（无控制台窗口）
pythonw 启动中医药智能问答系统.pyw

# 开发模式（命令行入口）
python src/main.py
```

### 命令行推理（无 GUI）

```bash
# 基本用法
python -m tcm_recommender.recommend "失眠 乏力 头晕" --device cpu

# 完整参数
python -m tcm_recommender.recommend "头痛 发热 恶寒" \
    --model tcm_recommender/best_model.pth \
    --db data/tcm_knowledge.db \
    --top-herb 10 \
    --top-formula 5 \
    --json
```

---

## 构建安装包

```bash
build_installer.bat
# 输出：installer_output/中医药智能问答系统_Setup.exe
```

构建流程：
1. PyInstaller 将 Python 项目打包为 onedir
2. Inno Setup 6 将 onedir 封装为 Windows 安装包（含开始菜单快捷方式、卸载程序）

需要预先安装 [Inno Setup 6](https://jrsoftware.org/isinfo.php)。

---

## 📄 许可证

本项目采用 **GNU Affero 通用公共许可证第 3 版（AGPL-3.0-or-later）**，全文见 [LICENSE](LICENSE)。

这是一份**强著佐权（copyleft）**许可证，核心要求如下：

- **分发即开源**：你可以自由使用、修改、分发本项目，但一旦对外分发（发布衍生版本、
  发布打包后的安装包等），你的衍生作品**必须同样以 AGPL-3.0 授权并提供完整源代码**。
- **网络服务同样开源**：依据 AGPL-3.0 第 13 条，如果你修改本项目并通过网络提供服务
  （例如部署为在线问答服务），**必须向该服务的所有使用者提供你修改后的对应源代码**。
  这是 AGPL 与 GPL 的关键区别，堵住了「改成 SaaS 就无需开源」的漏洞。
- **保留声明**：分发时须保留版权声明、许可证文本与本项目文件头部的 SPDX 声明。
- **无担保**：本程序按「希望有用」的原则分发，不提供任何形式的担保。

### 第三方组件

AGPL-3.0 只覆盖本项目作者原创的代码（`src/`、`tcm_recommender/`、`scripts/` 等）。
项目依赖的 PyTorch（BSD 风格）、Transformers（Apache-2.0）、Qwen2.5 模型权重
（Apache-2.0）、NetworkX（BSD-3）、CustomTkinter（MIT）、bitsandbytes（MIT）
等第三方组件适用其各自的许可证，**不受本项目 AGPL-3.0 约束，也不能被本项目重新授权**。
详见 [NOTICE](NOTICE)。

### 免责声明

本项目仅供学习研究。系统生成的药方不作为医疗建议，仅供参考，用药请遵医嘱。
