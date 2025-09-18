# 🌿 中药知识图谱与智能处方推荐系统

欢迎来到中药知识图谱与智能处方推荐系统！这是一个完整的系统，集成了数据爬取、清洗、知识图谱构建、图神经网络训练和智能处方推荐功能。本系统旨在为中医研究和临床应用提供智能化支持。

## 🏗️ 系统架构

本系统采用模块化设计，包含以下核心组件：

1. **数据采集模块**
   - `tcm_scraper.py`: 从 [中华药材网](http://www.cnzyao.com) 爬取原始中药数据 🕷️
   
2. **数据处理模块**
   - `tcm_data_cleaner.py`: 清洗和规整原始数据，转换为结构化格式 🧹
   - `tcm_neo4j_importer.py`: 将清洗后的数据导入 Neo4j 图数据库 📥
   
3. **核心处理流水线**
   - `tcm_data_collection_pipeline.py`: 协调各个模块，执行完整的数据处理流程 ⚙️
   
4. **图神经网络模块**
   - `tcm_gnn_components.py`: 定义GNN模型的核心组件
   - `tcm_gnn_input.py`: 处理模型输入数据
   - `tcm_gnn_output.py`: 处理模型输出结果
   - `tcm_complete_gnn_model.py`: 完整的GNN模型定义
   - `tcm_gnn_training.py`: GNN模型训练组件
   - `tcm_gnn_real_training.py`: 使用真实数据进行模型训练的脚本
   - `tcm_neo4j_data_access.py`: 从Neo4j数据库获取训练数据
   
5. **应用接口模块**
   - `tcm_model_interface.py`: 图形用户界面，用于与训练好的模型进行交互 💻
   - `tcm_prescription_main.py`: 命令行处方推荐接口

## 🚀 快速开始

### 📦 安装依赖

在运行系统之前，请确保安装了所需的依赖包：

```bash
pip install requests beautifulsoup4 neo4j lxml torch torch-geometric
```

### ⚙️ 数据库配置

系统需要连接到 Neo4j 数据库。请在以下文件中修改数据库连接信息：

- `tcm_neo4j_importer.py`
- `tcm_neo4j_data_access.py`
- `tcm_data_collection_pipeline.py`

修改以下参数：
- `uri`: Neo4j数据库URI（默认: `"bolt://localhost:7687"`）
- `user`: 数据库用户名（默认: `"neo4j"`）
- `password`: 数据库密码（请替换为实际密码，建议使用环境变量或配置文件安全地存储密码）

### ▶️ 运行系统

#### 方法1: 运行完整数据处理流水线

```bash
python tcm_data_collection_pipeline.py
```

该脚本会依次执行以下步骤：
1. 爬取中药数据 🕷️
2. 清洗和规整数据 🧹
3. 构建知识图谱数据 📊
4. 准备训练数据 📦

#### 方法2: 分步运行各个模块

##### 1. 爬取数据
```bash
python tcm_scraper.py
```

##### 2. 清洗数据
```bash
python tcm_data_cleaner.py
```

##### 3. 导入数据库
```bash
python tcm_neo4j_importer.py
```

#### 方法3: 训练图神经网络模型

```bash
python tcm_gnn_real_training.py --mode train
```

#### 方法4: 使用训练好的模型进行推理

```bash
python tcm_prescription_main.py
```

#### 方法5: 使用图形界面与模型交互

```bash
python tcm_model_interface.py
```

图形界面程序提供了一个用户友好的方式来与训练好的模型进行交互。用户可以输入症状描述，系统会推荐相应的中药处方 💡

## 🧠 模型结构

我们的图神经网络模型包含以下组件：

1. **输入层**: 处理症状输入和知识图谱嵌入
2. **GNN编码器**: 使用GraphSAGE或GAT进行实体表示学习
3. **注意力融合层**: 融合症状表示和知识图谱表示
4. **解码器**: 生成药材推荐
5. **输出层**: 生成最终的处方得分

## 📊 训练过程

1. 从Neo4j数据库加载训练数据
2. 创建知识图谱的PyG图数据对象
3. 初始化GNN模型
4. 使用真实数据训练模型
5. 保存训练好的模型

## 🖥️ 图形界面使用说明

### 环境要求

- Python 3.7+
- PyTorch
- PyTorch Geometric
- Tkinter (通常包含在Python标准库中)

### 使用方法

1. 确保以下文件存在于项目目录中：
   - `tcm_cleaned_data.json` (清洗后的数据文件)
   - `tcm_gnn_model.pth` (训练好的模型权重文件)

2. 运行界面程序：
   ```bash
   python tcm_model_interface.py
   ```

3. 在界面中输入症状描述，点击"推荐处方"按钮或按回车键获取推荐

### 功能说明

- **症状输入**：在输入框中输入患者的症状描述 📝
- **处方推荐**：系统会根据输入的症状推荐相应的中药处方 💊
- **结果显示**：推荐结果按照相关性得分排序显示 📈

## ⚠️ 注意事项

1. **合法爬取**: 系统遵循robots.txt协议，设置合理的请求间隔，确保合法爬取数据。🕷️
2. **数据质量**: 系统包含数据清洗和验证机制，确保导入数据的质量。✅
3. **扩展性**: 系统采用模块化设计，易于扩展新的数据源和功能。🔧
4. **性能**: 对于大量数据导入，系统使用Neo4j的批量导入功能以提高性能。⚡
5. **安全性**: 密码等敏感信息不应硬编码在代码中，应使用环境变量或安全的配置文件存储。🔒
6. **硬件要求**: 模型训练可能需要较长时间，建议在GPU上运行以提高效率。🖥️
7. **专业建议**: 智能推荐结果仅供参考，实际用药请咨询专业中医师。👨‍⚕️

## 🤝 技术支持

如有任何问题，请联系技术支持团队。我们希望这个系统能为中医药的现代化研究贡献一份力量！🌟