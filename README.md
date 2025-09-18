<<<<<<< HEAD
# 中药知识图谱系统

这是一个完整的中药知识图谱系统，包括数据爬取、清洗、构建知识图谱和图神经网络训练的功能。

## 系统架构

1. **tcm_scraper.py**: 数据爬虫模块，从http://www.cnzyao.com爬取中药数据
2. **tcm_data_cleaner.py**: 数据清洗和规整模块，将爬取的原始数据清洗为结构化数据
3. **tcm_neo4j_importer.py**: Neo4j数据导入模块，将清洗后的数据导入到Neo4j图数据库
4. **tcm_data_collection_pipeline.py**: 完整的数据处理流水线，协调各个模块的工作
5. **tcm_gnn_training.py**: 图神经网络训练模块
6. **tcm_gnn_inference.py**: 图神经网络推理模块
7. **tcm_model_interface.py**: 图形用户界面，用于与训练好的模型进行交互

## 安装依赖

在运行系统之前，请确保安装了所需的依赖包：

```bash
pip install requests beautifulsoup4 neo4j lxml torch torch-geometric
```

## 数据库配置

系统需要连接到Neo4j数据库。请在以下文件中修改数据库连接信息：

1. `tcm_neo4j_importer.py`
2. `tcm_data_collection_pipeline.py`

修改以下参数：
- `uri`: Neo4j数据库URI（默认: "bolt://localhost:7687"）
- `user`: 数据库用户名（默认: "neo4j"）
- `password`: 数据库密码（请替换为实际密码，建议使用环境变量或配置文件安全地存储密码）

## 运行系统

### 方法1: 运行完整数据处理流水线

```bash
python tcm_data_collection_pipeline.py
```

该脚本会依次执行以下步骤：
1. 爬取中药数据
2. 清洗和规整数据
3. 构建知识图谱数据
4. 准备训练数据

### 方法2: 分步运行各个模块

#### 1. 爬取数据
```bash
python tcm_scraper.py
```

#### 2. 清洗数据
```bash
python tcm_data_cleaner.py
```

#### 3. 导入数据库
```bash
python tcm_neo4j_importer.py
```

### 方法3: 训练图神经网络模型

```bash
python tcm_gnn_real_training.py --mode train
```

### 方法4: 使用训练好的模型进行推理

```bash
python tcm_prescription_main.py
```

### 方法5: 使用图形界面与模型交互

```bash
python tcm_model_interface.py
```

图形界面程序提供了一个用户友好的方式来与训练好的模型进行交互。用户可以输入症状描述，系统会推荐相应的中药处方。

## 注意事项

1. **合法爬取**: 系统遵循robots.txt协议，设置合理的请求间隔，确保合法爬取数据。
2. **数据质量**: 系统包含数据清洗和验证机制，确保导入数据的质量。
3. **扩展性**: 系统采用模块化设计，易于扩展新的数据源和功能。
4. **性能**: 对于大量数据导入，系统使用Neo4j的批量导入功能以提高性能。
5. **安全性**: 密码等敏感信息不应硬编码在代码中，应使用环境变量或安全的配置文件存储。
=======
SRT项目
开始写代码吧兄弟们
>>>>>>> 9d1b1c97bfc726d4edbada5bf135f65dbca19ada
