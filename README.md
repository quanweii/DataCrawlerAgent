# 数据爬取多Agent系统

一个基于多Agent架构的数据爬取和处理系统，专门用于收集论文PDF并构建多模态属性图数据集。

## 系统架构

```
DataCrawlerOrchestrator (主协调器)
    ├── PaperCollectorAgent (论文收集Agent)
    │   └── 负责论文PDF下载和多模态数据提取
    ├── DataCleanerAgent (数据清洗Agent)
    │   └── 负责数据质量检查和清洗
    └── DatasetFormatterAgent (数据集封装Agent)
        └── 负责转换为多模态属性图格式
```

## 执行流程

1. **论文收集** → 下载PDF，提取文本/图像/表格/公式
2. **数据清洗** → 去重、验证、标准化
3. **数据封装** → 构建属性图，保存数据集

## 快速开始

### 1. 安装依赖

```bash
cd /home/dev_qw/DataCrawlerAgent
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key
```

### 3. 运行

```bash
# 交互模式
python main.py --interactive

# 使用示例配置运行
python main.py --sample

# 使用配置文件
python main.py --config task_config_example.json
```

## 项目结构

```
DataCrawlerAgent/
├── agents/                 # Agent模块
│   ├── __init__.py
│   ├── base_agent.py       # Agent基类
│   ├── paper_collector.py  # 论文收集Agent
│   ├── data_cleaner.py     # 数据清洗Agent
│   └── dataset_formatter.py # 数据集封装Agent
├── tools/                  # 工具模块
│   ├── __init__.py
│   ├── mcp_tools.py        # MCP工具管理器
│   └── pdf_extractor_tool.py # PDF多模态提取工具
├── prompts/                # 提示词
│   ├── paper_collector_prompt.txt
│   ├── data_cleaner_prompt.txt
│   └── dataset_formatter_prompt.txt
├── data/                   # 数据目录
│   ├── papers/            # 论文PDF
│   ├── extracted/         # 提取的多模态数据
│   ├── cleaned/           # 清洗后数据
│   └── datasets/          # 最终数据集
├── logs/                   # 日志文件
├── main.py                 # 主入口
├── orchestrator.py         # 主协调器
├── task_config_example.json # 示例配置文件
├── .env.example           # 环境变量示例
└── requirements.txt       # 依赖列表
```

## 核心MCP工具

### PDF多模态提取工具 (extract_pdf_multimodal)

这是核心工具，调用 [pdf-extract-kit](../pdf-extract-kit) 进行论文PDF的多模态数据提取。

**功能：**
- 布局检测（标题、正文、图片、表格、公式等）
- OCR文字识别
- 公式检测与识别（转LaTeX）
- 图片/表格提取

**输入：** PDF文件路径

**输出结构：**
```
output_dir/
└── {paper_name}/              # 以论文名称命名的文件夹
    ├── {paper_name}.md        # Markdown文件
    ├── {paper_name}.npy       # NPY格式数据
    └── images/                # 图片文件夹
        ├── 0.png
        ├── 1.png
        └── ...
```

**使用示例：**

```python
# 方式1：通过配置文件
# task_config.json
{
    "paper_sources": [
        {
            "type": "local",
            "path": "/path/to/paper.pdf",
            "name": "my_paper"
        }
    ],
    "extract_multimodal": true,
    "output_dir": "./data/extracted",
    "device": "cuda:0"
}

# 方式2：直接调用工具
from tools import MCPToolManager

manager = MCPToolManager()
result = manager.call_tool(
    "extract_pdf_multimodal",
    pdf_path="/path/to/paper.pdf",
    output_dir="./output",
    paper_name="my_paper",
    device="cuda:0"
)
```

### 其他MCP工具

- `download_arxiv_paper` - 下载arXiv论文
- `download_doi_paper` - 通过DOI下载论文
- `batch_extract_pdf_multimodal` - 批量提取多个PDF
- `clean_text` - 文本清洗
- `deduplicate_data` - 数据去重
- `validate_data` - 数据验证
- `format_attribute_graph` - 格式化属性图
- `save_dataset` - 保存数据集

## 任务配置格式

```json
{
    "paper_sources": [
        {
            "type": "local",
            "path": "/path/to/paper.pdf",
            "name": "paper_name"
        },
        {
            "type": "arxiv",
            "id": "2301.07041",
            "title": "论文标题"
        }
    ],
    "extract_multimodal": true,
    "output_dir": "./data/extracted",
    "device": "cuda:0",
    "cleaning_rules": {
        "min_text_length": 50,
        "remove_special_chars": true
    },
    "dataset_name": "my_dataset",
    "output_format": "json"
}
```

## 数据集输出格式

最终输出的多模态属性图数据集格式：

```json
{
  "nodes": [
    {
      "id": "paper_0",
      "type": "paper",
      "attributes": {"title": "...", "source": {...}},
      "modalities": ["text"]
    },
    {
      "id": "image_1",
      "type": "image",
      "attributes": {"path": "...", "source_paper": "..."},
      "modalities": ["image"]
    }
  ],
  "edges": [
    {
      "id": "edge_0",
      "source": "paper_0",
      "target": "image_1",
      "type": "contains",
      "attributes": {"relationship": "paper_contains_modality"}
    }
  ],
  "metadata": {
    "name": "dataset_name",
    "version": "1.0.0",
    "created_at": "...",
    "statistics": {...}
  }
}
```

## 依赖说明

本项目依赖以下外部工具：

- **pdf-extract-kit**: 位于 `/home/dev_qw/pdf-extract-kit`，用于PDF多模态提取
  - 布局检测模型: DocLayout-YOLO
  - 公式检测模型: YOLOv8
  - 公式识别模型: UniMERNet
  - OCR模型: PaddleOCR

## 参考

- 本项目架构参考了 [XianyuAutoAgent](../XianyuAutoAgent) 的多Agent设计模式
- PDF提取基于 [pdf-extract-kit](../pdf-extract-kit)
