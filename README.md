# AnimeQA 动漫问答微调项目

一个基于 LoRA 微调技术的动漫问答模型项目，使用 microsoft/DialoGPT-small 作为基础模型，专门针对动漫相关问答进行优化。

> 数据集来源：[AniPersonCaps](https://hf-mirror.com/datasets/mrzjy/AniPersonaCaps/tree/main)


## 项目特色

- **轻量化微调**: 使用 LoRA (Low-Rank Adaptation) 技术，高效微调预训练模型
- **快速验证**: 内置超快速测试模式，50个训练样本 + 10个验证样本，2-5分钟完成训练
- **自动化流程**: 从数据处理到模型训练的完整自动化管道
- **配置灵活**: 支持测试配置和生产配置，方便开发调试

## 快速开始

### 环境要求

- Python 3.8+
- 至少 8GB 内存
- 磁盘空间 5GB+

### 安装依赖

```bash
# 克隆项目
git clone <your-repo-url>
cd llm-finetune

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows

# 安装依赖
make install
# 或
pip install -r requirements.txt
```

### 快速测试

```bash
# 一键运行完整测试（数据准备 + 模型训练）
make test-quick

# 只测试数据准备
make test-data-only

# 只测试训练过程
make test-train-only
```

### 完整训练流程

```bash
# 1. 数据准备
make prepare-data

# 2. 开始训练
make train

# 3. 评估模型
make evaluate
```

## 项目结构

```
llm-finetune/
├── config/                 # 配置文件
│   ├── config.yaml        # 主配置
│   └── config.test.yaml   # 测试配置
├── data/                   # 数据模块
│   ├── data_processor.py   # 数据处理器
│   └── dataset_builder.py  # 数据集构建器
├── database/              # 数据库模块
│   ├── models.py          # 数据库模型
│   └── init_db.py         # 数据库初始化
├── models/                # 模型模块
│   └── model_loader.py    # 模型加载器
├── training/              # 训练模块
│   ├── config.py          # 训练配置
│   └── trainer.py         # 训练器
├── scripts/               # 脚本目录
│   ├── prepare_data.py    # 数据准备脚本
│   ├── finetune.py        # 模型微调脚本
│   └── evaluate.py        # 模型评估脚本
├── quick_test.py          # 快速测试脚本
├── create_test_data.py    # 测试数据生成器
├── Makefile              # 自动化命令
└── requirements.txt      # 项目依赖
```

## 主要功能

### 1. 数据处理
- 支持本地 JSONL 格式数据
- 自动生成问答对
- 数据清洗和验证
- SQLite 数据库存储

### 2. 模型微调
- 基于 microsoft/DialoGPT-small 模型
- LoRA 微调技术（参数效率高）
- 支持 CPU 和 GPU 训练
- 自动保存检查点

### 3. 训练监控
- 实时训练日志
- 训练进度跟踪
- 早停机制
- 内存使用监控

### 4. 便捷命令

```bash
# 环境管理
make check-env           # 检查环境
make setup-dirs         # 创建目录
make clean-cache        # 清理缓存

# 数据相关
make prepare-data       # 准备训练数据
make reset-data         # 重置数据库

# 训练相关
make train-small        # 小模型训练
make train-medium       # 中等模型训练
make train-resume       # 恢复训练

# 测试相关
make test-quick         # 快速测试
make clean-test-data    # 清理测试数据

# 其他工具
make logs              # 查看日志
make status            # 项目状态
make clean-all         # 清理所有
```

## 配置说明

### 测试配置 (config.test.yaml)
```yaml
training:
  max_train_samples: 50      # 训练样本数
  max_val_samples: 10        # 验证样本数
  batch_size: 4              # 批次大小
  num_epochs: 1              # 训练轮数
  max_length: 128            # 序列长度
```

### 生产配置 (config.yaml)
```yaml
training:
  batch_size: 4
  learning_rate: 0.00005
  num_epochs: 3
  max_length: 512
  lora_config:
    rank: 16
    alpha: 32
    dropout: 0.1
```

## 测试数据示例

> 数据集来源：[AniPersonCaps](https://hf-mirror.com/datasets/mrzjy/AniPersonaCaps/tree/main)

项目内置了动漫问答测试数据：

```json
{
  "question": "鸣人的梦想是什么？",
  "answer": "鸣人的梦想是成为火影，保护木叶村的所有人。",
  "character": "鸣人",
  "anime": "火影忍者"
}
```

## 常见问题

### Q: 内存不足怎么办？
A: 调整 `batch_size` 和 `max_length` 参数，或使用更小的模型。

### Q: 训练速度慢怎么办？
A: 可以使用 GPU 或减少 `max_length` 和训练样本数。

### Q: 如何添加自己的数据？
A: 修改 `create_test_data.py` 或准备 JSONL 格式的数据文件。

### Q: 如何恢复中断的训练？
A: 使用 `make train-resume CHECKPOINT=检查点路径`。

## 📈 后续计划

- [ ] API 接口服务开发
- [ ] 缓存支持
- [ ] 推理性能优化
- [ ] 更多基础模型支持
- [ ] Web 界面开发
- [ ] Docker 容器化部署


## 许可证

本项目采用 MIT 许可证。


```bash
# 立即体验
make test-quick
```