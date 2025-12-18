# Seq2Seq NER+RE with LoRA Fine-tuning

基于 PromptCLUE 的序列到序列命名实体识别和关系抽取模型，使用 LoRA 微调技术。

## 项目简介

本项目使用 Seq2Seq 模型（T5架构）进行命名实体识别（NER）和关系抽取（RE）任务，采用 LoRA（Low-Rank Adaptation）微调技术，大幅减少训练参数和显存占用。

## 主要特性

- ✅ 基于 PromptCLUE（中文 T5 模型）
- ✅ LoRA 微调（参数量减少 98%+）
- ✅ 支持 CUDA GPU 加速
- ✅ 序列到序列生成式 NER+RE
- ✅ 支持多种医疗领域关系类型

## 项目结构

```
seq2seq_ner_re/
├── config.py              # 配置文件
├── requirements.txt        # 依赖列表
├── scripts/
│   ├── train.py          # 训练脚本
│   └── generate_data.py   # 数据生成脚本
├── models/
│   ├── dataset.py        # 数据集类
│   └── inference.py      # 推理类
├── data/
│   ├── train_seq2seq.txt # 训练数据
│   └── test_seq2seq.txt  # 测试数据
└── clueAI/               # PromptCLUE 模型文件
```

## 环境配置

### 1. 创建虚拟环境

```bash
conda create -n seq2seq_env python=3.11 -y
conda activate seq2seq_env
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
python scripts/train.py
```

### 配置说明

主要配置在 `config.py` 中：

- `USE_LORA`: 是否使用 LoRA 微调（默认 True）
- `LORA_R`: LoRA rank（默认 16）
- `BATCH_SIZE`: 批次大小（默认 8）
- `MAX_LENGTH`: 最大输入长度（默认 512）
- `MAX_TARGET_LENGTH`: 最大输出长度（默认 256）

## LoRA 微调优势

| 特性 | 全量微调 | LoRA微调 |
|------|---------|---------|
| 可训练参数 | 100% | ~1.4% |
| 显存占用 | 高 | 低（约1/3）|
| 训练速度 | 慢 | 快 |
| 模型文件 | GB级 | MB级 |
| 效果 | 好 | 接近全量微调 |

## 数据格式

训练数据格式：
```
文本: [输入文本] 关系类型: [关系类型列表] <SEP> (实体1, 关系, 实体2); (实体3, 关系, 实体4)
```

示例：
```
文本: 肺泡蛋白质沉积症有什么症状？ 关系类型: 疾病-症状 <SEP> (肺泡蛋白质沉积症, 症状, 紫绀); (肺泡蛋白质沉积症, 症状, 胸痛)
```

## 支持的关系类型

- 疾病-症状
- 疾病-药品
- 疾病-食物
- 疾病-并发症
- 症状-疾病
- 药品-疾病
- 检查项目-疾病
- 等13种关系类型

## 注意事项

1. 模型文件（`pytorch_model.bin`）较大，已添加到 `.gitignore`
2. 首次运行需要下载或准备 PromptCLUE 模型
3. 确保有足够的显存（建议 8GB+）或使用 CPU 训练

## License

MIT License

## 参考

- [PromptCLUE](https://github.com/CLUEbenchmark/PromptCLUE)
- [PEFT](https://github.com/huggingface/peft)
- [Transformers](https://github.com/huggingface/transformers)

