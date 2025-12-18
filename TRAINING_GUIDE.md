# Seq2Seq NER+RE 模型训练详细文档

## 目录

1. [项目概述](#项目概述)
2. [环境配置](#环境配置)
3. [数据准备](#数据准备)
4. [配置说明](#配置说明)
5. [训练流程](#训练流程)
6. [训练参数详解](#训练参数详解)
7. [LoRA微调详解](#lora微调详解)
8. [训练监控](#训练监控)
9. [模型保存与加载](#模型保存与加载)
10. [常见问题](#常见问题)
11. [性能优化建议](#性能优化建议)

---

## 项目概述

本项目基于 **PromptCLUE**（中文 T5 模型）实现序列到序列（Seq2Seq）的命名实体识别（NER）和关系抽取（RE）任务。采用 **LoRA（Low-Rank Adaptation）** 微调技术，大幅减少训练参数和显存占用。

### 主要特性

- ✅ 基于 PromptCLUE（中文 T5 模型）
- ✅ LoRA 微调（参数量减少 98%+）
- ✅ 支持 CUDA GPU 加速
- ✅ 序列到序列生成式 NER+RE
- ✅ 支持多种医疗领域关系类型（13种）

### 项目结构

```
seq2seq_ner_re/
├── config.py                 # 配置文件
├── requirements.txt           # 依赖列表
├── scripts/
│   ├── train.py              # 训练脚本
│   └── generate_data.py      # 数据生成脚本
├── models/
│   ├── dataset.py            # 数据集类
│   └── inference.py          # 推理类
├── data/
│   ├── train_seq2seq.txt     # 训练数据
│   └── test_seq2seq.txt      # 测试数据
├── clueAI/                   # PromptCLUE 模型文件
├── saved_model/              # 训练保存的模型
└── logs/                     # 训练日志
```

---

## 环境配置

### 方式1: 使用 Docker（推荐）

#### CPU 版本

```bash
# 构建镜像
docker-compose build

# 运行训练
docker-compose up

# 后台运行
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止容器
docker-compose down
```

#### GPU 版本（需要 NVIDIA GPU 和 nvidia-docker）

```bash
# 构建 GPU 镜像
docker-compose -f docker-compose.gpu.yml build

# 运行训练（GPU）
docker-compose -f docker-compose.gpu.yml up

# 使用便捷脚本（自动检测 GPU）
./docker-run.sh
```

**GPU 版本前置要求：**
- NVIDIA GPU（建议显存 8GB+）
- NVIDIA Docker Runtime（nvidia-docker2）
- Docker Compose

### 方式2: 本地环境

#### 1. 创建虚拟环境

```bash
# 使用 conda
conda create -n seq2seq_env python=3.11 -y
conda activate seq2seq_env

# 或使用 venv
python3.11 -m venv seq2seq_env
source seq2seq_env/bin/activate  # Linux/Mac
# 或
seq2seq_env\Scripts\activate  # Windows
```

#### 2. 安装依赖

```bash
# 安装 PyTorch（根据你的 CUDA 版本选择）
# CUDA 11.8
pip install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU 版本
pip install torch==2.2.2 torchvision torchaudio

# 安装其他依赖
pip install -r requirements.txt
```

#### 3. 验证安装

```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "from transformers import T5Tokenizer; print('Transformers安装成功')"
python -c "from peft import LoraConfig; print('PEFT安装成功')"
```

---

## 数据准备

### 数据格式

训练数据格式为：`input_text <SEP> output_text`

**输入格式：**
```
文本: [输入文本] 关系类型: [关系类型]
```

**输出格式：**
```
(实体1, 关系, 实体2); (实体3, 关系, 实体4)
```

**完整示例：**
```
文本: 肺泡蛋白质沉积症有什么症状？ 关系类型: 疾病-症状 <SEP> (肺泡蛋白质沉积症, 症状, 紫绀); (肺泡蛋白质沉积症, 症状, 胸痛)
```

### 生成训练数据

如果已有 `medical.json` 文件，可以使用数据生成脚本：

```bash
python scripts/generate_data.py
```

脚本会从 `medical.json` 生成 `data/train_seq2seq.txt` 训练文件。

### 数据文件位置

- **训练数据**: `data/train_seq2seq.txt`
- **验证数据**: `data/dev_seq2seq.txt`（可选）
- **测试数据**: `data/test_seq2seq.txt`（可选）

### 数据统计

训练前建议检查数据：

```bash
# 统计训练样本数量
wc -l data/train_seq2seq.txt

# 查看前几行数据
head -n 5 data/train_seq2seq.txt
```

---

## 配置说明

所有配置在 `config.py` 文件中，主要配置项如下：

### 基础配置

```python
# 模型配置
MODEL_NAME = os.path.join(BASE_DIR, 'clueAI')  # PromptCLUE 模型路径

# 数据路径
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_seq2seq.txt')
DEV_DATA_PATH = os.path.join(DATA_DIR, 'dev_seq2seq.txt')

# 输出路径
MODEL_DIR = os.path.join(BASE_DIR, 'saved_model')  # 模型保存目录
LOG_DIR = os.path.join(BASE_DIR, 'logs')  # 日志目录
```

### 训练参数

```python
MAX_LENGTH = 512              # 最大输入长度
MAX_TARGET_LENGTH = 256       # 最大输出长度
BATCH_SIZE = 8                # 批次大小
LEARNING_RATE = 2e-4          # 学习率
NUM_EPOCHS = 4                 # 训练轮数
WARMUP_STEPS = 500            # 预热步数
SAVE_STEPS = 1000             # 保存步数
LOGGING_STEPS = 100           # 日志步数
EVAL_STEPS = 500              # 评估步数
GRADIENT_ACCUMULATION_STEPS = 1  # 梯度累积步数
```

### LoRA 配置

```python
USE_LORA = True               # 是否使用 LoRA 微调
LORA_R = 8                    # LoRA rank（控制参数量）
LORA_ALPHA = 16               # LoRA alpha（通常为 r 的 1-2 倍）
LORA_DROPOUT = 0.05           # LoRA dropout 率
LORA_TARGET_MODULES = ["q", "k", "v", "o"]  # 目标模块
LORA_BIAS = "none"            # bias 训练策略
```

### 设备配置

```python
CUDA_DEVICE_ID = 7            # 指定使用的 CUDA 设备 ID
DEVICE = f"cuda:{CUDA_DEVICE_ID}" if torch.cuda.is_available() else "cpu"
```

### 推理参数

```python
NUM_BEAMS = 4                 # Beam search 数量
EARLY_STOPPING = True         # 早停
DO_SAMPLE = False             # 是否采样
```

---

## 训练流程

### 1. 检查环境

```bash
# 检查 Python 版本（需要 3.8+）
python --version

# 检查 CUDA（如果使用 GPU）
nvidia-smi

# 检查依赖
pip list | grep -E "torch|transformers|peft"
```

### 2. 准备数据

确保训练数据文件存在：

```bash
ls -lh data/train_seq2seq.txt
```

如果没有数据，运行数据生成脚本：

```bash
python scripts/generate_data.py
```

### 3. 检查模型文件

确保 PromptCLUE 模型文件存在：

```bash
ls -lh clueAI/
# 应该包含：
# - config.json
# - pytorch_model.bin
# - spiece.model
# - spiece.vocab
```

### 4. 开始训练

```bash
python scripts/train.py
```

### 5. 训练过程

训练脚本会依次执行：

1. **加载模型和 Tokenizer**
   - 加载 PromptCLUE 模型
   - 配置 pad_token

2. **配置 LoRA**（如果启用）
   - 创建 LoRA 配置
   - 应用 LoRA 到模型
   - 打印可训练参数信息

3. **准备数据集**
   - 加载训练数据
   - 加载验证数据（如果存在）

4. **设置训练参数**
   - 配置学习率、批次大小等
   - 设置 GPU/CPU 设备

5. **创建 Trainer**
   - 使用 HuggingFace Trainer
   - 配置数据整理器

6. **开始训练**
   - 自动保存检查点
   - 记录训练日志

7. **保存最终模型**
   - LoRA 模式：保存 LoRA 权重和合并模型
   - 全量微调：直接保存完整模型

---

## 训练参数详解

### 批次大小（BATCH_SIZE）

- **默认值**: 8
- **说明**: 每个 GPU 的批次大小
- **调整建议**:
  - GPU 显存 8GB: 4-8
  - GPU 显存 16GB: 8-16
  - GPU 显存 24GB+: 16-32
  - CPU 训练: 2-4

### 梯度累积（GRADIENT_ACCUMULATION_STEPS）

- **默认值**: 1
- **说明**: 梯度累积步数，等效批次大小 = BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS
- **使用场景**: 显存不足时，通过梯度累积增大有效批次大小
- **示例**: BATCH_SIZE=4, GRADIENT_ACCUMULATION_STEPS=4 → 等效批次大小=16

### 学习率（LEARNING_RATE）

- **默认值**: 2e-4
- **说明**: 
  - 全量微调: 2e-4 到 5e-4
  - LoRA 微调: 1e-3 到 5e-3（通常比全量微调高 5-10 倍）
- **调整建议**:
  - 如果损失不下降：降低学习率（除以 2）
  - 如果损失震荡：降低学习率
  - 如果收敛太慢：提高学习率（乘以 1.5）

### 训练轮数（NUM_EPOCHS）

- **默认值**: 4
- **说明**: 完整遍历训练集的次数
- **调整建议**:
  - 小数据集（<10K）: 10-20 轮
  - 中等数据集（10K-100K）: 5-10 轮
  - 大数据集（>100K）: 3-5 轮

### 预热步数（WARMUP_STEPS）

- **默认值**: 500
- **说明**: 学习率从 0 线性增加到设定值的步数
- **自动计算**: 训练脚本会自动限制 warmup_steps 不超过总步数的 10%
- **公式**: `warmup_steps = min(设定值, 总步数 // 10)`

### 最大长度（MAX_LENGTH / MAX_TARGET_LENGTH）

- **MAX_LENGTH**: 512（输入最大长度）
- **MAX_TARGET_LENGTH**: 256（输出最大长度）
- **调整建议**:
  - 如果数据较长：增加 MAX_LENGTH
  - 如果输出三元组较多：增加 MAX_TARGET_LENGTH
  - 注意：增加长度会显著增加显存占用

### 保存和评估步数

- **SAVE_STEPS**: 1000（每 N 步保存一次模型）
- **EVAL_STEPS**: 500（每 N 步评估一次）
- **LOGGING_STEPS**: 100（每 N 步记录一次日志）

---

## LoRA微调详解

### 什么是 LoRA？

LoRA（Low-Rank Adaptation）是一种参数高效的微调方法，通过在预训练模型的权重矩阵旁添加低秩矩阵来实现微调，而不是更新所有参数。

### LoRA 优势

| 特性 | 全量微调 | LoRA微调 |
|------|---------|---------|
| 可训练参数 | 100% | ~1-2% |
| 显存占用 | 高 | 低（约1/3）|
| 训练速度 | 慢 | 快 |
| 模型文件 | GB级 | MB级（仅权重）|
| 效果 | 好 | 接近全量微调 |

### LoRA 参数说明

#### LORA_R（Rank）

- **默认值**: 8
- **说明**: 低秩矩阵的秩，控制参数量
- **影响**:
  - 越大：参数量越多，效果可能更好，但训练更慢
  - 越小：参数量越少，训练更快，但可能欠拟合
- **推荐值**: 4-16

#### LORA_ALPHA

- **默认值**: 16
- **说明**: LoRA 的缩放因子，通常设为 r 的 1-2 倍
- **公式**: 实际缩放 = alpha / r
- **推荐值**: r 的 1-2 倍

#### LORA_DROPOUT

- **默认值**: 0.05
- **说明**: LoRA 层的 dropout 率，防止过拟合
- **推荐值**: 0.05-0.1

#### LORA_TARGET_MODULES

- **默认值**: `["q", "k", "v", "o"]`
- **说明**: 应用 LoRA 的目标模块
- **选项**:
  - `["q", "k", "v", "o"]`: Attention 的所有矩阵（推荐）
  - `["q", "v"]`: 只应用 Q 和 V 矩阵（参数更少）
  - `["q", "k", "v", "o", "wi", "wo"]`: 包括 FFN 层（参数更多）

#### LORA_BIAS

- **默认值**: `"none"`
- **说明**: 是否训练 bias
- **选项**:
  - `"none"`: 不训练 bias（推荐）
  - `"all"`: 训练所有 bias
  - `"lora_only"`: 只训练 LoRA 的 bias

### LoRA 模型保存

训练完成后会保存两种模型：

1. **LoRA 权重** (`final_model/`)
   - 只包含 LoRA 权重（MB 级）
   - 需要基础模型才能使用

2. **合并模型** (`final_model_merged/`)
   - LoRA 权重已合并到基础模型
   - 可以直接使用，无需基础模型

### 使用 LoRA 模型推理

```python
from models.inference import Seq2SeqNER_RE

# 使用合并模型（推荐）
model = Seq2SeqNER_RE(model_path='saved_model/final_model_merged')

# 或使用 LoRA 权重（需要基础模型）
model = Seq2SeqNER_RE(model_path='saved_model/final_model')
```

---

## 训练监控

### 日志文件

训练日志保存在 `logs/` 目录：

```bash
# 查看最新日志
ls -lt logs/ | head -n 5

# 实时查看日志
tail -f logs/training_*.log
```

### 训练指标

训练过程中会记录以下指标：

- **loss**: 训练损失
- **eval_loss**: 验证损失（如果有验证集）
- **learning_rate**: 当前学习率
- **epoch**: 当前轮数
- **step**: 当前步数

### 使用 TensorBoard（可选）

如果需要可视化训练过程，可以安装 TensorBoard：

```bash
pip install tensorboard

# 启动 TensorBoard
tensorboard --logdir logs/

# 在浏览器打开 http://localhost:6006
```

### 检查点文件

训练过程中会定期保存检查点：

```
saved_model/
├── checkpoint-1000/    # 第 1000 步的检查点
├── checkpoint-2000/    # 第 2000 步的检查点
├── checkpoint-3000/    # 第 3000 步的检查点
├── final_model/        # 最终模型（LoRA 权重）
└── final_model_merged/ # 最终模型（合并后）
```

**注意**: 默认只保留最近 3 个检查点（`save_total_limit=3`）

---

## 模型保存与加载

### 保存格式

#### LoRA 模式

```
saved_model/
├── final_model/              # LoRA 权重
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer files...
└── final_model_merged/       # 合并后的完整模型
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer files...
```

#### 全量微调模式

```
saved_model/
└── final_model/              # 完整模型
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer files...
```

### 加载模型

#### 方式1: 使用推理类（推荐）

```python
from models.inference import Seq2SeqNER_RE

# 加载模型
model = Seq2SeqNER_RE(model_path='saved_model/final_model_merged')

# 提取三元组
triplets = model.extract_triplets("糖尿病有什么症状？", relation_types=["疾病-症状"])
```

#### 方式2: 直接加载（高级）

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel

# 加载合并后的模型（推荐）
tokenizer = T5Tokenizer.from_pretrained('saved_model/final_model_merged')
model = T5ForConditionalGeneration.from_pretrained('saved_model/final_model_merged')

# 或加载 LoRA 权重（需要基础模型）
base_model = T5ForConditionalGeneration.from_pretrained('clueAI')
model = PeftModel.from_pretrained(base_model, 'saved_model/final_model')
```

### 继续训练

如果需要从检查点继续训练，修改训练脚本：

```python
# 在 train_model() 函数中，创建 Trainer 后添加：
trainer.train(resume_from_checkpoint='saved_model/checkpoint-2000')
```

---

## 常见问题

### 1. 显存不足（OOM）

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
- 减小 `BATCH_SIZE`（如从 8 降到 4）
- 增加 `GRADIENT_ACCUMULATION_STEPS`（保持有效批次大小）
- 减小 `MAX_LENGTH` 或 `MAX_TARGET_LENGTH`
- 启用 LoRA 微调（`USE_LORA=True`）
- 使用 CPU 训练（速度较慢）

### 2. 训练损失不下降

**可能原因**:
- 学习率过高或过低
- 数据格式不正确
- 模型未正确加载

**解决方案**:
- 检查学习率设置（LoRA 模式建议 1e-3 到 5e-3）
- 检查数据格式是否正确
- 查看训练日志，确认模型已加载
- 尝试降低学习率

### 3. 训练速度慢

**优化建议**:
- 使用 GPU 训练（CUDA）
- 启用 FP16 混合精度（自动启用）
- 增加 `dataloader_num_workers`（GPU 模式）
- 使用 LoRA 微调（减少参数量）

### 4. 模型文件太大

**解决方案**:
- 使用 LoRA 微调（只保存 LoRA 权重，MB 级）
- 删除不需要的检查点
- 使用 `save_total_limit=3` 限制检查点数量

### 5. CUDA 版本不匹配

**症状**: `CUDA runtime version mismatch`

**解决方案**:
- 检查 PyTorch 和 CUDA 版本是否匹配
- 重新安装匹配的 PyTorch 版本
- 或使用 CPU 训练

### 6. Transformers 版本检查错误

**症状**: PyTorch 版本检查相关错误

**解决方案**:
- 训练脚本已包含版本检查补丁
- 如果仍有问题，更新 transformers 版本：
  ```bash
  pip install transformers>=4.30.0
  ```

### 7. 数据加载错误

**症状**: `FileNotFoundError` 或数据格式错误

**解决方案**:
- 检查数据文件路径是否正确
- 检查数据格式是否符合要求
- 运行数据生成脚本重新生成数据

---

## 性能优化建议

### 1. 硬件优化

- **GPU**: 建议使用 NVIDIA GPU（8GB+ 显存）
- **内存**: 建议 16GB+ 系统内存
- **存储**: 使用 SSD 存储数据，加快数据加载

### 2. 训练优化

- **批次大小**: 根据显存调整，尽量使用较大批次
- **梯度累积**: 显存不足时使用梯度累积
- **混合精度**: 自动启用 FP16（CUDA GPU）
- **数据加载**: GPU 模式使用多进程加载（`dataloader_num_workers=4`）

### 3. 模型优化

- **LoRA**: 使用 LoRA 微调减少显存和训练时间
- **长度限制**: 根据实际需求设置合理的最大长度
- **早停**: 监控验证损失，避免过拟合

### 4. 数据优化

- **数据清洗**: 确保数据格式正确
- **数据平衡**: 尽量保证各类关系样本平衡
- **数据增强**: 可以尝试数据增强技术

### 5. 超参数调优

建议的超参数搜索顺序：

1. **学习率**: 最重要，优先调整
2. **批次大小**: 根据显存调整
3. **LoRA_R**: 影响模型容量
4. **训练轮数**: 根据数据集大小调整
5. **Warmup 步数**: 通常使用默认值即可

---

## 训练示例

### 示例1: 基础训练（LoRA）

```bash
# 1. 确保数据已准备
python scripts/generate_data.py

# 2. 开始训练
python scripts/train.py
```

### 示例2: 全量微调

修改 `config.py`:

```python
USE_LORA = False
BATCH_SIZE = 4  # 全量微调需要更小的批次
LEARNING_RATE = 2e-4
```

然后运行：

```bash
python scripts/train.py
```

### 示例3: 自定义配置训练

创建自定义配置文件或直接修改 `config.py`:

```python
# 大显存 GPU 配置
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
MAX_LENGTH = 512
MAX_TARGET_LENGTH = 256

# LoRA 配置
USE_LORA = True
LORA_R = 16  # 更大的 rank
LORA_ALPHA = 32
```

---

## 附录

### 支持的关系类型

1. 疾病-症状
2. 疾病-药品
3. 疾病-食物
4. 疾病-并发症
5. 疾病-忌口食物
6. 疾病-宜吃食物
7. 疾病-检查项目
8. 疾病-病因
9. 疾病-预防措施
10. 疾病-治疗方式
11. 症状-疾病
12. 药品-疾病
13. 检查项目-疾病

### 参考资源

- [PromptCLUE GitHub](https://github.com/CLUEbenchmark/PromptCLUE)
- [PEFT 文档](https://huggingface.co/docs/peft)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)

### 联系与支持

如有问题，请检查：
1. 训练日志 (`logs/`)
2. 错误信息
3. 配置文件设置
4. 数据格式

---

**最后更新**: 2024年

**文档版本**: 1.0

