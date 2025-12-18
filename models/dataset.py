"""
最终稳定版 Seq2Seq Dataset（NER + RE 专用）

设计目标：
- 绝不出现空 label
- 不在 Dataset 阶段 padding target
- token 数稳定，FP16 + LoRA 不炸
- 适配 Trainer + DataCollatorForSeq2Seq
"""

import os
import sys
import numpy as np
from torch.utils.data import Dataset


class Seq2SeqDataset(Dataset):
    """稳定版 Seq2Seq 训练数据集"""

    def __init__(self, data_path, tokenizer, max_length=512, max_target_length=256):
        """
        Args:
            data_path: 数据文件路径，每行格式：input <SEP> output
            tokenizer: HuggingFace tokenizer (T5Tokenizer)
            max_length: 输入最大长度
            max_target_length: target 最大长度（仅用于 truncation，不 padding）
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_target_length = max_target_length

        print(f"📥 正在读取数据文件: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.samples = [line.strip() for line in f if line.strip()]

        print(f"✅ 共加载 {len(self.samples)} 条样本")

    def __len__(self):
        return len(self.samples)

    # ============================
    # 核心：稳定 target 构造逻辑
    # ============================
    def _normalize_output(self, output_text: str) -> str:
        """
        保证 target：
        1. 永远非空
        2. 结构稳定（匹配实际数据格式：三元组列表）
        3. token 数不少于安全线
        """
        output_text = output_text.strip()

        # 情况 1：完全空或只有空白
        if output_text == "" or not output_text:
            # 返回一个有效的空三元组格式（保持格式一致性）
            output_text = "(NONE, NONE, NONE)"

        # 情况 2：只有 NONE（大小写不敏感）
        elif output_text.upper().strip() == "NONE":
            output_text = "(NONE, NONE, NONE)"

        # 情况 3：已有三元组格式（默认信任上游）
        # 实际格式示例: (实体1, 关系, 实体2); (实体3, 关系, 实体4)
        # 这里可以添加更严格的格式校验，但为了稳定性，先信任上游数据

        # ============================
        # token 数安全保护（非常重要）
        # 确保有足够的 token 避免 FP16 精度问题
        # ============================
        token_len = len(self.tokenizer.tokenize(output_text))
        if token_len < 5:
            # 工程兜底：如果 token 数太少，补充一些 padding
            # 注意：这里不添加 <PAD> token，因为会被 mask 掉
            # 而是添加一些不影响语义的占位符
            output_text = output_text + " . . ."
        
        return output_text

    def __getitem__(self, idx):
        line = self.samples[idx]

        # ----------------------------
        # 1. 拆分 input / output
        # ----------------------------
        if ' <SEP> ' in line:
            input_text, output_text = line.split(' <SEP> ', 1)
        else:
            input_text = line
            output_text = ""

        # ----------------------------
        # 2. 标准化 target（关键）
        # ----------------------------
        output_text = self._normalize_output(output_text)

        # ----------------------------
        # 3. Tokenize input（padding 到 max_length）
        # ----------------------------
        input_enc = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors=None,
        )

        # ----------------------------
        # 4. Tokenize target（不 padding）
        # ----------------------------
        target_enc = self.tokenizer(
            output_text,
            max_length=self.max_target_length,
            truncation=True,
            padding=False,  # 关键：不在 Dataset 阶段 padding
            return_tensors=None,
        )

        labels = np.array(target_enc['input_ids'], dtype=np.int64)

        # 将 pad token mask 为 -100（忽略 loss）
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        labels[labels == pad_token_id] = -100

        return {
            'input_ids': np.array(input_enc['input_ids'], dtype=np.int64),
            'attention_mask': np.array(input_enc['attention_mask'], dtype=np.int64),
            'labels': labels,
        }


# ============================
# 推理数据集（可选）
# ============================
class Seq2SeqInferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors=None,
        )
        return {
            'input_ids': np.array(enc['input_ids'], dtype=np.int64),
            'attention_mask': np.array(enc['attention_mask'], dtype=np.int64),
        }