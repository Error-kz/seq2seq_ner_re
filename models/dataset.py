"""
Seq2Seq数据集类
"""
import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import T5Tokenizer

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


class Seq2SeqDataset(Dataset):
    """Seq2Seq训练数据集"""
    
    def __init__(self, data_path, tokenizer, max_length=512, max_target_length=256):
        """
        Args:
            data_path: 训练数据文件路径
            tokenizer: T5 tokenizer
            max_length: 最大输入长度
            max_target_length: 最大输出长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_target_length = max_target_length
        
        # 读取数据
        print(f"正在读取数据文件: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.samples = [line.strip() for line in f if line.strip()]
        
        print(f"共加载 {len(self.samples)} 条样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        line = self.samples[idx]
        
        # 分割输入和输出
        if ' <SEP> ' in line:
            input_text, output_text = line.split(' <SEP> ', 1)
        else:
            # 如果没有分隔符，尝试其他方式
            parts = line.split(' <SEP> ')
            if len(parts) >= 2:
                input_text = parts[0]
                output_text = ' <SEP> '.join(parts[1:])
            else:
                input_text = line
                output_text = ""
        
        # Tokenize输入（返回列表而不是tensor，避免DataCollator警告）
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None  # 返回列表而不是tensor
        )
        
        # Tokenize输出（作为labels）
        target_encoding = self.tokenizer(
            output_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors=None  # 返回列表而不是tensor
        )
        
        # 将padding token的label设为-100（忽略计算loss）
        labels = np.array(target_encoding['input_ids'], dtype=np.int64)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        labels[labels == pad_token_id] = -100
        
        # 确保labels不是全部为-100（至少有一些有效标签）
        if np.all(labels == -100):
            # 如果全部是padding，至少保留第一个token
            labels[0] = target_encoding['input_ids'][0]
        
        return {
            'input_ids': np.array(input_encoding['input_ids'], dtype=np.int64),
            'attention_mask': np.array(input_encoding['attention_mask'], dtype=np.int64),
            'labels': labels
        }


class Seq2SeqInferenceDataset(Dataset):
    """Seq2Seq推理数据集"""
    
    def __init__(self, texts, tokenizer, max_length=512):
        """
        Args:
            texts: 输入文本列表
            tokenizer: T5 tokenizer
            max_length: 最大输入长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize输入
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None  # 返回列表而不是tensor
        )
        
        return {
            'input_ids': np.array(encoding['input_ids'], dtype=np.int64),
            'attention_mask': np.array(encoding['attention_mask'], dtype=np.int64),
        }

