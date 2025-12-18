"""
Seq2Seq数据集类
"""
import os
import sys
import torch
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
        
        # Tokenize输入
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize输出（作为labels）
        target_encoding = self.tokenizer(
            output_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 将padding token的label设为-100（忽略计算loss）
        labels = target_encoding['input_ids'].squeeze().clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
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
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }

