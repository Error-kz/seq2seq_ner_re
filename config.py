"""
Seq2Seq NER+RE 模型配置文件
"""
import os
import torch

class Config:
    """配置类"""
    
    # 数据路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(BASE_DIR)
    
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODEL_DIR = os.path.join(BASE_DIR, 'saved_model')
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    
    # 训练数据路径
    TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_seq2seq.txt')
    DEV_DATA_PATH = os.path.join(DATA_DIR, 'test_seq2seq.txt')
    
    # 模型配置
    # 推荐模型（按优先级）:
    # 1. "ClueAI/PromptCLUE-base" - 中文T5模型，推荐
    MODEL_NAME = os.path.join(BASE_DIR, 'clueAI')
    
    # 训练参数
    MAX_LENGTH = 512  # 最大输入长度
    MAX_TARGET_LENGTH = 256  # 最大输出长度
    BATCH_SIZE = 8  # 批次大小
    LEARNING_RATE = 2e-4  # 学习率
    NUM_EPOCHS = 4  # 训练轮数
    WARMUP_STEPS = 500  # 预热步数
    SAVE_STEPS = 1000  # 保存步数
    LOGGING_STEPS = 100  # 日志步数
    EVAL_STEPS = 500  # 评估步数
    
    # LoRA微调参数
    USE_LORA = True  # 是否使用LoRA微调
    LORA_R = 8  # LoRA rank，控制参数量，越大效果越好但参数越多
    LORA_ALPHA = 16  # LoRA alpha，通常设为r的1-2倍
    LORA_DROPOUT = 0.05  # LoRA dropout率
    LORA_TARGET_MODULES = ["q", "k", "v", "o"]  # 目标模块：Attention的Q、K、V、O矩阵
    LORA_BIAS = "none"  # 是否训练bias："none"不训练，"all"训练所有，"lora_only"只训练LoRA的bias
    GRADIENT_ACCUMULATION_STEPS = 1  # 梯度累积步数

    # 推理参数
    NUM_BEAMS = 4  # Beam search数量
    EARLY_STOPPING = True  # 早停
    DO_SAMPLE = False  # 是否采样
    
    # 设备配置
    CUDA_VISIBLE_DEVICES = "4,5,6,7"  # 指定使用的 GPU 列表
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 关系类型定义
    RELATION_TYPES = [
        '疾病-症状',
        '疾病-药品',
        '疾病-食物',
        '疾病-并发症',
        '疾病-忌口食物',
        '疾病-宜吃食物',
        '疾病-检查项目',
        '疾病-病因',
        '疾病-预防措施',
        '疾病-治疗方式',
        '症状-疾病',
        '药品-疾病',
        '检查项目-疾病',
    ]
    
    # 问题类型到关系类型的映射
    QUESTION_TYPE_TO_RELATION = {
        'disease_symptom': ['疾病-症状'],
        'symptom_disease': ['症状-疾病'],
        'disease_drug': ['疾病-药品'],
        'drug_disease': ['药品-疾病'],
        'disease_food': ['疾病-食物'],
        'disease_not_food': ['疾病-忌口食物'],
        'disease_do_food': ['疾病-宜吃食物'],
        'food_not_disease': ['疾病-忌口食物'],
        'food_do_disease': ['疾病-宜吃食物'],
        'disease_check': ['疾病-检查项目'],
        'check_disease': ['检查项目-疾病'],
        'disease_cause': ['疾病-病因'],
        'disease_prevent': ['疾病-预防措施'],
        'disease_cureway': ['疾病-治疗方式'],
    }

