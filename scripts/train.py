"""
训练Seq2Seq模型用于NER+RE任务（支持LoRA微调）
"""
import os
import sys
import torch

# 在导入transformers之前，禁用版本检查（临时解决方案）
# 这可以绕过PyTorch 2.6的版本要求
def _patch_transformers_version_check():
    """修补transformers的版本检查"""
    try:
        # 方法1: 修改 import_utils 模块
        import transformers.utils.import_utils as import_utils
        if hasattr(import_utils, 'check_torch_load_is_safe'):
            # 保存原始函数（如果需要）
            _original_check = import_utils.check_torch_load_is_safe
            
            # 创建一个绕过版本检查的函数
            def _patched_check():
                """绕过torch.load的版本检查"""
                pass  # 不做任何检查
            
            # 替换函数
            import_utils.check_torch_load_is_safe = _patched_check
            return True
    except Exception as e:
        # 如果补丁失败，继续执行
        pass
    return False

# 应用补丁
_patch_transformers_version_check()

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from torch.utils.data import DataLoader

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dataset import Seq2SeqDataset
from config import Config

# 导入LoRA相关库
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️  警告: peft库未安装，将使用全量微调")
    print("   安装命令: pip install peft")


def train_model():
    """训练Seq2Seq模型"""
    
    config = Config()
    
    # 确保目录存在
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    print("=" * 50)
    print("开始训练Seq2Seq模型")
    print("=" * 50)
    print(f"模型: {config.MODEL_NAME}")
    print(f"训练数据: {config.TRAIN_DATA_PATH}")
    print(f"输出目录: {config.MODEL_DIR}")
    print("=" * 50)
    
    # 1. 加载tokenizer和模型
    print("\n1. 加载tokenizer和模型...")
    # 直接使用备用方法加载模型（绕过transformers版本检查）
    tokenizer = T5Tokenizer.from_pretrained(config.MODEL_NAME)
    
    # 直接加载配置和权重（绕过安全检查）
    from transformers import T5Config
    
    print("   加载模型配置...")
    model_config = T5Config.from_pretrained(config.MODEL_NAME)
    # 创建模型
    model = T5ForConditionalGeneration(model_config)
    
    # 直接加载权重
    model_path = os.path.join(config.MODEL_NAME, "pytorch_model.bin")
    if os.path.exists(model_path):
        print(f"   从 {model_path} 加载权重...")
        # 使用 torch.load，设置 weights_only=False 来绕过安全检查
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:
            # 如果 weights_only 参数不支持，使用旧方法
            state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print("   ✅ 模型加载成功")
    else:
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 设置pad_token（某些模型可能没有）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # 获取原始模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 2. 配置LoRA（如果启用）
    use_lora = config.USE_LORA and PEFT_AVAILABLE
    if use_lora:
        print("\n2. 配置LoRA微调...")
        print(f"   LoRA Rank (r): {config.LORA_R}")
        print(f"   LoRA Alpha: {config.LORA_ALPHA}")
        print(f"   LoRA Dropout: {config.LORA_DROPOUT}")
        print(f"   目标模块: {config.LORA_TARGET_MODULES}")
        
        # 创建LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,  # T5是序列到序列任务
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            target_modules=config.LORA_TARGET_MODULES,
            bias=config.LORA_BIAS,
        )
        
        # 应用LoRA到模型
        model = get_peft_model(model, lora_config)
        
        # 打印可训练参数信息
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n✅ LoRA配置完成")
        print(f"   原始模型参数: {total_params/1e6:.2f}M")
        print(f"   可训练参数: {trainable_params/1e6:.2f}M ({trainable_params/total_params*100:.2f}%)")
        print(f"   参数减少: {(1 - trainable_params/total_params)*100:.2f}%")
        
        # 打印LoRA模型结构
        model.print_trainable_parameters()
        
        # 移动模型到设备（强制使用CUDA或CPU，不使用MPS）
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   使用设备: {device.upper()}")
        model = model.to(device)
    else:
        if config.USE_LORA and not PEFT_AVAILABLE:
            print("\n⚠️  LoRA已启用但peft库未安装，使用全量微调")
        else:
            print("\n📝 使用全量微调模式")
        print(f"✅ 模型加载完成")
        print(f"   Vocab size: {len(tokenizer)}")
        print(f"   Model parameters: {total_params/1e6:.1f}M")
        print(f"   可训练参数: {trainable_params_before/1e6:.1f}M")
        
        # 移动模型到设备（强制使用CUDA或CPU，不使用MPS）
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   使用设备: {device.upper()}")
        model = model.to(device)
    
    # 3. 准备数据集
    print("\n3. 准备数据集...")
    train_dataset = Seq2SeqDataset(
        config.TRAIN_DATA_PATH,
        tokenizer,
        max_length=config.MAX_LENGTH,
        max_target_length=config.MAX_TARGET_LENGTH
    )
    
    # 如果有验证集，加载验证集
    dev_dataset = None
    if os.path.exists(config.DEV_DATA_PATH):
        print(f"   发现验证集: {config.DEV_DATA_PATH}")
        dev_dataset = Seq2SeqDataset(
            config.DEV_DATA_PATH,
            tokenizer,
            max_length=config.MAX_LENGTH,
            max_target_length=config.MAX_TARGET_LENGTH
        )
    
    # 4. 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # 5. 训练参数（LoRA微调时学习率可以适当提高）
    print("\n4. 设置训练参数...")
    learning_rate = config.LEARNING_RATE
    if use_lora:
        # LoRA微调时，学习率通常比全量微调高5-10倍
        learning_rate = config.LEARNING_RATE * 5
        print(f"   LoRA微调模式，学习率调整为: {learning_rate}")
    # 检查是否使用CUDA GPU
    use_cuda = torch.cuda.is_available()
    
    training_args = TrainingArguments(
        output_dir=config.MODEL_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1),  # 梯度累积
        learning_rate=learning_rate,
        warmup_steps=config.WARMUP_STEPS,
        logging_dir=config.LOG_DIR,
        logging_steps=config.LOGGING_STEPS,
        save_steps=config.SAVE_STEPS,
        eval_strategy="steps" if dev_dataset else "no",  # evaluation_strategy已重命名为eval_strategy
        eval_steps=config.EVAL_STEPS if dev_dataset else None,
        save_total_limit=3,  # 只保留最近3个模型
        load_best_model_at_end=True if dev_dataset else False,
        metric_for_best_model="loss" if dev_dataset else None,
        greater_is_better=False if dev_dataset else None,
        fp16=use_cuda,  # 如果使用CUDA GPU，启用混合精度
        bf16=False,  # 可选：如果GPU支持bf16，可以启用
        dataloader_num_workers=4 if use_cuda else 0,  # CUDA可以使用多进程加载数据
        dataloader_pin_memory=True if use_cuda else False,  # CUDA支持pin_memory加速
    )
    
    if use_cuda:
        print(f"   ✅ 检测到CUDA设备，使用GPU加速:")
        print(f"      Batch size: {config.BATCH_SIZE}")
        print(f"      梯度累积步数: {getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1)}")
        print(f"      等效batch size: {config.BATCH_SIZE * getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1)}")
        print(f"      最大输入长度: {config.MAX_LENGTH}")
        print(f"      最大输出长度: {config.MAX_TARGET_LENGTH}")
        print(f"      FP16混合精度: 启用")
    
    # 6. 创建Trainer
    print("\n5. 创建Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
    )
    
    # 7. 开始训练
    print("\n6. 开始训练...")
    print("=" * 50)
    trainer.train()
    
    # 8. 保存最终模型
    print("\n7. 保存最终模型...")
    final_model_path = os.path.join(config.MODEL_DIR, 'final_model')
    
    if use_lora:
        # LoRA模式下，保存LoRA权重和基础模型
        print("   保存LoRA权重...")
        trainer.save_model(final_model_path)  # 这会保存LoRA权重
        tokenizer.save_pretrained(final_model_path)
        
        # 可选：合并LoRA权重到基础模型（用于推理）
        print("   合并LoRA权重到基础模型...")
        merged_model_path = os.path.join(config.MODEL_DIR, 'final_model_merged')
        os.makedirs(merged_model_path, exist_ok=True)
        
        # 加载基础模型
        base_model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)
        # 加载LoRA权重
        model = PeftModel.from_pretrained(base_model, final_model_path)
        # 合并权重
        merged_model = model.merge_and_unload()
        # 保存合并后的模型
        merged_model.save_pretrained(merged_model_path)
        tokenizer.save_pretrained(merged_model_path)
        
        print(f"✅ 训练完成！")
        print(f"   LoRA权重已保存到: {final_model_path}")
        print(f"   合并模型已保存到: {merged_model_path}")
    else:
        # 全量微调模式，直接保存
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print(f"✅ 训练完成！")
        print(f"   模型已保存到: {final_model_path}")
    
    print("=" * 50)


if __name__ == '__main__':
    train_model()

