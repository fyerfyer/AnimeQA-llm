import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch

@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration"""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    # 修改为DialoGPT适用的目标模块
    target_modules: List[str] = field(default_factory=lambda: ["c_attn", "c_proj"])
    bias: str = "none"  # Options: "none", "all", "lora_only"
    task_type: str = "CAUSAL_LM"

@dataclass
class HuggingFaceConfig:
    """HuggingFace configuration including mirror settings"""
    endpoint: str = field(default_factory=lambda: os.getenv("HF_ENDPOINT", "https://hf-mirror.com"))
    token: Optional[str] = field(default_factory=lambda: os.getenv("HF_TOKEN"))
    cache_dir: str = "./cache/huggingface"
    
    def __post_init__(self):
        """Set environment variables for HuggingFace"""
        if self.endpoint != "https://huggingface.co":
            os.environ['HF_ENDPOINT'] = self.endpoint
        
        if self.token:
            os.environ['HF_TOKEN'] = self.token

@dataclass
class TrainingConfig:
    """Main training configuration"""
    # Model settings
    base_model_name: str = "microsoft/DialoGPT-medium"
    model_save_path: str = "./models/anime-qa-model"
    
    # Training hyperparameters
    learning_rate: float = 5e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_length: int = 512
    warmup_steps: int = 100
    
    # Optimization settings
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # LoRA configuration
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    
    # HuggingFace configuration  
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    
    # Data settings
    train_data_path: str = "./data/processed/train_dataset"
    val_data_path: str = "./data/processed/val_dataset"
    validation_split: float = 0.1
    
    # Training monitoring
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Hardware settings
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    fp16: bool = False  # 在CPU上训练时关闭FP16
    dataloader_num_workers: int = 0  # Set to 0 to avoid multiprocessing issues
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Checkpointing
    resume_from_checkpoint: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Ensure paths exist
        Path(self.model_save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.huggingface.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # CPU优化：调整批次大小和禁用FP16
        if self.device == "cpu":
            self.fp16 = False
            if self.batch_size > 2:
                self.batch_size = 1  # CPU训练时使用更小的批次
                print(f"Adjusted batch size to {self.batch_size} for CPU training")
        
        # Validate batch size for available memory
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            # Rough estimation: reduce batch size if GPU memory < 8GB
            if gpu_memory < 8 * 1024**3 and self.batch_size > 2:
                self.batch_size = 2
                print(f"Reduced batch size to {self.batch_size} due to limited GPU memory")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for saving"""
        config_dict = {}
        
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                # Handle nested dataclass objects
                config_dict[key] = {k: v for k, v in value.__dict__.items()}
            else:
                config_dict[key] = value
        
        return config_dict
    
    def get_training_args(self) -> Dict[str, Any]:
        """Get arguments compatible with transformers.TrainingArguments"""
        return {
            "output_dir": self.model_save_path,
            "learning_rate": self.learning_rate,
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_train_epochs": self.num_epochs,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "fp16": self.fp16,
            "dataloader_num_workers": self.dataloader_num_workers,
            "remove_unused_columns": False,
            "report_to": [],  # Disable wandb/tensorboard for now
        }
    
    def get_lora_config(self) -> Dict[str, Any]:
        """Get LoRA configuration for PEFT"""
        return {
            "r": self.lora.rank,
            "lora_alpha": self.lora.alpha,
            "lora_dropout": self.lora.dropout,
            "target_modules": self.lora.target_modules,
            "bias": self.lora.bias,
            "task_type": self.lora.task_type,
        }

def load_training_config(config_file: Optional[str] = None) -> TrainingConfig:
    """Load training configuration from file or use defaults"""
    if config_file and Path(config_file).exists():
        # Could implement YAML/JSON loading here if needed
        pass
    
    return TrainingConfig()

# Global training configuration instance
training_config = load_training_config()

# Convenience functions
def get_device() -> str:
    """Get the device for training"""
    return training_config.device

def get_hf_endpoint() -> str:
    """Get HuggingFace endpoint URL"""
    return training_config.huggingface.endpoint

def is_fp16_enabled() -> bool:
    """Check if mixed precision training is enabled"""
    return training_config.fp16

if __name__ == "__main__":
    # Test configuration
    config = TrainingConfig()
    print("Training Configuration:")
    print(f"Base Model: {config.base_model_name}")
    print(f"Device: {config.device}")
    print(f"HF Endpoint: {config.huggingface.endpoint}")
    print(f"LoRA Rank: {config.lora.rank}")
    print(f"LoRA Target Modules: {config.lora.target_modules}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"FP16 Enabled: {config.fp16}")