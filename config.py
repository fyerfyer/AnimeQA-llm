import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LoRAConfig(BaseModel):
    """LoRA fine-tuning configuration"""
    rank: int = Field(default=16)
    alpha: int = Field(default=32)
    dropout: float = Field(default=0.1)

class HuggingFaceConfig(BaseModel):
    """HuggingFace related configuration"""
    endpoint: str = Field(default="https://hf-mirror.com")
    token: Optional[str] = Field(default=None)

class DatabaseConfig(BaseModel):
    """Database configuration"""
    path: str = Field(default="./data/anime_qa.db")

class ModelConfig(BaseModel):
    """Model configuration"""
    base_model_name: str = Field(default="microsoft/DialoGPT-medium")
    save_path: str = Field(default="./models/anime-qa-model")
    cache_dir: str = Field(default="./cache/models")

class DataConfig(BaseModel):
    """Data configuration"""
    dataset_name: str = Field(default="local")  # 修改默认值
    dataset_path: str = Field(default="./dataset")  # 新增字段
    dataset_file: str = Field(default="metadata.jsonl")  # 新增字段
    cache_dir: str = Field(default="./cache/datasets")
    processed_path: str = Field(default="./data/processed")

class APIConfig(BaseModel):
    """API configuration"""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    enable_cors: bool = Field(default=True)
    key: Optional[str] = Field(default=None)

class TrainingConfig(BaseModel):
    """Training configuration"""
    batch_size: int = Field(default=4)
    learning_rate: float = Field(default=5e-5)
    num_epochs: int = Field(default=3)
    max_length: int = Field(default=512)
    lora_config: LoRAConfig = Field(default_factory=LoRAConfig)

class InferenceConfig(BaseModel):
    """Inference configuration"""
    max_length: int = Field(default=128)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.9)
    do_sample: bool = Field(default=True)

class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = Field(default="INFO")
    file: str = Field(default="./logs/anime_qa.log")
    format: str = Field(default="[{time:YYYY-MM-DD HH:mm:ss}] {level} | {message}")

class CacheConfig(BaseModel):
    """Cache configuration"""
    enable: bool = Field(default=True)
    max_size: int = Field(default=1000)
    ttl: int = Field(default=3600)

class Config(BaseModel):
    """Main configuration class"""
    huggingface: HuggingFaceConfig = Field(default_factory=HuggingFaceConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

def load_config(config_file: str = "config.yaml") -> Config:
    """Load configuration from specified config file and environment variables"""
    config_data = {}
    
    # 1. Read config file
    config_path = Path(config_file)
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}
        print(f"Loaded configuration from: {config_file}")
    else:
        print(f"Warning: Config file {config_file} not found, using defaults")
    
    # 2. Override sensitive configurations with environment variables
    if os.getenv('HF_TOKEN'):
        config_data.setdefault('huggingface', {})['token'] = os.getenv('HF_TOKEN')
    
    if os.getenv('API_KEY'):
        config_data.setdefault('api', {})['key'] = os.getenv('API_KEY')
    
    if os.getenv('LOG_LEVEL'):
        config_data.setdefault('logging', {})['level'] = os.getenv('LOG_LEVEL')
    
    return Config(**config_data)

# 修复配置加载逻辑
def get_settings():
    """Get configuration with proper test config handling"""
    test_config_file = "config.test.yaml"
    use_test_config = os.getenv('USE_TEST_CONFIG', '').lower() == 'true'
    
    if use_test_config and Path(test_config_file).exists():
        print(f"Using test configuration: {test_config_file}")
        return load_config(test_config_file)
    else:
        return load_config()

# Global configuration instance
settings = get_settings()

# Convenience access functions
def get_hf_endpoint() -> str:
    """Get HuggingFace mirror endpoint"""
    return settings.huggingface.endpoint

def get_database_path() -> str:
    """Get database path"""
    return settings.database.path

def is_production() -> bool:
    """Check if running in production environment"""
    return os.getenv('ENVIRONMENT', 'development') == 'production'

def ensure_directories():
    """Ensure necessary directories exist"""
    directories = [
        Path(settings.model.save_path).parent,
        Path(settings.model.cache_dir),
        Path(settings.data.cache_dir),
        Path(settings.data.processed_path),
        Path(settings.database.path).parent,
        Path(settings.logging.file).parent
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Test configuration loading
    print("Configuration loading test:")
    print(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"USE_TEST_CONFIG: {os.getenv('USE_TEST_CONFIG')}")
    print(f"HF Endpoint: {settings.huggingface.endpoint}")
    print(f"Database Path: {settings.database.path}")
    print(f"Dataset Name: {settings.data.dataset_name}")
    print(f"Dataset Path: {settings.data.dataset_path}")
    print(f"Base Model: {settings.model.base_model_name}")
    print(f"API Port: {settings.api.port}")
    print(f"LoRA Rank: {settings.training.lora_config.rank}")
    
    # Create necessary directories
    ensure_directories()
    print("Directory creation completed")