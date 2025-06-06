import json
import logging
import psutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Union
import torch

def setup_logger(name: str, log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """Setup logger with consistent formatting"""
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        ensure_dir(Path(log_file).parent)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def format_time(seconds: float) -> str:
    """Format seconds to human readable time string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def get_memory_usage() -> Dict[str, str]:
    """Get current memory usage information"""
    memory = psutil.virtual_memory()
    return {
        "total": f"{memory.total / (1024**3):.1f} GB",
        "used": f"{memory.used / (1024**3):.1f} GB", 
        "available": f"{memory.available / (1024**3):.1f} GB",
        "percent": f"{memory.percent:.1f}%"
    }

def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file"""
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

def load_json(file_path: Union[str, Path]) -> Any:
    """Load data from JSON file"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_model_size(model_path: Union[str, Path]) -> Dict[str, Union[str, int]]:
    """Calculate model file size and parameter count"""
    model_path = Path(model_path)
    
    if not model_path.exists():
        return {"error": "Model path not found"}
    
    # Calculate total size
    total_size = 0
    file_count = 0
    
    if model_path.is_file():
        total_size = model_path.stat().st_size
        file_count = 1
    else:
        for file_path in model_path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
    
    # Format size
    if total_size < 1024**2:
        size_str = f"{total_size / 1024:.1f} KB"
    elif total_size < 1024**3:
        size_str = f"{total_size / (1024**2):.1f} MB"
    else:
        size_str = f"{total_size / (1024**3):.1f} GB"
    
    return {
        "total_size_bytes": total_size,
        "total_size": size_str,
        "file_count": file_count
    }

def get_gpu_info() -> Dict[str, Union[str, bool]]:
    """Get GPU availability and information"""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "current_device": "cpu"
    }
    
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["current_device"] = torch.cuda.get_device_name(0)
        
        # Get memory info for current device
        if torch.cuda.device_count() > 0:
            memory_allocated = torch.cuda.memory_allocated(0)
            memory_cached = torch.cuda.memory_reserved(0)
            
            info["memory_allocated"] = f"{memory_allocated / (1024**3):.1f} GB"
            info["memory_cached"] = f"{memory_cached / (1024**3):.1f} GB"
    
    return info

def format_training_time(start_time: datetime, current_time: Optional[datetime] = None) -> str:
    """Format training elapsed time"""
    if current_time is None:
        current_time = datetime.now()
    
    elapsed = current_time - start_time
    return format_time(elapsed.total_seconds())

def create_checkpoint_name(epoch: int, step: int, loss: float) -> str:
    """Create standardized checkpoint filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"checkpoint-epoch-{epoch:03d}-step-{step:06d}-loss-{loss:.4f}-{timestamp}"

def clean_text_for_training(text: str) -> str:
    """Clean text for training (basic preprocessing)"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove control characters but keep basic punctuation
    import re
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
    
    return text.strip()

def is_valid_qa_pair(question: str, answer: str, min_q_len: int = 5, min_a_len: int = 10) -> bool:
    """Validate if a QA pair meets basic quality criteria"""
    if not question or not answer:
        return False
    
    question = question.strip()
    answer = answer.strip()
    
    # Length checks
    if len(question) < min_q_len or len(answer) < min_a_len:
        return False
    
    # Avoid empty or meaningless content
    if question.lower() in ['', 'none', 'null', 'n/a', 'unknown']:
        return False
    
    if answer.lower() in ['', 'none', 'null', 'n/a', 'unknown']:
        return False
    
    return True