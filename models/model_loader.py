import os
import sys
import logging
import torch
from pathlib import Path
from typing import Tuple, Dict, Any, List
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

# Import training config and utils
sys.path.append(str(Path(__file__).parent.parent))
from training.config import TrainingConfig, training_config
from utils.helpers import setup_logger, get_gpu_info, calculate_model_size

# Setup logging
logger = setup_logger(__name__)

class ModelLoader:
    """Model loader for base models and fine-tuned models"""
    
    def __init__(self, config: TrainingConfig = None):
        """Initialize model loader"""
        self.config = config or training_config
        self.model = None
        self.tokenizer = None
        self.device = self.config.device
        
        # Set HuggingFace cache directory
        os.environ['HF_HOME'] = self.config.huggingface.cache_dir
        os.environ['TRANSFORMERS_CACHE'] = self.config.huggingface.cache_dir
        
        logger.info(f"ModelLoader initialized with device: {self.device}")
        
        # Log GPU info if available
        gpu_info = get_gpu_info()
        if gpu_info["cuda_available"]:
            logger.info(f"GPU: {gpu_info['current_device']}")
            if "memory_allocated" in gpu_info:
                logger.info(f"GPU Memory: {gpu_info['memory_allocated']}")
    
    def load_tokenizer(self, model_name: str = None) -> PreTrainedTokenizer:
        """Load tokenizer for the model"""
        try:
            model_name = model_name or self.config.base_model_name
            logger.info(f"Loading tokenizer: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.config.huggingface.cache_dir,
                trust_remote_code=True
            )
            
            # Add special tokens if needed
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            
            self.tokenizer = tokenizer
            logger.info("Tokenizer loaded successfully")
            return tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def load_base_model(self, model_name: str = None) -> PreTrainedModel:
        """Load base model for training or inference"""
        try:
            model_name = model_name or self.config.base_model_name
            logger.info(f"Loading base model: {model_name}")
            
            # Model loading arguments
            model_kwargs = {
                "cache_dir": self.config.huggingface.cache_dir,
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.config.fp16 else torch.float32,
            }
            
            # 只在GPU可用时使用device_map
            if torch.cuda.is_available() and self.device != "cpu":
                model_kwargs["device_map"] = "auto"
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if not model_kwargs.get("device_map"):
                model = model.to(self.device)
                logger.info(f"Model moved to device: {self.device}")
            
            self.model = model
            logger.info("Base model loaded successfully")
            
            # Log model size
            try:
                cache_path = Path(self.config.huggingface.cache_dir) / model_name.replace("/", "--")
                model_info = calculate_model_size(cache_path)
                if "total_size" in model_info:
                    logger.info(f"Model size: {model_info['total_size']}")
            except Exception as e:
                logger.debug(f"Could not calculate model size: {e}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise
    
    def find_target_modules(self, model: PreTrainedModel) -> List[str]:
        """Automatically find suitable target modules for LoRA"""
        linear_layers = []
        
        for name, module in model.named_modules():
            # 检查Conv1D和Linear层
            if hasattr(module, 'weight') and len(module.weight.shape) == 2:
                # Skip embedding and final classification layers
                if not any(skip in name.lower() for skip in ['embed', 'lm_head', 'classifier']):
                    linear_layers.append(name.split('.')[-1])  # Get just the layer name
        
        # Remove duplicates while preserving order
        unique_layers = []
        for layer in linear_layers:
            if layer not in unique_layers:
                unique_layers.append(layer)
        
        # For DialoGPT, prioritize c_attn and c_proj layers
        attention_layers = [layer for layer in unique_layers if any(attn in layer for attn in ['c_attn', 'c_proj'])]
        
        if attention_layers:
            logger.info(f"Found DialoGPT attention layers: {attention_layers}")
            return attention_layers
        else:
            # Fallback: look for other attention patterns
            other_attention = [layer for layer in unique_layers if any(attn in layer for attn in ['attn', 'attention'])]
            if other_attention:
                logger.info(f"Found other attention layers: {other_attention[:2]}")
                return other_attention[:2]
            else:
                # Last resort: use first 2 linear layers
                logger.info(f"Using first 2 linear layers: {unique_layers[:2]}")
                return unique_layers[:2]
    
    def verify_target_modules(self, model: PreTrainedModel, target_modules: List[str]) -> List[str]:
        """Verify that target modules exist in the model"""
        all_module_names = [name for name, _ in model.named_modules()]
        
        # Find modules that match the target pattern
        valid_modules = []
        for target in target_modules:
            # Check for exact matches or modules ending with the target name
            matches = [name for name in all_module_names if name.endswith(f".{target}")]
            if matches:
                valid_modules.append(target)
                logger.info(f"✓ Found target module '{target}' in model")
            else:
                logger.warning(f"✗ Target module '{target}' not found in model")
        
        return valid_modules
    
    def setup_lora_model(self, model: PreTrainedModel = None) -> PreTrainedModel:
        """Setup LoRA configuration for fine-tuning"""
        try:
            model = model or self.model
            if model is None:
                raise ValueError("No model available. Load base model first.")
            
            logger.info("Setting up LoRA configuration")
            
            # Get target modules from config
            target_modules = self.config.lora.target_modules.copy()
            logger.info(f"Initial target modules: {target_modules}")
            
            # Verify target modules exist in the model
            valid_modules = self.verify_target_modules(model, target_modules)
            
            if not valid_modules:
                logger.warning("No valid target modules found, attempting auto-detection")
                target_modules = self.find_target_modules(model)
                logger.info(f"Auto-detected target modules: {target_modules}")
            else:
                target_modules = valid_modules
                logger.info(f"Using verified target modules: {target_modules}")
            
            if not target_modules:
                raise ValueError("No suitable target modules found for LoRA")
            
            # Create LoRA configuration
            lora_config = LoraConfig(
                r=self.config.lora.rank,
                lora_alpha=self.config.lora.alpha,
                lora_dropout=self.config.lora.dropout,
                target_modules=target_modules,
                bias=self.config.lora.bias,
                task_type=TaskType.CAUSAL_LM,
            )
            
            logger.info(f"LoRA Config:")
            logger.info(f"  Rank: {lora_config.r}")
            logger.info(f"  Alpha: {lora_config.lora_alpha}")
            logger.info(f"  Dropout: {lora_config.lora_dropout}")
            logger.info(f"  Target Modules: {lora_config.target_modules}")
            
            # Apply LoRA to model
            lora_model = get_peft_model(model, lora_config)
            
            # Print trainable parameters
            lora_model.print_trainable_parameters()
            
            self.model = lora_model
            logger.info("LoRA model setup completed")
            return lora_model
            
        except Exception as e:
            logger.error(f"Failed to setup LoRA model: {e}")
            raise
    
    def load_lora_adapter(self, adapter_path: str, model: PreTrainedModel = None) -> PreTrainedModel:
        """Load LoRA adapter from saved checkpoint"""
        try:
            model = model or self.model
            if model is None:
                raise ValueError("No model available. Load base model first.")
            
            adapter_path = Path(adapter_path)
            if not adapter_path.exists():
                raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
            
            logger.info(f"Loading LoRA adapter from: {adapter_path}")
            
            # Load PEFT model with adapter
            peft_model = PeftModel.from_pretrained(
                model,
                str(adapter_path),
                torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
            )
            
            self.model = peft_model
            logger.info("LoRA adapter loaded successfully")
            return peft_model
            
        except Exception as e:
            logger.error(f"Failed to load LoRA adapter: {e}")
            raise
    
    def load_finetuned_model(self, model_path: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load complete fine-tuned model with tokenizer"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
            
            logger.info(f"Loading fine-tuned model from: {model_path}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                device_map="auto" if torch.cuda.is_available() and self.device != "cpu" else None,
            )
            
            # Move to device if needed
            if not torch.cuda.is_available() or self.device == "cpu":
                model = model.to(self.device)
            
            self.model = model
            self.tokenizer = tokenizer
            
            logger.info("Fine-tuned model loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            raise
    
    def save_model(self, save_path: str, save_tokenizer: bool = True):
        """Save current model and tokenizer"""
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            if self.model is None:
                raise ValueError("No model to save")
            
            logger.info(f"Saving model to: {save_path}")
            
            # Save model
            self.model.save_pretrained(str(save_path))
            
            # Save tokenizer
            if save_tokenizer and self.tokenizer is not None:
                self.tokenizer.save_pretrained(str(save_path))
                logger.info("Tokenizer saved")
            
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current loaded model"""
        info = {
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "device": self.device,
            "model_type": None,
            "trainable_params": 0,
            "total_params": 0
        }
        
        if self.model is not None:
            info["model_type"] = type(self.model).__name__
            
            # Count parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            
            info["trainable_params"] = trainable_params
            info["total_params"] = total_params
            info["trainable_ratio"] = trainable_params / total_params if total_params > 0 else 0
        
        return info
    
    def unload_model(self):
        """Unload model and free memory"""
        if self.model is not None:
            del self.model
            self.model = None
            logger.info("Model unloaded")
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            logger.info("Tokenizer unloaded")
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    
    def set_training_mode(self, training: bool = True):
        """Set model training mode"""
        if self.model is not None:
            self.model.train(training)
            logger.info(f"Model set to {'training' if training else 'evaluation'} mode")
    
    def set_eval_mode(self):
        """Set model to evaluation mode"""
        self.set_training_mode(False)

# Convenience functions
def load_base_model(model_name: str = None, config: TrainingConfig = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load base model and tokenizer"""
    loader = ModelLoader(config)
    tokenizer = loader.load_tokenizer(model_name)
    model = loader.load_base_model(model_name)
    return model, tokenizer

def load_finetuned_model(model_path: str, config: TrainingConfig = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load fine-tuned model and tokenizer"""
    loader = ModelLoader(config)
    return loader.load_finetuned_model(model_path)

if __name__ == "__main__":
    # Test model loader
    try:
        logger.info("Testing ModelLoader...")
        
        # Test base model loading
        loader = ModelLoader()
        
        # Load tokenizer
        tokenizer = loader.load_tokenizer()
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        
        # Load base model
        model = loader.load_base_model()
        
        # Get model info
        info = loader.get_model_info()
        print(f"Model info: {info}")
        
        # Test LoRA setup
        lora_model = loader.setup_lora_model()
        
        # Get updated info
        lora_info = loader.get_model_info()
        print(f"LoRA model info: {lora_info}")
        
        print("ModelLoader test completed successfully!")
        
    except Exception as e:
        print(f"ModelLoader test failed: {e}")
    finally:
        # Clean up
        if 'loader' in locals():
            loader.unload_model()