import os
import sys
import json
import logging
import torch
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import Dataset

# Import project modules
sys.path.append(str(Path(__file__).parent.parent))
from .config import TrainingConfig, training_config
from models.model_loader import ModelLoader
from utils.helpers import (
    setup_logger, 
    format_time, 
    get_memory_usage, 
    save_json,
    create_checkpoint_name
)

# Setup logging
logger = setup_logger(__name__)

class AnimeQATrainer:
    """Trainer for AnimeQA model fine-tuning"""
    
    def __init__(self, config: TrainingConfig = None):
        """Initialize trainer"""
        self.config = config or training_config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.model_loader = ModelLoader(self.config)
        
        # Training state
        self.training_started = False
        self.start_time = None
        self.best_eval_loss = float('inf')
        self.training_history = []
        
        logger.info("AnimeQATrainer initialized")
        logger.info(f"Device: {self.config.device}")
        logger.info(f"FP16: {self.config.fp16}")
        logger.info(f"LoRA rank: {self.config.lora.rank}")
    
    def setup_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Setup model and tokenizer for training"""
        try:
            logger.info("Setting up model and tokenizer for training...")
            
            # Load tokenizer
            self.tokenizer = self.model_loader.load_tokenizer()
            
            # Load base model
            base_model = self.model_loader.load_base_model()
            
            # Setup LoRA configuration
            self.model = self.model_loader.setup_lora_model(base_model)
            
            # Get model info
            model_info = self.model_loader.get_model_info()
            logger.info(f"Model setup completed:")
            logger.info(f"  Total parameters: {model_info['total_params']:,}")
            logger.info(f"  Trainable parameters: {model_info['trainable_params']:,}")
            logger.info(f"  Trainable ratio: {model_info['trainable_ratio']:.2%}")
            
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Failed to setup model and tokenizer: {e}")
            raise
    
    def prepare_datasets(self, train_dataset: Dataset, val_dataset: Dataset = None) -> Tuple[Dataset, Optional[Dataset]]:
        """Prepare datasets for training"""
        try:
            logger.info("Preparing datasets for training...")
            
            if not isinstance(train_dataset, Dataset):
                raise ValueError("train_dataset must be a HuggingFace Dataset object")
            
            logger.info(f"Training dataset size: {len(train_dataset)}")
            
            if val_dataset:
                if not isinstance(val_dataset, Dataset):
                    raise ValueError("val_dataset must be a HuggingFace Dataset object")
                logger.info(f"Validation dataset size: {len(val_dataset)}")
            else:
                logger.info("No validation dataset provided")
            
            # Verify dataset format
            sample = train_dataset[0]
            required_fields = ['input_ids', 'attention_mask', 'labels']
            for field in required_fields:
                if field not in sample:
                    raise ValueError(f"Dataset missing required field: {field}")
            
            logger.info("Dataset preparation completed")
            return train_dataset, val_dataset
            
        except Exception as e:
            logger.error(f"Failed to prepare datasets: {e}")
            raise
    
    def create_data_collator(self):
        """Create data collator for training"""
        try:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # We're doing causal language modeling, not masked LM
                pad_to_multiple_of=8 if self.config.fp16 else None,
            )
            
            logger.info("Data collator created")
            return data_collator
            
        except Exception as e:
            logger.error(f"Failed to create data collator: {e}")
            raise
    
    def create_training_arguments(self) -> TrainingArguments:
        """Create training arguments"""
        try:
            # Get base training arguments from config
            training_args_dict = self.config.get_training_args()
            
            # Add additional arguments specific to our training
            # 修复参数名称以兼容新版本的 Transformers
            training_args_dict.update({
                "eval_strategy": "steps" if training_args_dict.get("eval_steps") else "no",  # 修改这里
                "save_strategy": "steps",
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "max_grad_norm": self.config.max_grad_norm,
                "prediction_loss_only": True,
                "group_by_length": True,
                "logging_first_step": True,
                "do_eval": True if training_args_dict.get("eval_steps") else False,  # 添加这个
            })
            
            # 移除可能导致冲突的旧参数名
            if "evaluation_strategy" in training_args_dict:
                del training_args_dict["evaluation_strategy"]
            
            # Handle resume from checkpoint
            if self.config.resume_from_checkpoint:
                training_args_dict["resume_from_checkpoint"] = self.config.resume_from_checkpoint
                logger.info(f"Will resume from checkpoint: {self.config.resume_from_checkpoint}")
            
            training_args = TrainingArguments(**training_args_dict)
            
            logger.info("Training arguments created:")
            logger.info(f"  Output dir: {training_args.output_dir}")
            logger.info(f"  Learning rate: {training_args.learning_rate}")
            logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
            logger.info(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
            logger.info(f"  Epochs: {training_args.num_train_epochs}")
            logger.info(f"  Eval strategy: {training_args.eval_strategy}")
            
            return training_args
            
        except Exception as e:
            logger.error(f"Failed to create training arguments: {e}")
            raise
    
    def create_callbacks(self) -> list:
        """Create training callbacks"""
        callbacks = []
        
        # Early stopping callback
        if self.config.early_stopping_patience > 0:
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=self.config.early_stopping_patience,
                early_stopping_threshold=self.config.early_stopping_threshold
            )
            callbacks.append(early_stopping)
            logger.info(f"Added early stopping with patience: {self.config.early_stopping_patience}")
        
        return callbacks
    
    def setup_trainer(self, train_dataset: Dataset, val_dataset: Dataset = None) -> Trainer:
        """Setup Hugging Face Trainer"""
        try:
            logger.info("Setting up Trainer...")
            
            # Ensure model and tokenizer are ready
            if self.model is None or self.tokenizer is None:
                self.setup_model_and_tokenizer()
            
            # Prepare datasets
            train_dataset, val_dataset = self.prepare_datasets(train_dataset, val_dataset)
            
            # Create components
            data_collator = self.create_data_collator()
            training_args = self.create_training_arguments()
            callbacks = self.create_callbacks()
            
            # Create trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                callbacks=callbacks,
            )
            
            logger.info("Trainer setup completed")
            return self.trainer
            
        except Exception as e:
            logger.error(f"Failed to setup trainer: {e}")
            raise
    
    def save_training_info(self, output_dir: str, training_result: Any = None):
        """Save training information and metadata"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Training metadata
            training_info = {
                "model_name": self.config.base_model_name,
                "training_config": self.config.to_dict(),
                "training_completed_at": datetime.now().isoformat(),
                "training_history": self.training_history
            }
            
            if training_result:
                training_info.update({
                    "final_training_loss": training_result.training_loss,
                    "training_steps": training_result.global_step,
                })
            
            # Save training info
            info_file = output_path / "training_info.json"
            save_json(training_info, info_file)
            
            logger.info(f"Training info saved to: {info_file}")
            
        except Exception as e:
            logger.error(f"Failed to save training info: {e}")
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset = None) -> Dict[str, Any]:
        """Main training function"""
        try:
            logger.info("=" * 60)
            logger.info("STARTING ANIMEQA MODEL TRAINING")
            logger.info("=" * 60)
            
            # Log system info
            memory_info = get_memory_usage()
            logger.info(f"System memory: {memory_info}")
            
            self.start_time = time.time()
            self.training_started = True
            
            # Setup trainer
            trainer = self.setup_trainer(train_dataset, val_dataset)
            
            # Start training
            logger.info("Starting training process...")
            training_result = trainer.train()
            
            # Save model and training info
            logger.info("Saving trained model...")
            trainer.save_model()
            trainer.save_state()
            
            # Save training metadata
            self.save_training_info(self.config.model_save_path, training_result)
            
            # Calculate training time
            training_time = time.time() - self.start_time
            
            # Prepare return data
            result = {
                "success": True,
                "model_path": self.config.model_save_path,
                "training_loss": training_result.training_loss,
                "training_steps": training_result.global_step,
                "training_time": training_time,
                "training_time_formatted": format_time(training_time)
            }
            
            # Add evaluation metrics if available
            if hasattr(training_result, 'eval_loss'):
                result["best_eval_loss"] = training_result.eval_loss
            
            logger.info("=" * 60)
            logger.info("TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"Final training loss: {result['training_loss']:.4f}")
            logger.info(f"Training steps: {result['training_steps']}")
            logger.info(f"Training time: {result['training_time_formatted']}")
            logger.info(f"Model saved to: {result['model_path']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """Evaluate model on dataset"""
        try:
            if self.trainer is None:
                raise ValueError("Trainer not initialized. Run train() first.")
            
            logger.info("Starting model evaluation...")
            eval_result = self.trainer.evaluate(eval_dataset)
            
            logger.info(f"Evaluation completed. Loss: {eval_result.get('eval_loss', 'N/A')}")
            return eval_result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def save_checkpoint(self, checkpoint_name: str = None) -> str:
        """Save training checkpoint"""
        try:
            if self.trainer is None:
                raise ValueError("Trainer not initialized")
            
            checkpoint_name = checkpoint_name or create_checkpoint_name()
            checkpoint_dir = Path(self.config.model_save_path) / "checkpoints" / checkpoint_name
            
            # Save checkpoint
            self.trainer.save_model(str(checkpoint_dir))
            self.trainer.save_state()
            
            logger.info(f"Checkpoint saved to: {checkpoint_dir}")
            return str(checkpoint_dir)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def cleanup(self):
        """Cleanup resources"""
        if self.model_loader:
            self.model_loader.unload_model()
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Trainer cleanup completed")

if __name__ == "__main__":
    # Test trainer setup
    try:
        trainer = AnimeQATrainer()
        
        # Test model setup
        model, tokenizer = trainer.setup_model_and_tokenizer()
        print(f"Model type: {type(model)}")
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        
        # Test training arguments
        training_args = trainer.create_training_arguments()
        print(f"Training output dir: {training_args.output_dir}")
        
        print("Trainer test completed successfully!")
        
    except Exception as e:
        print(f"Trainer test failed: {e}")
    finally:
        if 'trainer' in locals():
            trainer.cleanup()