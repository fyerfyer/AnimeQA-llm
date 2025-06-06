import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from training import AnimeQATrainer, TrainingConfig
from data import AnimeQADatasetBuilder
from utils.helpers import setup_logger, format_time, get_memory_usage, get_gpu_info
from config import settings, ensure_directories

def setup_logging(log_level: str = "INFO"):
    """Setup logging for training script"""
    # Create logs directory
    log_dir = Path(project_root) / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Setup logger with file output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"finetune_{timestamp}.log"
    
    logger = setup_logger(
        name="finetune",
        log_file=str(log_file),
        level=log_level
    )
    
    return logger

def load_datasets(dataset_builder: AnimeQADatasetBuilder, 
                 rebuild: bool = False) -> tuple:
    """Load or build training datasets"""
    logger = logging.getLogger("finetune")
    
    try:
        # Try to load existing datasets first
        if not rebuild:
            try:
                train_dataset = dataset_builder.load_dataset("train_dataset")
                val_dataset = dataset_builder.load_dataset("val_dataset")
                
                logger.info(f"Loaded existing datasets:")
                logger.info(f"  Train: {len(train_dataset)} samples")
                logger.info(f"  Validation: {len(val_dataset)} samples")
                
                return train_dataset, val_dataset
                
            except Exception as e:
                logger.warning(f"Failed to load existing datasets: {e}")
                logger.info("Will build new datasets...")
        
        # Build new datasets
        logger.info("Building training datasets...")
        train_dataset, val_dataset = dataset_builder.build_training_dataset(
            use_database=True,
            use_processed_file=True,
            save_datasets=True
        )
        
        return train_dataset, val_dataset
        
    except Exception as e:
        logger.error(f"Failed to load/build datasets: {e}")
        raise

def create_training_config(args) -> TrainingConfig:
    """Create training configuration from arguments"""
    config = TrainingConfig()
    
    # Override with command line arguments
    if args.model_name:
        config.base_model_name = args.model_name
    if args.output_dir:
        config.model_save_path = args.output_dir
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.max_length:
        config.max_length = args.max_length
    
    # LoRA configuration
    if args.lora_rank:
        config.lora.rank = args.lora_rank
    if args.lora_alpha:
        config.lora.alpha = args.lora_alpha
    if args.lora_dropout:
        config.lora.dropout = args.lora_dropout
    
    # Resume from checkpoint
    if args.resume_from_checkpoint:
        config.resume_from_checkpoint = args.resume_from_checkpoint
    
    return config

def print_system_info(logger):
    """Print system and environment information"""
    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)
    
    # Memory info
    memory_info = get_memory_usage()
    logger.info(f"System Memory: {memory_info}")
    
    # GPU info
    gpu_info = get_gpu_info()
    logger.info(f"CUDA Available: {gpu_info['cuda_available']}")
    if gpu_info['cuda_available']:
        logger.info(f"GPU Device: {gpu_info['current_device']}")
        logger.info(f"GPU Count: {gpu_info['device_count']}")
        if 'memory_allocated' in gpu_info:
            logger.info(f"GPU Memory: {gpu_info['memory_allocated']}")
    
    # Environment
    logger.info(f"HF Endpoint: {settings.huggingface.endpoint}")
    logger.info(f"Cache Dir: {settings.data.cache_dir}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description="Fine-tune AnimeQA model using LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default settings
  python scripts/finetune.py
  
  # Custom model and parameters
  python scripts/finetune.py --model-name microsoft/DialoGPT-small --batch-size 8 --num-epochs 5
  
  # Resume from checkpoint
  python scripts/finetune.py --resume-from-checkpoint ./models/checkpoints/checkpoint-1000
  
  # LoRA configuration
  python scripts/finetune.py --lora-rank 32 --lora-alpha 64 --learning-rate 1e-4
        """
    )
    
    # Model configuration
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Base model name (default: from config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for trained model (default: from config)'
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (default: 5e-5)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Training batch size (default: 4)'
    )
    
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=None,
        help='Number of training epochs (default: 3)'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=None,
        help='Maximum sequence length (default: 512)'
    )
    
    # LoRA configuration
    parser.add_argument(
        '--lora-rank',
        type=int,
        default=None,
        help='LoRA rank (default: 16)'
    )
    
    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=None,
        help='LoRA alpha (default: 32)'
    )
    
    parser.add_argument(
        '--lora-dropout',
        type=float,
        default=None,
        help='LoRA dropout (default: 0.1)'
    )
    
    # Training options
    parser.add_argument(
        '--resume-from-checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--rebuild-datasets',
        action='store_true',
        help='Rebuild datasets even if they exist'
    )
    
    parser.add_argument(
        '--no-eval',
        action='store_true',
        help='Skip evaluation during training'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    if not args.quiet:
        print_system_info(logger)
    
    try:
        logger.info("Starting AnimeQA model fine-tuning...")
        start_time = datetime.now()
        
        # Ensure directories exist
        ensure_directories()
        
        # Create training configuration
        config = create_training_config(args)
        logger.info("Training configuration created:")
        logger.info(f"  Base Model: {config.base_model_name}")
        logger.info(f"  Output Dir: {config.model_save_path}")
        logger.info(f"  Learning Rate: {config.learning_rate}")
        logger.info(f"  Batch Size: {config.batch_size}")
        logger.info(f"  Epochs: {config.num_epochs}")
        logger.info(f"  LoRA Rank: {config.lora.rank}")
        logger.info(f"  LoRA Alpha: {config.lora.alpha}")
        
        # Initialize dataset builder
        logger.info("Initializing dataset builder...")
        dataset_builder = AnimeQADatasetBuilder()
        
        # Load datasets
        train_dataset, val_dataset = load_datasets(
            dataset_builder, 
            rebuild=args.rebuild_datasets
        )
        
        # Skip validation if requested
        if args.no_eval:
            val_dataset = None
            logger.info("Skipping validation as requested")
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = AnimeQATrainer(config)
        
        # Start training
        logger.info("Starting training process...")
        training_result = trainer.train(train_dataset, val_dataset)
        
        # Training completed
        end_time = datetime.now()
        total_time = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Total Time: {format_time(total_time.total_seconds())}")
        logger.info(f"Model Path: {training_result['model_path']}")
        logger.info(f"Final Loss: {training_result['training_loss']:.4f}")
        
        if 'best_eval_loss' in training_result:
            logger.info(f"Best Eval Loss: {training_result['best_eval_loss']:.4f}")
        
        logger.info(f"Training Steps: {training_result['training_steps']}")
        
        # Print summary for console
        print(f"\n{'='*60}")
        print("FINE-TUNING COMPLETED!")
        print(f"{'='*60}")
        print(f"Model saved to: {training_result['model_path']}")
        print(f"Training time: {format_time(total_time.total_seconds())}")
        print(f"Final training loss: {training_result['training_loss']:.4f}")
        print(f"You can now use the model for inference!")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        print("\nTraining interrupted by user. Partial results may be saved.")
        return 1
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        print(f"\nTraining failed: {e}")
        return 1
        
    finally:
        # Cleanup
        if 'trainer' in locals():
            trainer.cleanup()

if __name__ == "__main__":
    exit(main())