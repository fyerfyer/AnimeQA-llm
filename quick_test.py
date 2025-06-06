"""
Ultra-fast test training script - 只用很少样本快速验证
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set test configuration
os.environ['USE_TEST_CONFIG'] = 'true'

# Import after setting environment
from scripts.prepare_data import DataPreparationPipeline
from scripts.finetune import main as finetune_main
import argparse

def prepare_mini_test_data():
    """Prepare minimal test data"""
    print("Preparing ultra-minimal test data...")
    
    pipeline = DataPreparationPipeline()
    
    # Run with very limited sample data
    success = pipeline.run_full_pipeline(
        reset_db=True,
        force_reprocess=True,
        load_sample=True  # Use sample data
    )
    
    return success

def run_ultra_fast_training():
    """Run ultra-fast training"""
    print("Starting ultra-fast training...")
    
    # Override sys.argv for finetune script
    original_argv = sys.argv.copy()
    
    try:
        sys.argv = [
            'finetune.py',
            '--model-name', 'microsoft/DialoGPT-small',
            '--batch-size', '8',        # 增加批次大小
            '--num-epochs', '1',
            '--max-length', '128',      # 减少序列长度
            '--lora-rank', '8',
            '--lora-alpha', '16',
            '--learning-rate', '2e-4',  # 提高学习率
            '--log-level', 'INFO'
        ]
        
        # Run training
        result = finetune_main()
        return result == 0
        
    finally:
        sys.argv = original_argv

def main():
    """Main ultra-fast test function"""
    parser = argparse.ArgumentParser(description="Ultra-fast test with minimal samples")
    parser.add_argument('--data-only', action='store_true', help='Only prepare data')
    parser.add_argument('--train-only', action='store_true', help='Only run training')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ULTRA-FAST TEST (50 training samples + 10 validation samples)")
    print("=" * 60)
    print("Configuration: CPU/GPU optimized")
    print("Model: microsoft/DialoGPT-small")
    print("Expected time: 2-5 minutes")
    print("=" * 60)
    
    try:
        success = True
        
        # Data preparation
        if not args.train_only:
            print("\n1. Preparing ultra-minimal test data...")
            if not prepare_mini_test_data():
                print("❌ Data preparation failed!")
                return 1
            print("✅ Data preparation completed!")
        
        # Training
        if not args.data_only:
            print("\n2. Starting ultra-fast training...")
            if not run_ultra_fast_training():
                print("❌ Training failed!")
                return 1
            print("✅ Training completed!")
        
        print("\n" + "=" * 60)
        print("🚀 ULTRA-FAST TEST COMPLETED!")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Ultra-fast test failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())