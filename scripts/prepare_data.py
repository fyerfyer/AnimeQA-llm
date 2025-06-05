"""
Data preparation script for AnimeQA project
Integrates all data processing and database initialization modules
"""

import argparse
import sys
from pathlib import Path
import logging
from typing import Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import settings, ensure_directories
from data import AnimeDataProcessor, AnimeQADatasetBuilder
from database import init_database, QAPairModel, DatabaseInitializer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreparationPipeline:
    """Complete data preparation pipeline"""
    
    def __init__(self):
        """Initialize data preparation pipeline"""
        self.data_processor = AnimeDataProcessor()
        self.dataset_builder = AnimeQADatasetBuilder(self.data_processor)
        self.qa_model = QAPairModel()
        self.db_initializer = DatabaseInitializer()
        
        # Ensure all necessary directories exist
        ensure_directories()
    
    def step1_process_raw_data(self, skip_if_exists: bool = True) -> bool:
        """Step 1: Process raw dataset and save processed QA pairs"""
        logger.info("=" * 60)
        logger.info("STEP 1: Processing raw dataset")
        logger.info("=" * 60)
        
        try:
            # Check if processed data already exists
            processed_file = Path(settings.data.processed_path) / "processed_qa_dataset.json"
            
            if skip_if_exists and processed_file.exists():
                logger.info(f"Processed data already exists: {processed_file}")
                logger.info("Skipping raw data processing (use --force to override)")
                return True
            
            # Process raw dataset
            logger.info("Loading and processing raw dataset...")
            qa_pairs = self.data_processor.process_full_pipeline(save_output=True)
            
            if not qa_pairs:
                logger.error("No QA pairs generated from raw dataset")
                return False
            
            # Display statistics
            stats = self.data_processor.get_dataset_statistics(qa_pairs)
            logger.info(f"Processing completed successfully!")
            logger.info(f"Generated {stats['total_pairs']} QA pairs")
            logger.info(f"Unique characters: {stats['unique_characters']}")
            logger.info(f"Unique anime: {stats['unique_anime']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Step 1 failed: {e}")
            return False
    
    def step2_initialize_database(self, reset_db: bool = False, 
                                 load_sample: bool = False) -> bool:
        """Step 2: Initialize database and load processed data"""
        logger.info("=" * 60)
        logger.info("STEP 2: Initializing database")
        logger.info("=" * 60)
        
        try:
            # Initialize database
            stats = init_database(
                reset=reset_db,
                load_sample=load_sample,
                load_processed=True
            )
            
            logger.info(f"Database initialization completed!")
            logger.info(f"Database path: {stats['database_path']}")
            logger.info(f"QA pairs in database: {stats['qa_pairs_count']}")
            logger.info(f"Models registered: {stats['models_count']}")
            
            if stats['active_model']:
                logger.info(f"Active model: {stats['active_model']['model_name']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Step 2 failed: {e}")
            return False
    
    def step3_build_training_datasets(self, save_datasets: bool = True) -> bool:
        """Step 3: Build training and validation datasets"""
        logger.info("=" * 60)
        logger.info("STEP 3: Building training datasets")
        logger.info("=" * 60)
        
        try:
            # Check if datasets already exist
            train_dataset_path = Path(settings.data.processed_path) / "train_dataset"
            val_dataset_path = Path(settings.data.processed_path) / "val_dataset"
            
            if save_datasets and train_dataset_path.exists() and val_dataset_path.exists():
                logger.info("Training datasets already exist")
                
                # Load existing datasets to get info
                try:
                    train_dataset = self.dataset_builder.load_dataset("train_dataset")
                    val_dataset = self.dataset_builder.load_dataset("val_dataset")
                    
                    logger.info(f"Loaded existing datasets:")
                    logger.info(f"Training samples: {len(train_dataset)}")
                    logger.info(f"Validation samples: {len(val_dataset)}")
                    
                    return True
                    
                except Exception as e:
                    logger.warning(f"Failed to load existing datasets: {e}")
                    logger.info("Will rebuild datasets...")
            
            # Build training datasets
            logger.info("Building training and validation datasets...")
            train_dataset, val_dataset = self.dataset_builder.build_training_dataset(
                use_database=True,
                use_processed_file=True,
                save_datasets=save_datasets
            )
            
            # Display dataset information
            train_info = self.dataset_builder.get_dataset_info(train_dataset)
            val_info = self.dataset_builder.get_dataset_info(val_dataset)
            
            logger.info(f"Dataset building completed successfully!")
            logger.info(f"Training samples: {train_info['num_samples']}")
            logger.info(f"Validation samples: {val_info['num_samples']}")
            logger.info(f"Features: {train_info['features']}")
            
            if 'avg_sequence_length' in train_info:
                logger.info(f"Average sequence length: {train_info['avg_sequence_length']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Step 3 failed: {e}")
            return False
    
    def verify_preparation(self) -> bool:
        """Verify that all preparation steps completed successfully"""
        logger.info("=" * 60)
        logger.info("VERIFICATION: Checking preparation results")
        logger.info("=" * 60)
        
        try:
            # Check processed data file
            processed_file = Path(settings.data.processed_path) / "processed_qa_dataset.json"
            if not processed_file.exists():
                logger.error(f"Processed data file not found: {processed_file}")
                return False
            
            # Check database
            db_stats = self.db_initializer.get_database_stats()
            if db_stats.get('qa_pairs_count', 0) == 0:
                logger.error("No QA pairs found in database")
                return False
            
            # Check training datasets
            train_dataset_path = Path(settings.data.processed_path) / "train_dataset"
            val_dataset_path = Path(settings.data.processed_path) / "val_dataset"
            
            if not train_dataset_path.exists():
                logger.error(f"Training dataset not found: {train_dataset_path}")
                return False
            
            if not val_dataset_path.exists():
                logger.error(f"Validation dataset not found: {val_dataset_path}")
                return False
            
            # All checks passed
            logger.info("✓ Processed data file exists")
            logger.info(f"✓ Database contains {db_stats['qa_pairs_count']} QA pairs")
            logger.info("✓ Training dataset exists")
            logger.info("✓ Validation dataset exists")
            logger.info("All preparation steps verified successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False
    
    def run_full_pipeline(self, reset_db: bool = False, 
                         force_reprocess: bool = False,
                         load_sample: bool = False) -> bool:
        """Run the complete data preparation pipeline"""
        logger.info("Starting AnimeQA Data Preparation Pipeline...")
        logger.info(f"Configuration: reset_db={reset_db}, force_reprocess={force_reprocess}")
        
        success = True
        
        # Step 1: Process raw data
        if not self.step1_process_raw_data(skip_if_exists=not force_reprocess):
            success = False
        
        # Step 2: Initialize database
        if success and not self.step2_initialize_database(
            reset_db=reset_db, 
            load_sample=load_sample
        ):
            success = False
        
        # Step 3: Build training datasets
        if success and not self.step3_build_training_datasets(save_datasets=True):
            success = False
        
        # Verification
        if success and not self.verify_preparation():
            success = False
        
        # Final summary
        logger.info("=" * 60)
        if success:
            logger.info("🎉 DATA PREPARATION COMPLETED SUCCESSFULLY! 🎉")
            logger.info("The system is ready for model training.")
        else:
            logger.error("❌ DATA PREPARATION FAILED!")
            logger.error("Please check the error messages above and retry.")
        logger.info("=" * 60)
        
        return success

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Prepare data for AnimeQA model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preparation
  python prepare_data.py
  
  # Reset database and force reprocessing
  python prepare_data.py --reset-db --force
  
  # Include sample data for testing
  python prepare_data.py --sample
  
  # Only process raw data
  python prepare_data.py --step process
  
  # Only initialize database
  python prepare_data.py --step database --reset-db
        """
    )
    
    parser.add_argument(
        '--reset-db', 
        action='store_true',
        help='Reset database (delete existing data)'
    )
    
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Force reprocessing even if data exists'
    )
    
    parser.add_argument(
        '--sample', 
        action='store_true',
        help='Load sample data for testing'
    )
    
    parser.add_argument(
        '--step',
        choices=['process', 'database', 'datasets', 'verify'],
        help='Run only a specific step'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Initialize pipeline
    pipeline = DataPreparationPipeline()
    
    try:
        if args.step:
            # Run specific step
            if args.step == 'process':
                success = pipeline.step1_process_raw_data(skip_if_exists=not args.force)
            elif args.step == 'database':
                success = pipeline.step2_initialize_database(
                    reset_db=args.reset_db,
                    load_sample=args.sample
                )
            elif args.step == 'datasets':
                success = pipeline.step3_build_training_datasets(save_datasets=True)
            elif args.step == 'verify':
                success = pipeline.verify_preparation()
        else:
            # Run full pipeline
            success = pipeline.run_full_pipeline(
                reset_db=args.reset_db,
                force_reprocess=args.force,
                load_sample=args.sample
            )
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Data preparation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()