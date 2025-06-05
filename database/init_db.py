import sqlite3
from pathlib import Path
import logging
import json

from config import get_database_path, settings
from .models import DatabaseConnection, QAPairModel, ModelInfoModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseInitializer:
    """Database initialization and management"""
    
    def __init__(self):
        """Initialize database initializer"""
        self.db_path = get_database_path()
        self.db_connection = DatabaseConnection(self.db_path)
        self.qa_model = QAPairModel(self.db_connection)
        self.model_info = ModelInfoModel(self.db_connection)
    
    def check_database_exists(self) -> bool:
        """Check if database file exists"""
        return Path(self.db_path).exists()
    
    def create_database(self):
        """Create database and all tables"""
        try:
            logger.info(f"Creating database at: {self.db_path}")
            
            # Ensure database directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create tables
            self.qa_model.create_table()
            self.model_info.create_table()
            
            logger.info("Database created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            raise
    
    def reset_database(self):
        """Reset database (delete and recreate)"""
        try:
            logger.warning("Resetting database - all data will be lost!")
            
            # Remove existing database file
            db_file = Path(self.db_path)
            if db_file.exists():
                db_file.unlink()
                logger.info("Existing database file removed")
            
            # Recreate database
            self.create_database()
            logger.info("Database reset completed")
            
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            raise
    
    def load_sample_data(self):
        """Load sample QA pairs for testing"""
        sample_qa_pairs = [
            {
                "question": "Who is Naruto Uzumaki?",
                "answer": "Naruto Uzumaki is the main protagonist of the Naruto series. He is a young ninja who dreams of becoming the Hokage of his village.",
                "character": "Naruto Uzumaki",
                "anime": "Naruto"
            },
            {
                "question": "What is Naruto's favorite food?",
                "answer": "Naruto's favorite food is ramen, especially from Ichiraku Ramen shop in the Hidden Leaf Village.",
                "character": "Naruto Uzumaki", 
                "anime": "Naruto"
            },
            {
                "question": "Who is Sasuke Uchiha?",
                "answer": "Sasuke Uchiha is one of the main characters in Naruto. He is Naruto's rival and best friend, and the last surviving member of the Uchiha clan.",
                "character": "Sasuke Uchiha",
                "anime": "Naruto"
            },
            {
                "question": "What is the Sharingan?",
                "answer": "The Sharingan is a special eye technique possessed by members of the Uchiha clan. It allows them to copy jutsu and see through illusions.",
                "character": "Sasuke Uchiha",
                "anime": "Naruto"
            },
            {
                "question": "Who is Monkey D. Luffy?",
                "answer": "Monkey D. Luffy is the main protagonist of One Piece. He is a pirate who dreams of becoming the Pirate King and finding the legendary treasure One Piece.",
                "character": "Monkey D. Luffy",
                "anime": "One Piece"
            }
        ]
        
        try:
            logger.info("Loading sample QA pairs...")
            inserted_count = self.qa_model.batch_insert_qa_pairs(sample_qa_pairs)
            logger.info(f"Loaded {inserted_count} sample QA pairs")
            
        except Exception as e:
            logger.error(f"Failed to load sample data: {e}")
            raise
    
    def load_processed_data(self, data_file: str = None):
        """Load processed QA data from file"""
        if not data_file:
            # Use default processed data file
            data_file = Path(settings.data.processed_path) / "train_qa_dataset.json"
        
        data_file = Path(data_file)
        
        if not data_file.exists():
            logger.warning(f"Data file not found: {data_file}")
            return 0
        
        try:
            logger.info(f"Loading processed data from: {data_file}")
            
            with open(data_file, 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)
            
            if not qa_pairs:
                logger.warning("No QA pairs found in data file")
                return 0
            
            # Batch insert QA pairs
            inserted_count = self.qa_model.batch_insert_qa_pairs(qa_pairs)
            logger.info(f"Loaded {inserted_count} QA pairs from processed data")
            
            return inserted_count
            
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
            raise
    
    def create_default_model_info(self):
        """Create default model information entry"""
        try:
            # Check if any model info exists
            existing_models = self.model_info.get_all_models()
            
            if not existing_models:
                logger.info("Creating default model info entry...")
                
                training_config = json.dumps({
                    "base_model": settings.model.base_model_name,
                    "lora_rank": settings.training.lora_config.rank,
                    "lora_alpha": settings.training.lora_config.alpha,
                    "learning_rate": settings.training.learning_rate,
                    "batch_size": settings.training.batch_size,
                    "num_epochs": settings.training.num_epochs
                })
                
                self.model_info.insert_model_info(
                    model_name="anime-qa-model-v1",
                    model_path=settings.model.save_path,
                    base_model=settings.model.base_model_name,
                    training_data_size=0,  # Will be updated after training
                    training_config=training_config
                )
                
                # Set as active model
                self.model_info.set_active_model("anime-qa-model-v1")
                logger.info("Default model info created")
            
        except Exception as e:
            logger.error(f"Failed to create default model info: {e}")
            raise
    
    def get_database_stats(self) -> dict:
        """Get database statistics"""
        try:
            stats = {
                "database_path": self.db_path,
                "database_exists": self.check_database_exists(),
                "qa_pairs_count": self.qa_model.get_qa_count(),
                "models_count": len(self.model_info.get_all_models()),
                "active_model": self.model_info.get_active_model()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}
    
    def initialize_full_database(self, load_sample: bool = False, 
                                load_processed: bool = True):
        """Complete database initialization"""
        try:
            logger.info("Starting full database initialization...")
            
            # Create database and tables
            if not self.check_database_exists():
                self.create_database()
            else:
                logger.info("Database already exists")
            
            # Load sample data if requested
            if load_sample:
                self.load_sample_data()
            
            # Load processed data if available
            if load_processed:
                try:
                    self.load_processed_data()
                except Exception as e:
                    logger.warning(f"Could not load processed data: {e}")
            
            # Create default model info
            self.create_default_model_info()
            
            # Display stats
            stats = self.get_database_stats()
            logger.info("Database initialization completed!")
            logger.info(f"QA pairs: {stats['qa_pairs_count']}")
            logger.info(f"Models: {stats['models_count']}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

def init_database(reset: bool = False, load_sample: bool = False, 
                 load_processed: bool = True) -> dict:
    """Initialize database with options"""
    initializer = DatabaseInitializer()
    
    try:
        if reset:
            initializer.reset_database()
        
        return initializer.initialize_full_database(
            load_sample=load_sample,
            load_processed=load_processed
        )
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize AnimeQA database")
    parser.add_argument("--reset", action="store_true", 
                       help="Reset database (delete existing)")
    parser.add_argument("--sample", action="store_true",
                       help="Load sample data")
    parser.add_argument("--no-processed", action="store_true",
                       help="Skip loading processed data")
    
    args = parser.parse_args()
    
    try:
        stats = init_database(
            reset=args.reset,
            load_sample=args.sample,
            load_processed=not args.no_processed
        )
        
        print("\nDatabase Initialization Summary:")
        print(f"Database Path: {stats['database_path']}")
        print(f"QA Pairs: {stats['qa_pairs_count']}")
        print(f"Models: {stats['models_count']}")
        
        if stats['active_model']:
            print(f"Active Model: {stats['active_model']['model_name']}")
        
        print("\nInitialization completed successfully!")
        
    except Exception as e:
        print(f"Initialization failed: {e}")
        exit(1)