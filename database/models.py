import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging

from config import get_database_path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Database connection manager"""
    
    def __init__(self, db_path: str = None):
        """Initialize database connection"""
        self.db_path = db_path or get_database_path()
        # Ensure database directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        return conn
    
    def execute_query(self, query: str, params: tuple = None) -> List[sqlite3.Row]:
        """Execute SELECT query and return results"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def execute_command(self, command: str, params: tuple = None) -> int:
        """Execute INSERT/UPDATE/DELETE command and return affected rows"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(command, params)
                else:
                    cursor.execute(command)
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise

class QAPairModel:
    """Model for managing QA pairs data"""
    
    def __init__(self, db_connection: DatabaseConnection = None):
        """Initialize QA pair model"""
        self.db = db_connection or DatabaseConnection()
    
    def create_table(self):
        """Create qa_pairs table"""
        create_sql = """
        CREATE TABLE IF NOT EXISTS qa_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            character TEXT,
            anime TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # Create index for better query performance
        index_sql = """
        CREATE INDEX IF NOT EXISTS idx_qa_pairs_character ON qa_pairs(character);
        CREATE INDEX IF NOT EXISTS idx_qa_pairs_anime ON qa_pairs(anime);
        """
        
        try:
            self.db.execute_command(create_sql)
            self.db.execute_command(index_sql)
            logger.info("QA pairs table created successfully")
        except Exception as e:
            logger.error(f"Failed to create qa_pairs table: {e}")
            raise
    
    def insert_qa_pair(self, question: str, answer: str, 
                      character: str = None, anime: str = None) -> int:
        """Insert a single QA pair"""
        insert_sql = """
        INSERT INTO qa_pairs (question, answer, character, anime)
        VALUES (?, ?, ?, ?)
        """
        
        try:
            self.db.execute_command(insert_sql, (question, answer, character, anime))
            logger.debug(f"Inserted QA pair: {question[:50]}...")
            return 1
        except Exception as e:
            logger.error(f"Failed to insert QA pair: {e}")
            raise
    
    def batch_insert_qa_pairs(self, qa_pairs: List[Dict[str, Any]]) -> int:
        """Batch insert multiple QA pairs"""
        insert_sql = """
        INSERT INTO qa_pairs (question, answer, character, anime)
        VALUES (?, ?, ?, ?)
        """
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                data_tuples = []
                for qa_pair in qa_pairs:
                    data_tuples.append((
                        qa_pair.get('question', ''),
                        qa_pair.get('answer', ''),
                        qa_pair.get('character'),
                        qa_pair.get('anime')
                    ))
                
                cursor.executemany(insert_sql, data_tuples)
                conn.commit()
                
                inserted_count = cursor.rowcount
                logger.info(f"Batch inserted {inserted_count} QA pairs")
                return inserted_count
                
        except Exception as e:
            logger.error(f"Failed to batch insert QA pairs: {e}")
            raise
    
    def get_qa_pairs(self, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Get QA pairs with optional pagination"""
        base_sql = "SELECT * FROM qa_pairs ORDER BY created_at DESC"
        
        if limit:
            sql = f"{base_sql} LIMIT {limit} OFFSET {offset}"
        else:
            sql = base_sql
        
        try:
            rows = self.db.execute_query(sql)
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get QA pairs: {e}")
            raise
    
    def get_qa_pairs_by_character(self, character: str) -> List[Dict[str, Any]]:
        """Get QA pairs for a specific character"""
        sql = "SELECT * FROM qa_pairs WHERE character = ? ORDER BY created_at DESC"
        
        try:
            rows = self.db.execute_query(sql, (character,))
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get QA pairs for character {character}: {e}")
            raise
    
    def get_qa_count(self) -> int:
        """Get total count of QA pairs"""
        sql = "SELECT COUNT(*) as count FROM qa_pairs"
        
        try:
            result = self.db.execute_query(sql)
            return result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"Failed to get QA count: {e}")
            raise
    
    def delete_all_qa_pairs(self) -> int:
        """Delete all QA pairs (for development/testing)"""
        sql = "DELETE FROM qa_pairs"
        
        try:
            deleted_count = self.db.execute_command(sql)
            logger.info(f"Deleted {deleted_count} QA pairs")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete QA pairs: {e}")
            raise

class ModelInfoModel:
    """Model for managing model information"""
    
    def __init__(self, db_connection: DatabaseConnection = None):
        """Initialize model info model"""
        self.db = db_connection or DatabaseConnection()
    
    def create_table(self):
        """Create model_info table"""
        create_sql = """
        CREATE TABLE IF NOT EXISTS model_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL UNIQUE,
            model_path TEXT NOT NULL,
            base_model TEXT,
            training_data_size INTEGER DEFAULT 0,
            training_config TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT FALSE
        )
        """
        
        try:
            self.db.execute_command(create_sql)
            logger.info("Model info table created successfully")
        except Exception as e:
            logger.error(f"Failed to create model_info table: {e}")
            raise
    
    def insert_model_info(self, model_name: str, model_path: str, 
                         base_model: str = None, training_data_size: int = 0,
                         training_config: str = None) -> int:
        """Insert model information"""
        insert_sql = """
        INSERT INTO model_info (model_name, model_path, base_model, 
                               training_data_size, training_config)
        VALUES (?, ?, ?, ?, ?)
        """
        
        try:
            self.db.execute_command(insert_sql, (
                model_name, model_path, base_model, 
                training_data_size, training_config
            ))
            logger.info(f"Inserted model info: {model_name}")
            return 1
        except Exception as e:
            logger.error(f"Failed to insert model info: {e}")
            raise
    
    def set_active_model(self, model_name: str) -> int:
        """Set a model as active (deactivate others)"""
        try:
            # Deactivate all models
            self.db.execute_command("UPDATE model_info SET is_active = FALSE")
            
            # Activate specified model
            update_sql = "UPDATE model_info SET is_active = TRUE WHERE model_name = ?"
            updated_count = self.db.execute_command(update_sql, (model_name,))
            
            if updated_count > 0:
                logger.info(f"Set active model: {model_name}")
            else:
                logger.warning(f"Model not found: {model_name}")
            
            return updated_count
            
        except Exception as e:
            logger.error(f"Failed to set active model: {e}")
            raise
    
    def get_active_model(self) -> Optional[Dict[str, Any]]:
        """Get currently active model"""
        sql = "SELECT * FROM model_info WHERE is_active = TRUE LIMIT 1"
        
        try:
            rows = self.db.execute_query(sql)
            return dict(rows[0]) if rows else None
        except Exception as e:
            logger.error(f"Failed to get active model: {e}")
            raise
    
    def get_all_models(self) -> List[Dict[str, Any]]:
        """Get all model information"""
        sql = "SELECT * FROM model_info ORDER BY created_at DESC"
        
        try:
            rows = self.db.execute_query(sql)
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get all models: {e}")
            raise
    
    def delete_model_info(self, model_name: str) -> int:
        """Delete model information"""
        sql = "DELETE FROM model_info WHERE model_name = ?"
        
        try:
            deleted_count = self.db.execute_command(sql, (model_name,))
            logger.info(f"Deleted model info: {model_name}")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete model info: {e}")
            raise

if __name__ == "__main__":
    # Test database models
    try:
        # Test QA pair model
        qa_model = QAPairModel()
        qa_model.create_table()
        
        # Test model info model
        model_info = ModelInfoModel()
        model_info.create_table()
        
        print("Database models test completed successfully!")
        print(f"Database path: {get_database_path()}")
        print(f"QA pairs count: {qa_model.get_qa_count()}")
        
    except Exception as e:
        print(f"Database models test failed: {e}")