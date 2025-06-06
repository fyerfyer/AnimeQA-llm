import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

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
    
    def execute_multiple_commands(self, commands: List[str]) -> int:
        """Execute multiple commands in a transaction"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                total_affected = 0
                for command in commands:
                    cursor.execute(command)
                    total_affected += cursor.rowcount
                conn.commit()
                return total_affected
        except Exception as e:
            logger.error(f"Multiple commands execution failed: {e}")
            raise

class QAPairModel:
    """Model for managing QA pairs data"""
    
    def __init__(self, db_connection: DatabaseConnection = None):
        """Initialize QA pair model"""
        self.db = db_connection or DatabaseConnection()
    
    def create_table(self):
        """Create qa_pairs table with proper SQL execution"""
        # 分别定义每个SQL语句
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS qa_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            character TEXT,
            anime TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # 索引语句分别执行
        index_character_sql = """
        CREATE INDEX IF NOT EXISTS idx_qa_pairs_character ON qa_pairs(character)
        """
        
        index_anime_sql = """
        CREATE INDEX IF NOT EXISTS idx_qa_pairs_anime ON qa_pairs(anime)
        """
        
        try:
            # 分别执行每条SQL语句
            self.db.execute_command(create_table_sql)
            logger.info("QA pairs table created successfully")
            
            self.db.execute_command(index_character_sql)
            logger.info("Character index created successfully")
            
            self.db.execute_command(index_anime_sql)
            logger.info("Anime index created successfully")
            
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
            affected_rows = self.db.execute_command(insert_sql, (question, answer, character, anime))
            logger.debug(f"Inserted QA pair: {question[:50]}...")
            return affected_rows
        except Exception as e:
            logger.error(f"Failed to insert QA pair: {e}")
            raise
    
    def batch_insert_qa_pairs(self, qa_pairs: List[Dict[str, Any]]) -> int:
        """Batch insert multiple QA pairs"""
        if not qa_pairs:
            return 0
        
        insert_sql = """
        INSERT INTO qa_pairs (question, answer, character, anime)
        VALUES (?, ?, ?, ?)
        """
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # 准备批量插入数据
                batch_data = []
                for qa_pair in qa_pairs:
                    question = qa_pair.get('question', '').strip()
                    answer = qa_pair.get('answer', '').strip()
                    character = qa_pair.get('character', '')
                    anime = qa_pair.get('anime', '')
                    
                    if question and answer:  # 只插入有效数据
                        batch_data.append((question, answer, character, anime))
                
                # 批量执行插入
                cursor.executemany(insert_sql, batch_data)
                conn.commit()
                
                inserted_count = len(batch_data)
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
            affected_rows = self.db.execute_command(insert_sql, (
                model_name, model_path, base_model, 
                training_data_size, training_config
            ))
            logger.info(f"Inserted model info: {model_name}")
            return affected_rows
        except Exception as e:
            logger.error(f"Failed to insert model info: {e}")
            raise
    
    def set_active_model(self, model_name: str) -> int:
        """Set a model as active (deactivate others)"""
        try:
            # 使用事务处理两个更新操作
            commands = [
                "UPDATE model_info SET is_active = FALSE",
                f"UPDATE model_info SET is_active = TRUE WHERE model_name = '{model_name}'"
            ]
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # 先取消所有模型的激活状态
                cursor.execute("UPDATE model_info SET is_active = FALSE")
                
                # 激活指定模型
                cursor.execute("UPDATE model_info SET is_active = TRUE WHERE model_name = ?", (model_name,))
                updated_count = cursor.rowcount
                
                conn.commit()
            
            if updated_count > 0:
                logger.info(f"Set {model_name} as active model")
            else:
                logger.warning(f"Model {model_name} not found")
            
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