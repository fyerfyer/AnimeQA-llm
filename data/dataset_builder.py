import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from datasets import Dataset
from transformers import AutoTokenizer

from config import settings, ensure_directories
from .data_processor import AnimeDataProcessor
from database import QAPairModel, DatabaseConnection

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnimeQADatasetBuilder:
    """Build training datasets for AnimeQA model"""
    
    def __init__(self, data_processor: AnimeDataProcessor = None):
        """Initialize dataset builder"""
        self.data_processor = data_processor or AnimeDataProcessor()
        self.qa_model = QAPairModel()
        self.tokenizer = None
        
        # Configuration
        self.max_length = settings.training.max_length
        self.processed_path = settings.data.processed_path
        
        ensure_directories()
    
    def load_tokenizer(self, model_name: str = None):
        """Load tokenizer for text processing"""
        try:
            model_name = model_name or settings.model.base_model_name
            logger.info(f"Loading tokenizer: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=settings.model.cache_dir,
                trust_remote_code=True
            )
            
            # Add special tokens if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def get_qa_pairs_from_database(self) -> List[Dict[str, str]]:
        """Get QA pairs from database"""
        try:
            qa_pairs = self.qa_model.get_qa_pairs()
            logger.info(f"Loaded {len(qa_pairs)} QA pairs from database")
            return qa_pairs
            
        except Exception as e:
            logger.warning(f"Failed to load QA pairs from database: {e}")
            return []
    
    def get_qa_pairs_from_file(self, filename: str = "processed_qa_dataset.json") -> List[Dict[str, str]]:
        """Get QA pairs from processed file"""
        try:
            qa_pairs = self.data_processor.load_processed_data(filename)
            return qa_pairs
            
        except Exception as e:
            logger.warning(f"Failed to load QA pairs from file: {e}")
            return []
    
    def prepare_training_data(self, qa_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Prepare data for conversational training"""
        training_data = []
        
        for qa_pair in qa_pairs:
            try:
                question = qa_pair.get('question', '').strip()
                answer = qa_pair.get('answer', '').strip()
                
                if not question or not answer:
                    continue
                
                # Format for conversational training
                # Using the format expected by dialog models
                conversation = {
                    'input_text': question,
                    'target_text': answer,
                    'character': qa_pair.get('character', ''),
                    'anime': qa_pair.get('anime', '')
                }
                
                training_data.append(conversation)
                
            except Exception as e:
                logger.warning(f"Failed to prepare training sample: {e}")
                continue
        
        logger.info(f"Prepared {len(training_data)} training samples")
        return training_data
    
    def tokenize_conversations(self, conversations: List[Dict[str, str]]) -> Dict[str, List]:
        """Tokenize conversations for training"""
        if not self.tokenizer:
            self.load_tokenizer()
        
        tokenized_data = {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }
        
        for conv in conversations:
            try:
                input_text = conv['input_text']
                target_text = conv['target_text']
                
                # Tokenize input
                input_encoding = self.tokenizer(
                    input_text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length // 2,  # Reserve space for target
                    return_tensors='pt'
                )
                
                # Tokenize target
                target_encoding = self.tokenizer(
                    target_text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length // 2,
                    return_tensors='pt'
                )
                
                # Combine for autoregressive training
                combined_input_ids = []
                combined_attention_mask = []
                labels = []
                
                # Add input tokens
                input_ids = input_encoding['input_ids'].squeeze().tolist()
                attention_mask = input_encoding['attention_mask'].squeeze().tolist()
                
                combined_input_ids.extend(input_ids)
                combined_attention_mask.extend(attention_mask)
                labels.extend([-100] * len(input_ids))  # Ignore input in loss
                
                # Add target tokens
                target_ids = target_encoding['input_ids'].squeeze().tolist()
                target_attention = target_encoding['attention_mask'].squeeze().tolist()
                
                combined_input_ids.extend(target_ids)
                combined_attention_mask.extend(target_attention)
                labels.extend(target_ids)  # Use target for loss calculation
                
                # Truncate if too long
                if len(combined_input_ids) > self.max_length:
                    combined_input_ids = combined_input_ids[:self.max_length]
                    combined_attention_mask = combined_attention_mask[:self.max_length]
                    labels = labels[:self.max_length]
                
                # Pad if too short
                while len(combined_input_ids) < self.max_length:
                    combined_input_ids.append(self.tokenizer.pad_token_id)
                    combined_attention_mask.append(0)
                    labels.append(-100)
                
                tokenized_data['input_ids'].append(combined_input_ids)
                tokenized_data['attention_mask'].append(combined_attention_mask)
                tokenized_data['labels'].append(labels)
                
            except Exception as e:
                logger.warning(f"Failed to tokenize conversation: {e}")
                continue
        
        logger.info(f"Tokenized {len(tokenized_data['input_ids'])} conversations")
        return tokenized_data
    
    def create_huggingface_dataset(self, tokenized_data: Dict[str, List]) -> Dataset:
        """Create HuggingFace Dataset object"""
        try:
            dataset = Dataset.from_dict(tokenized_data)
            logger.info(f"Created HuggingFace dataset with {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to create HuggingFace dataset: {e}")
            raise
    
    def split_dataset(self, dataset: Dataset, 
                     train_ratio: float = 0.8) -> Tuple[Dataset, Dataset]:
        """Split dataset into train and validation sets"""
        try:
            dataset_size = len(dataset)
            train_size = int(dataset_size * train_ratio)
            
            # Shuffle and split
            shuffled_dataset = dataset.shuffle(seed=42)
            train_dataset = shuffled_dataset.select(range(train_size))
            val_dataset = shuffled_dataset.select(range(train_size, dataset_size))
            
            logger.info(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} validation")
            return train_dataset, val_dataset
            
        except Exception as e:
            logger.error(f"Failed to split dataset: {e}")
            raise
    
    def save_dataset(self, dataset: Dataset, filename: str):
        """Save dataset to file"""
        try:
            output_dir = Path(self.processed_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / filename
            dataset.save_to_disk(str(output_path))
            
            logger.info(f"Saved dataset to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            raise
    
    def load_dataset(self, filename: str) -> Dataset:
        """Load dataset from file"""
        try:
            dataset_path = Path(self.processed_path) / filename
            
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            
            dataset = Dataset.load_from_disk(str(dataset_path))
            logger.info(f"Loaded dataset with {len(dataset)} samples from {dataset_path}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def get_dataset_info(self, dataset: Dataset) -> Dict[str, Any]:
        """Get information about the dataset"""
        try:
            info = {
                'num_samples': len(dataset),
                'features': list(dataset.features.keys()),
                'dataset_size': dataset.data.nbytes if hasattr(dataset.data, 'nbytes') else 'unknown'
            }
            
            # Sample statistics
            if len(dataset) > 0:
                sample = dataset[0]
                if 'input_ids' in sample:
                    info['avg_sequence_length'] = len(sample['input_ids'])
                    info['vocab_size'] = max(sample['input_ids']) + 1 if sample['input_ids'] else 0
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get dataset info: {e}")
            return {'error': str(e)}
    
    def build_training_dataset(self, use_database: bool = True, 
                              use_processed_file: bool = True,
                              save_datasets: bool = True) -> Tuple[Dataset, Dataset]:
        """Build complete training dataset"""
        try:
            logger.info("Starting training dataset building...")
            
            # Collect QA pairs from different sources
            all_qa_pairs = []
            
            # From database
            if use_database:
                db_pairs = self.get_qa_pairs_from_database()
                all_qa_pairs.extend(db_pairs)
                logger.info(f"Added {len(db_pairs)} pairs from database")
            
            # From processed file
            if use_processed_file:
                file_pairs = self.get_qa_pairs_from_file()
                all_qa_pairs.extend(file_pairs)
                logger.info(f"Added {len(file_pairs)} pairs from processed file")
            
            # If no data available, process raw data
            if not all_qa_pairs:
                logger.info("No existing data found, processing raw dataset...")
                all_qa_pairs = self.data_processor.process_full_pipeline()
            
            if not all_qa_pairs:
                raise ValueError("No QA pairs available for training")
            
            # Remove duplicates
            unique_pairs = []
            seen_questions = set()
            for pair in all_qa_pairs:
                question = pair.get('question', '').strip().lower()
                if question and question not in seen_questions:
                    unique_pairs.append(pair)
                    seen_questions.add(question)
            
            logger.info(f"Removed duplicates: {len(unique_pairs)} unique pairs")
            
            # Prepare training data
            training_data = self.prepare_training_data(unique_pairs)
            
            if not training_data:
                raise ValueError("No valid training data prepared")
            
            # Tokenize conversations
            tokenized_data = self.tokenize_conversations(training_data)
            
            # Create HuggingFace dataset
            dataset = self.create_huggingface_dataset(tokenized_data)
            
            # Split into train/validation
            train_dataset, val_dataset = self.split_dataset(dataset)
            
            # Save datasets
            if save_datasets:
                self.save_dataset(train_dataset, "train_dataset")
                self.save_dataset(val_dataset, "val_dataset")
            
            # Log dataset information
            train_info = self.get_dataset_info(train_dataset)
            val_info = self.get_dataset_info(val_dataset)
            
            logger.info(f"Training dataset built successfully!")
            logger.info(f"Train samples: {train_info['num_samples']}")
            logger.info(f"Validation samples: {val_info['num_samples']}")
            
            return train_dataset, val_dataset
            
        except Exception as e:
            logger.error(f"Failed to build training dataset: {e}")
            raise

if __name__ == "__main__":
    # Test dataset builder
    builder = AnimeQADatasetBuilder()
    
    try:
        # Build training datasets
        train_dataset, val_dataset = builder.build_training_dataset()
        
        print(f"\nDataset building completed successfully!")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Display sample
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"\nSample training data shape:")
            for key, value in sample.items():
                if isinstance(value, list):
                    print(f"{key}: length {len(value)}")
                else:
                    print(f"{key}: {type(value)}")
        
    except Exception as e:
        print(f"Dataset building failed: {e}")