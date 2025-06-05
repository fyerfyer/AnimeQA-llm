import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from datasets import load_dataset, Dataset

from config import settings, get_hf_endpoint, ensure_directories

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnimeDataProcessor:
    """Anime character chat data processor"""
    
    def __init__(self):
        """Initialize data processor"""
        self.dataset_name = settings.data.dataset_name
        self.cache_dir = settings.data.cache_dir
        self.processed_path = settings.data.processed_path
        self.hf_endpoint = get_hf_endpoint()
        
        # Ensure directories exist
        ensure_directories()
        
        # Configure HuggingFace endpoint
        if self.hf_endpoint != "https://huggingface.co":
            os.environ['HF_ENDPOINT'] = self.hf_endpoint
            logger.info(f"Using HuggingFace mirror: {self.hf_endpoint}")
    
    def load_raw_dataset(self) -> Dataset:
        """Load raw dataset from HuggingFace"""
        try:
            logger.info(f"Loading dataset: {self.dataset_name}")
            
            dataset = load_dataset(
                self.dataset_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Use train split if available, otherwise use the first available split
            if 'train' in dataset:
                dataset = dataset['train']
            else:
                available_splits = list(dataset.keys())
                logger.info(f"Available splits: {available_splits}")
                dataset = dataset[available_splits[0]]
            
            logger.info(f"Loaded {len(dataset)} samples from dataset")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove extra whitespaces and normalize
        text = text.strip()
        text = ' '.join(text.split())
        
        # Remove special characters that might interfere with training
        # Keep basic punctuation for natural conversation
        import re
        text = re.sub(r'[^\w\s\.\!\?\,\:\;\-\(\)\[\]\"\']+', '', text)
        
        return text
    
    def extract_text_fields(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """Extract relevant text fields from dataset sample"""
        extracted = {}
        
        # Common field names to look for
        text_fields = ['text', 'content', 'message', 'conversation', 'chat']
        character_fields = ['character', 'name', 'speaker', 'character_name']
        anime_fields = ['anime', 'title', 'series', 'source']
        
        # Extract main text content
        for field in text_fields:
            if field in sample and sample[field]:
                extracted['text'] = self.clean_text(str(sample[field]))
                break
        
        # Extract character information
        for field in character_fields:
            if field in sample and sample[field]:
                extracted['character'] = self.clean_text(str(sample[field]))
                break
        
        # Extract anime/series information
        for field in anime_fields:
            if field in sample and sample[field]:
                extracted['anime'] = self.clean_text(str(sample[field]))
                break
        
        # Handle nested structures if needed
        if 'text' not in extracted and isinstance(sample, dict):
            # Look for text in nested structures
            for key, value in sample.items():
                if isinstance(value, dict):
                    nested_text = self.extract_text_fields(value)
                    if 'text' in nested_text:
                        extracted.update(nested_text)
                        break
        
        return extracted
    
    def convert_to_qa_pairs(self, samples: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Convert dataset samples to question-answer pairs"""
        qa_pairs = []
        
        for sample in samples:
            try:
                extracted = self.extract_text_fields(sample)
                
                if not extracted.get('text'):
                    continue
                
                text = extracted['text']
                character = extracted.get('character', '')
                anime = extracted.get('anime', '')
                
                # Generate Q&A pairs based on text content
                generated_pairs = self._generate_qa_from_text(text, character, anime)
                qa_pairs.extend(generated_pairs)
                
            except Exception as e:
                logger.warning(f"Failed to process sample: {e}")
                continue
        
        logger.info(f"Generated {len(qa_pairs)} QA pairs from {len(samples)} samples")
        return qa_pairs
    
    def _generate_qa_from_text(self, text: str, character: str = "", 
                              anime: str = "") -> List[Dict[str, str]]:
        """Generate question-answer pairs from text content"""
        qa_pairs = []
        
        # Simple rule-based Q&A generation
        # This can be enhanced with more sophisticated NLP techniques
        
        # Pattern 1: Character-based questions
        if character:
            qa_pairs.extend([
                {
                    "question": f"Tell me about {character}",
                    "answer": text,
                    "character": character,
                    "anime": anime
                },
                {
                    "question": f"What do you know about {character}?",
                    "answer": text,
                    "character": character,
                    "anime": anime
                },
                {
                    "question": f"Describe {character}",
                    "answer": text,
                    "character": character,
                    "anime": anime
                }
            ])
        
        # Pattern 2: Series-based questions
        if anime:
            qa_pairs.extend([
                {
                    "question": f"What happens in {anime}?",
                    "answer": text,
                    "character": character,
                    "anime": anime
                },
                {
                    "question": f"Tell me about the anime {anime}",
                    "answer": text,
                    "character": character,
                    "anime": anime
                }
            ])
        
        # Pattern 3: General conversation
        qa_pairs.append({
            "question": "Tell me something interesting about anime",
            "answer": text,
            "character": character,
            "anime": anime
        })
        
        return qa_pairs
    
    def filter_and_validate(self, qa_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Filter and validate QA pairs"""
        filtered_pairs = []
        
        for qa_pair in qa_pairs:
            try:
                question = qa_pair.get('question', '').strip()
                answer = qa_pair.get('answer', '').strip()
                
                # Validation rules
                if not question or not answer:
                    continue
                
                if len(question) < 5 or len(answer) < 10:
                    continue
                
                if len(question) > 500 or len(answer) > 1000:
                    continue
                
                # Check for meaningful content
                if question.lower() in ['', 'none', 'null', 'n/a']:
                    continue
                
                if answer.lower() in ['', 'none', 'null', 'n/a']:
                    continue
                
                filtered_pairs.append(qa_pair)
                
            except Exception as e:
                logger.warning(f"Failed to validate QA pair: {e}")
                continue
        
        logger.info(f"Filtered {len(filtered_pairs)} valid QA pairs from {len(qa_pairs)} total")
        return filtered_pairs
    
    def save_processed_data(self, qa_pairs: List[Dict[str, str]], 
                           filename: str = "processed_qa_dataset.json"):
        """Save processed QA pairs to file"""
        try:
            output_dir = Path(self.processed_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / filename
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(qa_pairs)} QA pairs to {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            raise
    
    def load_processed_data(self, filename: str = "processed_qa_dataset.json") -> List[Dict[str, str]]:
        """Load processed QA pairs from file"""
        try:
            input_file = Path(self.processed_path) / filename
            
            if not input_file.exists():
                logger.warning(f"Processed data file not found: {input_file}")
                return []
            
            with open(input_file, 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)
            
            logger.info(f"Loaded {len(qa_pairs)} QA pairs from {input_file}")
            return qa_pairs
            
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
            raise
    
    def get_dataset_statistics(self, qa_pairs: List[Dict[str, str]]) -> Dict[str, Any]:
        """Get statistics about processed dataset"""
        if not qa_pairs:
            return {"total_pairs": 0}
        
        df = pd.DataFrame(qa_pairs)
        
        stats = {
            "total_pairs": len(qa_pairs),
            "unique_characters": df['character'].nunique() if 'character' in df else 0,
            "unique_anime": df['anime'].nunique() if 'anime' in df else 0,
            "avg_question_length": df['question'].str.len().mean() if 'question' in df else 0,
            "avg_answer_length": df['answer'].str.len().mean() if 'answer' in df else 0,
            "top_characters": df['character'].value_counts().head(5).to_dict() if 'character' in df else {},
            "top_anime": df['anime'].value_counts().head(5).to_dict() if 'anime' in df else {}
        }
        
        return stats
    
    def process_full_pipeline(self, save_output: bool = True) -> List[Dict[str, str]]:
        """Run the complete data processing pipeline"""
        try:
            logger.info("Starting full data processing pipeline...")
            
            # Step 1: Load raw dataset
            raw_dataset = self.load_raw_dataset()
            
            # Step 2: Convert to list for processing
            samples = list(raw_dataset)
            logger.info(f"Processing {len(samples)} samples")
            
            # Step 3: Convert to QA pairs
            qa_pairs = self.convert_to_qa_pairs(samples)
            
            # Step 4: Filter and validate
            filtered_pairs = self.filter_and_validate(qa_pairs)
            
            # Step 5: Save processed data
            if save_output and filtered_pairs:
                self.save_processed_data(filtered_pairs)
            
            # Step 6: Generate statistics
            stats = self.get_dataset_statistics(filtered_pairs)
            logger.info(f"Processing completed. Statistics: {stats}")
            
            return filtered_pairs
            
        except Exception as e:
            logger.error(f"Data processing pipeline failed: {e}")
            raise

if __name__ == "__main__":
    # Test data processor
    processor = AnimeDataProcessor()
    
    try:
        # Run processing pipeline
        qa_pairs = processor.process_full_pipeline()
        
        # Display results
        print(f"\nProcessing completed successfully!")
        print(f"Generated {len(qa_pairs)} QA pairs")
        
        if qa_pairs:
            print(f"\nSample QA pair:")
            print(f"Q: {qa_pairs[0]['question']}")
            print(f"A: {qa_pairs[0]['answer'][:200]}...")
        
    except Exception as e:
        print(f"Processing failed: {e}")