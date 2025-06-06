import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from models.model_loader import ModelLoader
from data import AnimeQADatasetBuilder
from utils.helpers import setup_logger, format_time, get_memory_usage, save_json

def setup_logging(log_level: str = "INFO"):
    """Setup logging for evaluation script"""
    log_dir = Path(project_root) / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"evaluate_{timestamp}.log"
    
    logger = setup_logger(
        name="evaluate",
        log_file=str(log_file),
        level=log_level
    )
    
    return logger

class ModelEvaluator:
    """Model evaluation class"""
    
    def __init__(self, model_path: str):
        """Initialize evaluator with model path"""
        self.model_path = model_path
        self.model_loader = ModelLoader()
        self.model = None
        self.tokenizer = None
        self.logger = logging.getLogger("evaluate")
        
    def load_model(self):
        """Load model and tokenizer"""
        try:
            self.logger.info(f"Loading model from: {self.model_path}")
            
            if Path(self.model_path).exists():
                # Load fine-tuned model
                self.model, self.tokenizer = self.model_loader.load_finetuned_model(self.model_path)
            else:
                # Load base model
                self.tokenizer = self.model_loader.load_tokenizer(self.model_path)
                self.model = self.model_loader.load_base_model(self.model_path)
            
            # Set to evaluation mode
            self.model_loader.set_eval_mode()
            
            model_info = self.model_loader.get_model_info()
            self.logger.info(f"Model loaded successfully:")
            self.logger.info(f"  Type: {model_info['model_type']}")
            self.logger.info(f"  Parameters: {model_info['total_params']:,}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_response(self, question: str, max_length: int = 128, 
                         temperature: float = 0.7, top_p: float = 0.9) -> Dict[str, Any]:
        """Generate response for a question"""
        try:
            start_time = time.time()
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                question,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][len(inputs[0]):], 
                skip_special_tokens=True
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            return {
                "question": question,
                "answer": response.strip(),
                "response_time": response_time,
                "input_tokens": len(inputs[0]),
                "output_tokens": len(outputs[0]) - len(inputs[0])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}")
            return {
                "question": question,
                "answer": "",
                "error": str(e),
                "response_time": 0,
                "input_tokens": 0,
                "output_tokens": 0
            }
    
    def evaluate_on_dataset(self, dataset, max_samples: int = None) -> Dict[str, Any]:
        """Evaluate model on dataset"""
        try:
            self.logger.info("Starting dataset evaluation...")
            
            if max_samples and len(dataset) > max_samples:
                # Sample subset for evaluation
                indices = list(range(0, len(dataset), len(dataset) // max_samples))[:max_samples]
                dataset_subset = dataset.select(indices)
            else:
                dataset_subset = dataset
            
            results = []
            total_samples = len(dataset_subset)
            
            for i, sample in enumerate(dataset_subset):
                if i % 10 == 0:
                    self.logger.info(f"Evaluating sample {i+1}/{total_samples}")
                
                # Extract question from sample
                if 'input_ids' in sample:
                    # Decode tokenized question
                    question = self.tokenizer.decode(
                        sample['input_ids'][:len(sample['input_ids'])//2], 
                        skip_special_tokens=True
                    )
                else:
                    question = sample.get('question', 'Tell me about anime')
                
                # Generate response
                result = self.generate_response(question)
                results.append(result)
            
            # Calculate metrics
            metrics = self.calculate_metrics(results)
            
            self.logger.info("Dataset evaluation completed")
            return {
                "total_samples": total_samples,
                "results": results,
                "metrics": metrics
            }
            
        except Exception as e:
            self.logger.error(f"Dataset evaluation failed: {e}")
            raise
    
    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        try:
            valid_results = [r for r in results if not r.get('error')]
            
            if not valid_results:
                return {"error": "No valid results"}
            
            # Response time metrics
            response_times = [r['response_time'] for r in valid_results]
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
            
            # Token metrics
            input_tokens = [r['input_tokens'] for r in valid_results]
            output_tokens = [r['output_tokens'] for r in valid_results]
            
            avg_input_tokens = sum(input_tokens) / len(input_tokens)
            avg_output_tokens = sum(output_tokens) / len(output_tokens)
            
            # Answer quality metrics (simple heuristics)
            answer_lengths = [len(r['answer']) for r in valid_results]
            avg_answer_length = sum(answer_lengths) / len(answer_lengths)
            
            # Non-empty answer ratio
            non_empty_answers = sum(1 for r in valid_results if r['answer'].strip())
            non_empty_ratio = non_empty_answers / len(valid_results)
            
            metrics = {
                "total_samples": len(results),
                "valid_samples": len(valid_results),
                "success_rate": len(valid_results) / len(results),
                "avg_response_time": avg_response_time,
                "min_response_time": min_response_time,
                "max_response_time": max_response_time,
                "avg_input_tokens": avg_input_tokens,
                "avg_output_tokens": avg_output_tokens,
                "avg_answer_length": avg_answer_length,
                "non_empty_answer_ratio": non_empty_ratio,
                "throughput": len(valid_results) / sum(response_times) if response_times else 0
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Metrics calculation failed: {e}")
            return {"error": str(e)}
    
    def benchmark_inference(self, questions: List[str], num_runs: int = 3) -> Dict[str, Any]:
        """Benchmark inference performance"""
        try:
            self.logger.info(f"Running inference benchmark with {len(questions)} questions, {num_runs} runs each")
            
            all_results = []
            
            for run in range(num_runs):
                self.logger.info(f"Benchmark run {run + 1}/{num_runs}")
                run_results = []
                
                for question in questions:
                    result = self.generate_response(question)
                    run_results.append(result)
                
                all_results.extend(run_results)
            
            # Calculate benchmark metrics
            metrics = self.calculate_metrics(all_results)
            
            # Add benchmark-specific metrics
            response_times = [r['response_time'] for r in all_results if not r.get('error')]
            if response_times:
                import statistics
                metrics.update({
                    "median_response_time": statistics.median(response_times),
                    "p95_response_time": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times),
                    "p99_response_time": statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else max(response_times)
                })
            
            return {
                "benchmark_config": {
                    "num_questions": len(questions),
                    "num_runs": num_runs,
                    "total_inferences": len(all_results)
                },
                "questions": questions,
                "results": all_results,
                "metrics": metrics
            }
            
        except Exception as e:
            self.logger.error(f"Inference benchmark failed: {e}")
            raise

def get_test_questions() -> List[str]:
    """Get predefined test questions"""
    return [
        "Who is Naruto Uzumaki?",
        "What is Naruto's favorite food?",
        "Who is Sasuke Uchiha?",
        "What is the Sharingan?",
        "Who is Monkey D. Luffy?",
        "What is the One Piece?",
        "Tell me about Dragon Ball Z",
        "Who is the strongest anime character?",
        "What is your favorite anime?",
        "Explain the concept of chakra in Naruto"
    ]

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(
        description="Evaluate AnimeQA model performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate fine-tuned model
  python scripts/evaluate.py --model-path ./models/anime-qa-model
  
  # Evaluate base model for comparison
  python scripts/evaluate.py --model-path microsoft/DialoGPT-medium --base-model
  
  # Run inference benchmark
  python scripts/evaluate.py --model-path ./models/anime-qa-model --benchmark-only
  
  # Evaluate on dataset with limited samples
  python scripts/evaluate.py --model-path ./models/anime-qa-model --dataset --max-samples 50
        """
    )
    
    # Model configuration
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to model directory or model name'
    )
    
    parser.add_argument(
        '--base-model',
        action='store_true',
        help='Evaluate base model (not fine-tuned)'
    )
    
    # Evaluation options
    parser.add_argument(
        '--dataset',
        action='store_true',
        help='Evaluate on validation dataset'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=100,
        help='Maximum samples for dataset evaluation'
    )
    
    parser.add_argument(
        '--benchmark-only',
        action='store_true',
        help='Run only inference benchmark'
    )
    
    parser.add_argument(
        '--benchmark-runs',
        type=int,
        default=3,
        help='Number of benchmark runs'
    )
    
    # Generation parameters
    parser.add_argument(
        '--max-length',
        type=int,
        default=128,
        help='Maximum generation length'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Generation temperature'
    )
    
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.9,
        help='Top-p sampling parameter'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    try:
        logger.info("Starting model evaluation...")
        start_time = datetime.now()
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize evaluator
        evaluator = ModelEvaluator(args.model_path)
        evaluator.load_model()
        
        # Log system info
        memory_info = get_memory_usage()
        logger.info(f"System memory: {memory_info}")
        
        evaluation_results = {}
        
        # Run inference benchmark
        test_questions = get_test_questions()
        logger.info("Running inference benchmark...")
        
        benchmark_results = evaluator.benchmark_inference(
            test_questions, 
            num_runs=args.benchmark_runs
        )
        evaluation_results['benchmark'] = benchmark_results
        
        # Dataset evaluation (if requested and not benchmark-only)
        if args.dataset and not args.benchmark_only:
            try:
                logger.info("Loading validation dataset...")
                dataset_builder = AnimeQADatasetBuilder()
                val_dataset = dataset_builder.load_dataset("val_dataset")
                
                logger.info("Running dataset evaluation...")
                dataset_results = evaluator.evaluate_on_dataset(
                    val_dataset, 
                    max_samples=args.max_samples
                )
                evaluation_results['dataset'] = dataset_results
                
            except Exception as e:
                logger.warning(f"Dataset evaluation failed: {e}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"evaluation_results_{timestamp}.json"
        
        evaluation_results.update({
            "model_path": args.model_path,
            "evaluation_time": datetime.now().isoformat(),
            "generation_params": {
                "max_length": args.max_length,
                "temperature": args.temperature,
                "top_p": args.top_p
            }
        })
        
        save_json(evaluation_results, results_file)
        
        # Print summary
        end_time = datetime.now()
        total_time = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total time: {format_time(total_time.total_seconds())}")
        logger.info(f"Results saved to: {results_file}")
        
        # Print benchmark metrics
        if 'benchmark' in evaluation_results:
            metrics = evaluation_results['benchmark']['metrics']
            logger.info("Benchmark Results:")
            logger.info(f"  Success Rate: {metrics.get('success_rate', 0):.2%}")
            logger.info(f"  Avg Response Time: {metrics.get('avg_response_time', 0):.3f}s")
            logger.info(f"  Throughput: {metrics.get('throughput', 0):.2f} req/s")
            logger.info(f"  Avg Answer Length: {metrics.get('avg_answer_length', 0):.1f} chars")
        
        # Print dataset metrics
        if 'dataset' in evaluation_results:
            metrics = evaluation_results['dataset']['metrics']
            logger.info("Dataset Results:")
            logger.info(f"  Samples Evaluated: {metrics.get('valid_samples', 0)}")
            logger.info(f"  Success Rate: {metrics.get('success_rate', 0):.2%}")
            logger.info(f"  Non-empty Answers: {metrics.get('non_empty_answer_ratio', 0):.2%}")
        
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETED!")
        print(f"{'='*60}")
        print(f"Results saved to: {results_file}")
        print(f"Check logs for detailed metrics")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    import torch  # Import here to avoid issues if PyTorch not available
    exit(main())