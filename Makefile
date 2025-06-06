# AnimeQA Project Makefile
# Provides convenient commands for development, testing, and training

# ================================
# Configuration
# ================================
PYTHON := python3
PIP := pip3
PROJECT_NAME := llm-finetune
VENV_DIR := venv
LOG_DIR := logs
DATA_DIR := data
MODELS_DIR := models
CACHE_DIR := cache

# Test configuration
TEST_CONFIG := config.test.yaml
export USE_TEST_CONFIG := true

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# ================================
# Help Target
# ================================
.PHONY: help
help: ## Show this help message
	@echo "$(BLUE)AnimeQA Project Makefile$(NC)"
	@echo "$(BLUE)========================$(NC)"
	@echo ""
	@echo "Available commands:"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""

# ================================
# Environment Setup
# ================================
.PHONY: install
install: ## Install all dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)Dependencies installed successfully!$(NC)"

.PHONY: install-dev
install-dev: install ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install pytest pytest-asyncio httpx
	@echo "$(GREEN)Development dependencies installed!$(NC)"

.PHONY: setup-venv
setup-venv: ## Create and setup virtual environment
	@echo "$(BLUE)Creating virtual environment...$(NC)"
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "$(GREEN)Virtual environment created!$(NC)"
	@echo "$(YELLOW)Activate with: source $(VENV_DIR)/bin/activate$(NC)"

.PHONY: clean-cache
clean-cache: ## Clean all cache directories
	@echo "$(BLUE)Cleaning cache directories...$(NC)"
	rm -rf $(CACHE_DIR)
	rm -rf __pycache__
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)Cache cleaned!$(NC)"

# ================================
# Environment Check
# ================================
.PHONY: check-env
check-env: ## Check environment and dependencies
	@echo "$(BLUE)Checking environment...$(NC)"
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PIP) --version)"
	@echo "Current directory: $$(pwd)"
	@echo "Available disk space: $$(df -h . | tail -1 | awk '{print $$4}')"
	@echo "Available memory: $$(free -h | grep '^Mem:' | awk '{print $$7}')"
	@echo "$(GREEN)Environment check completed!$(NC)"

# ================================
# Directory Setup
# ================================
.PHONY: setup-dirs
setup-dirs: ## Create necessary directories
	@echo "$(BLUE)Creating project directories...$(NC)"
	mkdir -p $(LOG_DIR)
	mkdir -p $(DATA_DIR)
	mkdir -p $(MODELS_DIR)
	mkdir -p $(CACHE_DIR)
	mkdir -p data/processed
	mkdir -p models/cache
	mkdir -p cache/datasets
	mkdir -p cache/huggingface
	@echo "$(GREEN)Directories created!$(NC)"

# ================================
# Data Preparation
# ================================
.PHONY: prepare-data
prepare-data: setup-dirs ## Prepare training data
	@echo "$(BLUE)Preparing training data...$(NC)"
	$(PYTHON) scripts/prepare_data.py
	@echo "$(GREEN)Data preparation completed!$(NC)"

.PHONY: prepare-data-sample
prepare-data-sample: setup-dirs ## Prepare sample data for testing
	@echo "$(BLUE)Preparing sample data...$(NC)"
	$(PYTHON) scripts/prepare_data.py --sample
	@echo "$(GREEN)Sample data preparation completed!$(NC)"

.PHONY: reset-data
reset-data: ## Reset database and reprocess data
	@echo "$(YELLOW)Warning: This will delete existing data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
	    echo ""; \
	    echo "$(BLUE)Resetting data...$(NC)"; \
	    $(PYTHON) scripts/prepare_data.py --reset-db --force; \
	    echo "$(GREEN)Data reset completed!$(NC)"; \
	else \
	    echo ""; \
	    echo "$(YELLOW)Data reset cancelled.$(NC)"; \
	fi

# ================================
# Testing Commands
# ================================
.PHONY: test-quick
test-quick: setup-dirs ## Run ultra-fast test with minimal samples (50 train + 10 val)
	@echo "$(BLUE)Running ultra-fast test...$(NC)"
	@echo "$(YELLOW)Only 50 training + 10 validation samples$(NC)"
	@echo "$(YELLOW)Expected completion time: 2-5 minutes$(NC)"
	@echo "Model: microsoft/DialoGPT-small"
	@echo "Configuration: $(TEST_CONFIG)"
	$(PYTHON) quick_test.py
	@echo "$(GREEN)Ultra-fast test completed!$(NC)"

.PHONY: test-data-only
test-data-only: setup-dirs ## Test data preparation only
	@echo "$(BLUE)Testing data preparation...$(NC)"
	$(PYTHON) quick_test.py --data-only
	@echo "$(GREEN)Data preparation test completed!$(NC)"

.PHONY: test-train-only
test-train-only: ## Test training only (assumes data is ready)
	@echo "$(BLUE)Testing training process...$(NC)"
	$(PYTHON) quick_test.py --train-only
	@echo "$(GREEN)Training test completed!$(NC)"

.PHONY: test-inference
test-inference: ## Test model inference
	@echo "$(BLUE)Testing model inference...$(NC)"
	$(PYTHON) scripts/evaluate.py --model-path ./models/anime-qa-model-test --benchmark-only
	@echo "$(GREEN)Inference test completed!$(NC)"

.PHONY: clean-test-data
clean-test-data: ## Clean test data and models
	@echo "$(YELLOW)Cleaning test data...$(NC)"
	rm -rf data/processed_test/
	rm -rf models/anime-qa-model-test/
	rm -rf data/anime_qa_test.db
	@echo "$(GREEN)Test data cleaned!$(NC)"

# ================================
# Training Commands
# ================================
.PHONY: train
train: setup-dirs ## Run full training with default settings
	@echo "$(BLUE)Starting model training...$(NC)"
	@echo "$(YELLOW)This may take a while depending on your hardware$(NC)"
	$(PYTHON) scripts/finetune.py
	@echo "$(GREEN)Training completed!$(NC)"

.PHONY: train-small
train-small: setup-dirs ## Train with small model for testing
	@echo "$(BLUE)Training with small model...$(NC)"
	$(PYTHON) scripts/finetune.py \
	    --model-name microsoft/DialoGPT-small \
	    --batch-size 1 \
	    --num-epochs 1 \
	    --max-length 256 \
	    --lora-rank 8 \
	    --lora-alpha 16 \
	    --learning-rate 1e-4
	@echo "$(GREEN)Small model training completed!$(NC)"

.PHONY: train-medium
train-medium: setup-dirs ## Train with medium model (default)
	@echo "$(BLUE)Training with medium model...$(NC)"
	$(PYTHON) scripts/finetune.py \
	    --model-name microsoft/DialoGPT-medium \
	    --batch-size 4 \
	    --num-epochs 3 \
	    --lora-rank 16 \
	    --lora-alpha 32
	@echo "$(GREEN)Medium model training completed!$(NC)"

.PHONY: train-resume
train-resume: ## Resume training from checkpoint
	@echo "$(BLUE)Resuming training from checkpoint...$(NC)"
	@if [ -z "$(CHECKPOINT)" ]; then \
	    echo "$(RED)Error: Please specify CHECKPOINT path$(NC)"; \
	    echo "Usage: make train-resume CHECKPOINT=./models/checkpoints/checkpoint-1000"; \
	    exit 1; \
	fi
	$(PYTHON) scripts/finetune.py --resume-from-checkpoint $(CHECKPOINT)
	@echo "$(GREEN)Resumed training completed!$(NC)"

# ================================
# Evaluation Commands
# ================================
.PHONY: evaluate
evaluate: ## Evaluate trained model
	@echo "$(BLUE)Evaluating model...$(NC)"
	$(PYTHON) scripts/evaluate.py --model-path ./models/anime-qa-model --dataset
	@echo "$(GREEN)Evaluation completed!$(NC)"

.PHONY: evaluate-test
evaluate-test: ## Evaluate test model
	@echo "$(BLUE)Evaluating test model...$(NC)"
	$(PYTHON) scripts/evaluate.py --model-path ./models/anime-qa-model-test --dataset
	@echo "$(GREEN)Test model evaluation completed!$(NC)"

.PHONY: benchmark
benchmark: ## Run inference benchmark
	@echo "$(BLUE)Running inference benchmark...$(NC)"
	$(PYTHON) scripts/evaluate.py --model-path ./models/anime-qa-model --benchmark-only
	@echo "$(GREEN)Benchmark completed!$(NC)"

.PHONY: compare-models
compare-models: ## Compare base model vs fine-tuned model
	@echo "$(BLUE)Comparing models...$(NC)"
	@echo "Testing base model..."
	$(PYTHON) scripts/evaluate.py --model-path microsoft/DialoGPT-medium --base-model --benchmark-only
	@echo "Testing fine-tuned model..."
	$(PYTHON) scripts/evaluate.py --model-path ./models/anime-qa-model --benchmark-only
	@echo "$(GREEN)Model comparison completed!$(NC)"

# ================================
# Development Workflow
# ================================
.PHONY: dev-setup
dev-setup: setup-dirs install-dev prepare-data-sample ## Complete development setup

.PHONY: dev-cycle
dev-cycle: test-data-only test-quick evaluate-test ## Complete development cycle

.PHONY: full-pipeline
full-pipeline: prepare-data train evaluate ## Run complete ML pipeline

# ================================
# Monitoring and Logs
# ================================
.PHONY: logs
logs: ## Show recent training logs
	@echo "$(BLUE)Recent training logs:$(NC)"
	@if [ -d "$(LOG_DIR)" ]; then \
	    find $(LOG_DIR) -name "*.log" -type f -exec ls -lt {} + | head -10; \
	    echo ""; \
	    echo "$(YELLOW)Use 'tail -f logs/latest.log' to follow logs$(NC)"; \
	else \
	    echo "$(YELLOW)No log directory found$(NC)"; \
	fi

.PHONY: clean-logs
clean-logs: ## Clean old log files
	@echo "$(BLUE)Cleaning old log files...$(NC)"
	@if [ -d "$(LOG_DIR)" ]; then \
	    find $(LOG_DIR) -name "*.log" -type f -mtime +7 -delete; \
	    echo "$(GREEN)Old log files cleaned!$(NC)"; \
	else \
	    echo "$(YELLOW)No log directory found$(NC)"; \
	fi

.PHONY: monitor
monitor: ## Monitor system resources
	@echo "$(BLUE)System resource monitoring:$(NC)"
	@echo "CPU Usage: $$(top -bn1 | grep "Cpu(s)" | awk '{print $$2}' | cut -d'%' -f1)%"
	@echo "Memory: $$(free -h | grep '^Mem:' | awk '{printf "Used: %s/%s (%.1f%%)", $$3, $$2, $$3/$$2*100}')"
	@echo "Disk: $$(df -h . | tail -1 | awk '{printf "Used: %s/%s (%s)", $$3, $$2, $$5}')"
	@if command -v nvidia-smi >/dev/null 2>&1; then \
	    echo "GPU: $$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1 | awk -F', ' '{printf "GPU: %s%%, Memory: %sMB/%sMB", $$1, $$2, $$3}')"; \
	fi

# ================================
# Cleanup Commands
# ================================
.PHONY: clean
clean: clean-cache clean-logs ## Clean cache and logs
	@echo "$(GREEN)General cleanup completed!$(NC)"

.PHONY: clean-models
clean-models: ## Clean downloaded and cached models
	@echo "$(BLUE)Cleaning model cache...$(NC)"
	rm -rf models/cache/
	rm -rf cache/models/
	rm -rf cache/huggingface/
	@echo "$(GREEN)Model cache cleaned!$(NC)"

.PHONY: clean-all
clean-all: clean clean-models clean-test-data ## Clean everything (cache, logs, models)
	@echo "$(GREEN)Complete cleanup finished!$(NC)"

# ================================
# Information Commands
# ================================
.PHONY: status
status: ## Show project status
	@echo "$(BLUE)Project Status:$(NC)"
	@echo "Configuration: $(TEST_CONFIG)"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Working Directory: $$(pwd)"
	@echo ""
	@echo "$(BLUE)Directory Status:$(NC)"
	@echo "Data: $$(if [ -d "$(DATA_DIR)" ]; then echo "✓ Exists"; else echo "✗ Missing"; fi)"
	@echo "Models: $$(if [ -d "$(MODELS_DIR)" ]; then echo "✓ Exists"; else echo "✗ Missing"; fi)"
	@echo "Cache: $$(if [ -d "$(CACHE_DIR)" ]; then echo "✓ Exists"; else echo "✗ Missing"; fi)"
	@echo "Logs: $$(if [ -d "$(LOG_DIR)" ]; then echo "✓ Exists"; else echo "✗ Missing"; fi)"
	@echo ""
	@echo "$(BLUE)File Status:$(NC)"
	@echo "Config: $$(if [ -f "$(TEST_CONFIG)" ]; then echo "✓ Found"; else echo "✗ Missing"; fi)"
	@echo "Requirements: $$(if [ -f "requirements.txt" ]; then echo "✓ Found"; else echo "✗ Missing"; fi)"
	@echo "Quick Test: $$(if [ -f "quick_test.py" ]; then echo "✓ Found"; else echo "✗ Missing"; fi)"

.PHONY: version
version: ## Show version information
	@echo "$(BLUE)Version Information:$(NC)"
	@echo "Project: $(PROJECT_NAME)"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Pip: $$($(PIP) --version)"
	@if $(PYTHON) -c "import torch" 2>/dev/null; then \
	    echo "PyTorch: $$($(PYTHON) -c 'import torch; print(torch.__version__)')"; \
	else \
	    echo "PyTorch: Not installed"; \
	fi
	@if $(PYTHON) -c "import transformers" 2>/dev/null; then \
	    echo "Transformers: $$($(PYTHON) -c 'import transformers; print(transformers.__version__)')"; \
	else \
	    echo "Transformers: Not installed"; \
	fi

# ================================
# Docker Commands (Optional)
# ================================
.PHONY: docker-build
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	@if [ -f "Dockerfile" ]; then \
	    docker build -t $(PROJECT_NAME):latest .; \
	    echo "$(GREEN)Docker image built successfully!$(NC)"; \
	else \
	    echo "$(RED)Dockerfile not found!$(NC)"; \
	    exit 1; \
	fi

.PHONY: docker-run
docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -it --rm -p 8000:8000 $(PROJECT_NAME):latest

# ================================
# Default Target
# ================================
.DEFAULT_GOAL := help

# Make sure some targets are always executed
.PHONY: all clean help status version check-env