# Makefile for NeMo Fine-Tuning Pipeline
# Provides convenient commands for Docker-based training

.PHONY: help setup build run shell jupyter tensorboard logs stop clean validate

# Default target
help:
	@echo "NeMo Fine-Tuning Pipeline - Docker Commands"
	@echo "==========================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup          Setup Docker environment (one-time)"
	@echo "  build          Build Docker image"
	@echo ""
	@echo "Training Commands:"
	@echo "  run            Run S32K144 fine-tuning (default: LLaMA2 7B)"
	@echo "  run-llama3     Run with LLaMA3 8B model"
	@echo "  run-codellama  Run with CodeLlama 7B model"
	@echo "  run-extended   Run extended training (10 epochs)"
	@echo ""
	@echo "Development Commands:"
	@echo "  shell          Start interactive shell in container"
	@echo "  jupyter        Start Jupyter Lab (http://localhost:8888)"
	@echo "  tensorboard    Start TensorBoard (http://localhost:6006)"
	@echo ""
	@echo "Monitoring Commands:"
	@echo "  logs           Show training logs"
	@echo "  status         Show container status"
	@echo "  validate       Validate training data"
	@echo ""
	@echo "Management Commands:"
	@echo "  stop           Stop all containers"
	@echo "  clean          Clean up containers and images"
	@echo "  outputs        Show output directory contents"
	@echo ""
	@echo "Examples:"
	@echo "  make setup && make run    # First-time setup and training"
	@echo "  make run-llama3          # Train LLaMA3 model"
	@echo "  make logs                # Monitor training progress"

# Setup Docker environment
setup:
	@echo "Setting up Docker environment..."
	chmod +x scripts/setup_docker_environment.sh
	./scripts/setup_docker_environment.sh

# Build Docker image
build:
	@echo "Building Docker image..."
	chmod +x scripts/docker_run.sh
	./scripts/docker_run.sh build

# Run default training (LLaMA2 7B)
run:
	@echo "Starting S32K144 fine-tuning with LLaMA2 7B..."
	chmod +x scripts/docker_run.sh
	./scripts/docker_run.sh run

# Run LLaMA3 training
run-llama3:
	@echo "Starting S32K144 fine-tuning with LLaMA3 8B..."
	chmod +x scripts/docker_run.sh
	./scripts/docker_run.sh --model-type llama3 --model-size 8b run

# Run CodeLlama training
run-codellama:
	@echo "Starting S32K144 fine-tuning with CodeLlama 7B..."
	chmod +x scripts/docker_run.sh
	./scripts/docker_run.sh --model-type codellama --model-size 7b run

# Run extended training
run-extended:
	@echo "Starting extended S32K144 fine-tuning (10 epochs)..."
	chmod +x scripts/docker_run.sh
	./scripts/docker_run.sh --epochs 10 run

# Run multi-GPU training
run-multi-gpu:
	@echo "Starting multi-GPU S32K144 fine-tuning..."
	chmod +x scripts/docker_run.sh
	./scripts/docker_run.sh --gpus 2 --batch-size 4 run

# Start interactive shell
shell:
	@echo "Starting interactive shell..."
	chmod +x scripts/docker_run.sh
	./scripts/docker_run.sh shell

# Start Jupyter Lab
jupyter:
	@echo "Starting Jupyter Lab..."
	chmod +x scripts/docker_run.sh
	./scripts/docker_run.sh jupyter
	@echo "Access Jupyter Lab at: http://localhost:8888"

# Start TensorBoard
tensorboard:
	@echo "Starting TensorBoard..."
	chmod +x scripts/docker_run.sh
	./scripts/docker_run.sh tensorboard
	@echo "Access TensorBoard at: http://localhost:6006"

# Show training logs
logs:
	@echo "Showing training logs..."
	chmod +x scripts/docker_run.sh
	./scripts/docker_run.sh logs

# Show container status
status:
	@echo "Container Status:"
	@echo "=================="
	@docker ps -a --filter "name=nemo-finetune" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	@echo "GPU Status:"
	@echo "==========="
	@nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "NVIDIA GPU not available"

# Validate training data
validate:
	@echo "Validating S32K144 training data..."
	@if [ -f "validate_s32k144_data.py" ]; then \
		python validate_s32k144_data.py; \
	else \
		echo "Validation script not found. Please ensure validate_s32k144_data.py exists."; \
	fi

# Stop containers
stop:
	@echo "Stopping containers..."
	chmod +x scripts/docker_run.sh
	./scripts/docker_run.sh stop

# Clean up containers and images
clean:
	@echo "Cleaning up containers and images..."
	chmod +x scripts/docker_run.sh
	./scripts/docker_run.sh clean

# Show output directory contents
outputs:
	@echo "Output Directory Contents:"
	@echo "=========================="
	@echo "Checkpoints:"
	@ls -la outputs/checkpoints/ 2>/dev/null || echo "  No checkpoints found"
	@echo ""
	@echo "Deployed Models:"
	@ls -la outputs/deploy/ 2>/dev/null || echo "  No deployed models found"
	@echo ""
	@echo "Logs:"
	@ls -la outputs/logs/ 2>/dev/null || echo "  No logs found"
	@echo ""
	@echo "Cache:"
	@du -sh outputs/cache/ 2>/dev/null || echo "  No cache found"

# Development targets
dev-setup: setup build
	@echo "Development environment ready!"

dev-run: validate run
	@echo "Development training started!"

# Production targets
prod-build:
	@echo "Building production Docker image..."
	docker-compose -f docker/docker-compose.prod.yml build

prod-run:
	@echo "Starting production training..."
	docker-compose -f docker/docker-compose.prod.yml up -d

prod-logs:
	@echo "Showing production logs..."
	docker-compose -f docker/docker-compose.prod.yml logs -f

prod-stop:
	@echo "Stopping production services..."
	docker-compose -f docker/docker-compose.prod.yml down

# Quick start for new users
quickstart: setup build validate run
	@echo ""
	@echo "🎉 Quick start completed!"
	@echo "========================"
	@echo "Training has started. Monitor progress with:"
	@echo "  make logs"
	@echo ""
	@echo "Access development tools:"
	@echo "  make jupyter     # Jupyter Lab at http://localhost:8888"
	@echo "  make tensorboard # TensorBoard at http://localhost:6006"
	@echo ""
	@echo "Check outputs:"
	@echo "  make outputs"

# Help for specific commands
help-docker:
	@echo "Docker-specific help:"
	@echo "===================="
	@echo "The pipeline uses Docker for:"
	@echo "- Consistent environment across different systems"
	@echo "- GPU support with NVIDIA Docker"
	@echo "- Persistent model outputs"
	@echo "- Integrated development tools"
	@echo ""
	@echo "Volume mounts:"
	@echo "  ./data -> /workspace/data (training data)"
	@echo "  ./outputs/checkpoints -> /workspace/checkpoints"
	@echo "  ./outputs/logs -> /workspace/logs"
	@echo "  ./outputs/deploy -> /workspace/deploy"
	@echo ""
	@echo "For detailed documentation: docker/README.md"

help-training:
	@echo "Training-specific help:"
	@echo "======================"
	@echo "Available models:"
	@echo "  LLaMA2: General instruction-following (7B, 13B, 70B)"
	@echo "  LLaMA3: Enhanced reasoning (8B, 70B)"
	@echo "  CodeLlama: Code generation (7B, 13B, 34B)"
	@echo ""
	@echo "Training data:"
	@echo "  data/train.yaml - 8 S32K144 training examples"
	@echo "  data/val.yaml   - 4 validation examples"
	@echo "  data/test.yaml  - 4 test examples"
	@echo ""
	@echo "Outputs:"
	@echo "  outputs/checkpoints/ - Model checkpoints"
	@echo "  outputs/deploy/      - Ready-to-use models"
	@echo "  outputs/logs/        - Training logs"
