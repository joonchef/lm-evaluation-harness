# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Tasks

### Installation
```bash
# Basic installation
pip install -e .

# Development installation with common extras
pip install -e ".[dev,sentencepiece,api]"

# For specific model backends
pip install -e ".[vllm]"  # vLLM support
pip install -e ".[mamba]"  # Mamba SSM models
pip install -e ".[optimum]"  # Intel OpenVINO models
```

### Running Evaluations
```bash
# Basic evaluation
lm_eval --model hf \
    --model_args pretrained=<model_name> \
    --tasks <task_name> \
    --device cuda:0 \
    --batch_size 8

# List available tasks
lm_eval --tasks list

# Run with automatic batch size detection
lm_eval --model hf \
    --model_args pretrained=<model_name> \
    --tasks <task_name> \
    --batch_size auto
```

### Testing
```bash
# Run all tests
python -m pytest

# Run tests with coverage
python -m pytest --cov

# Run tests in parallel
python -m pytest -n=auto

# Run specific test file
python -m pytest tests/test_evaluator.py
```

### Linting and Code Quality
```bash
# Run pre-commit checks
pre-commit run --all-files

# Ruff linting (configured in pyproject.toml)
ruff check .
ruff format .
```

## Architecture Overview

### Core Components

**lm_eval/evaluator.py**
Central evaluation orchestrator that coordinates model loading, task execution, and result aggregation. The `simple_evaluate()` function is the main entry point for programmatic evaluation.

**lm_eval/models/**
Model implementations for different backends (HuggingFace, vLLM, OpenAI API, etc.). Each model type inherits from the base `LM` class and implements request handling for different evaluation types (generate_until, loglikelihood, etc.).

**lm_eval/tasks/**
Task definitions and configurations. Tasks are defined in YAML files with templates, metrics, and evaluation settings. The TaskManager handles loading and instantiation of tasks.

**lm_eval/api/**
Core abstractions including the base `LM` model interface, `Task` interface, metrics, and the registry system for dynamic component loading.

**lm_eval/__main__.py**
CLI interface that parses arguments and calls the evaluator. Supports both `lm_eval` and `lm-eval` commands.

### Key Design Patterns

- **Request Batching**: Models process evaluation requests in batches for efficiency. Batch size can be set manually or detected automatically.
- **Task Configuration**: Tasks are defined declaratively in YAML with Jinja2 templating for prompts.
- **Registry System**: Dynamic registration of models, tasks, and metrics allows for extensibility.
- **Caching**: Results can be cached to disk to resume interrupted evaluations.

### Adding New Components

**New Tasks**: Create YAML configuration in `lm_eval/tasks/` following existing patterns. See docs/new_task_guide.md for details.

**New Models**: Implement the `LM` interface from `lm_eval/api/model.py`. Handle the request types your model supports.

**New Metrics**: Add to `lm_eval/api/metrics.py` and register in the metrics registry.

## Important Notes

- The library supports both local models (via HuggingFace, vLLM, etc.) and API-based models (OpenAI, Anthropic, etc.)
- Multi-GPU evaluation is supported through data parallelism (accelerate) or model parallelism (device_map)
- Task configurations use Jinja2 templating for flexible prompt construction
- Results are deterministic given the same seed values (random_seed, numpy_random_seed, torch_random_seed, fewshot_random_seed)

## Language and Communication Guidelines

- 코드 및 전문용어, 대명사 등을 제외한 언어는 한국어를 사용
- 커밋 메시지는 한국어로 작성하고, CLAUDE에 관한 정보는 제외