# vis_ifeval

Visual Instruction Following Evaluation Benchmark

## Overview

`vis_ifeval` is a Python benchmark for evaluating visual instruction following capabilities of image generation models. The benchmark tests whether generated images satisfy dense prompts with multiple constraints (8-12 per prompt).

### Key Concepts

- **Prompts**: Dense text descriptions with multiple constraints
- **Constraints**: Specific requirements that must be satisfied (e.g., object counts, text content, spatial relationships)
- **VIPR (Visual Instruction Pass Rate)**: Primary metric measuring the percentage of constraints satisfied

## Current Status

### Implemented

- ✅ Multiple image models:
  - DummyModel (random images for testing)
  - SDXLModel (Stable Diffusion XL)
  - SD3Model (Stable Diffusion 3)
  - FluxModel (FLUX.1-dev)
  - OpenAIModel (DALL-E 3/2)
- ✅ Text evaluator using Tesseract OCR with backend abstraction
- ✅ Nutrition label evaluator (table_slot constraints) with OCR parsing
- ✅ Logic evaluator (percent_dv_consistency for sodium)
- ✅ CLIP-based negative evaluator (checks for forbidden concepts)
- ✅ CLIP-based composition evaluator (count, attribute, state)
- ✅ Spatial evaluator (stub, ready for GroundingDINO)
- ✅ OCR backend abstraction (Tesseract + placeholder for advanced backends)
- ✅ End-to-end pipeline (generate → evaluate → aggregate)
- ✅ Automated multi-model evaluation script
- ✅ Results export utilities (tables, plots)
- ✅ Enhanced W&B integration with visualization tables
- ✅ Expanded prompt suite (23 prompts, 143 constraints)

### Partially Implemented / Stubs

- 🔲 Spatial evaluator (structure ready, needs GroundingDINO integration)
- 🔲 Advanced OCR backends (Surya, DeepSeek-OCR) - placeholder ready

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR (required for text evaluation):

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

### Basic Pipeline

Run the full evaluation pipeline:

```bash
# 1. Generate images
PYTHONPATH=src python -m vis_ifeval.runners.generate_images --model-name dummy

# 2. Evaluate constraints
PYTHONPATH=src python -m vis_ifeval.runners.evaluate_constraints --model-name dummy

# 3. Aggregate metrics
PYTHONPATH=src python -m vis_ifeval.runners.aggregate_metrics --model-name dummy
```

### Multi-Model Evaluation

Run evaluation for multiple models automatically:

```bash
PYTHONPATH=src python scripts/run_all_models.py --models dummy,sdxl,sd3
```

### Export Results

Generate tables and plots for papers:

```bash
PYTHONPATH=src python -m vis_ifeval.utils.export_results --models dummy,sdxl
```

This creates:
- `results/tables/model_vipr_table.md` - Markdown table
- `results/tables/model_vipr_table.csv` - CSV table
- `results/plots/vipr_by_model.png` - Bar chart
- `results/plots/vipr_by_category.png` - Category comparison
- `results/plots/vipr_by_type.png` - Type comparison

### Command Line Options

All runners support `--model-name` and `--use-wandb` flags:

```bash
# Generate with custom model
python -m vis_ifeval.runners.generate_images --model-name dummy

# Enable W&B logging
python -m vis_ifeval.runners.generate_images --use-wandb

# Evaluate with W&B
python -m vis_ifeval.runners.evaluate_constraints --model-name dummy --use-wandb
```

### Weights & Biases Integration

Enable W&B logging to track experiments and visualize results:

1. **Set environment variables:**

```bash
export VIS_IFEVAL_USE_WANDB=1
export VIS_IFEVAL_WANDB_PROJECT=vis-ifeval
export VIS_IFEVAL_WANDB_ENTITY=your-username  # optional
export VIS_IFEVAL_WANDB_GROUP=experiment-name  # optional
```

2. **Or use command line flags:**

```bash
python -m vis_ifeval.runners.generate_images --use-wandb
```

3. **What gets logged:**

   - **Generation step**: Per-image latency, model name, category, prompt ID, sample images
   - **Evaluation step**: Per-constraint scores and labels, constraint types, sample images with scores
   - **Aggregation step**: VIPR metrics, VIPR by type, VIPR by category, latency statistics

4. **Available dashboards:**

   - VIPR by constraint type (bar chart)
   - VIPR by category (bar chart)
   - Latency distribution (histogram)
   - Score distributions by constraint type (histograms/boxplots)
   - Sample image gallery with captions

The system gracefully degrades if wandb is not installed or no API key is present—it will print a warning and continue without logging.

## Project Structure

```
vis_ifeval/
├── src/
│   └── vis_ifeval/
│       ├── config.py              # Configuration management
│       ├── models/                # Image generation models
│       │   ├── base_model.py
│       │   ├── dummy_model.py
│       │   └── sdxl_model.py     # SDXL model (optional)
│       ├── evaluators/            # Constraint evaluators
│       │   ├── base.py
│       │   ├── text_eval.py       # Text evaluator (OCR-based)
│       │   ├── label_eval.py      # Nutrition label evaluator
│       │   ├── logic_eval.py      # Logic consistency evaluator
│       │   ├── negative_eval.py   # Negative constraint evaluator (CLIP placeholder)
│       │   └── comp_eval.py       # Composition evaluator (stub)
│       ├── runners/               # Pipeline scripts
│       │   ├── generate_images.py
│       │   ├── evaluate_constraints.py
│       │   └── aggregate_metrics.py
│       └── utils/
│           ├── io.py
│           ├── ocr_backend.py     # OCR backend abstraction
│           ├── clip_utils.py      # CLIP utilities (placeholder)
│           └── wandb_logger.py
├── prompts/
│   └── prompts.jsonl             # Benchmark prompts
├── data/
│   └── outputs/                  # Generated images
├── results/                      # Evaluation results
│   ├── generation_*.jsonl
│   ├── scores_*.jsonl
│   └── metrics_*.json
└── requirements.txt
```

## Prompt Format

Prompts are stored in JSONL format (`prompts/prompts.jsonl`). Each line is a JSON object:

```json
{
  "id": "comp_001",
  "category": "composition",
  "prompt": "A photo of three blue mugs...",
  "constraints": [
    {
      "id": "mug_count",
      "type": "count",
      "object": "blue mug",
      "target": 3
    },
    ...
  ]
}
```

### Constraint Types

- `text`: Text content (evaluated with OCR) ✅
- `table_slot`: Nutrition label fields (evaluated with OCR parsing) ✅
- `logic`: Logical relationships (e.g., percent_dv_consistency) ✅
- `negative`: Absence checks (placeholder, CLIP-ready) 🔲
- `count`: Object counts (stub) 🔲
- `attribute`: Object attributes (stub) 🔲
- `spatial`: Spatial relationships (stub) 🔲
- `state`: Object states (stub) 🔲

## Extending the Benchmark

### Adding a New Model

1. Create a new model class in `src/vis_ifeval/models/` that inherits from `ImageModel`
2. Implement the `generate()` method
3. Update `_build_model()` in `generate_images.py` to add your model

Example:
```python
elif model_name == "my_model":
    from vis_ifeval.models.my_model import MyModel
    return MyModel()
```

### Adding a New Evaluator

1. Create a new evaluator class in `src/vis_ifeval/evaluators/` that inherits from `ConstraintEvaluator`
2. Implement `can_handle()` and `score()` methods
3. Register it in `EvaluatorRegistry` (in `evaluators/__init__.py`)

If your evaluator needs OCR, accept a `TextBackend` in the constructor and use it for text extraction.

## License

[Add your license here]

