# vis_ifeval Project Status Summary

## Project Overview
**vis_ifeval** is a Visual Instruction Following Evaluation Benchmark that tests whether generated images satisfy dense prompts with multiple constraints (8-12 per prompt). The benchmark computes VIPR (Visual Instruction Pass Rate) as the primary metric.

---

## Current Implementation Status

### ✅ FULLY IMPLEMENTED (Production Ready)

#### 1. **Image Generation Models**
- **DummyModel** (`src/vis_ifeval/models/dummy_model.py`)
  - **Status**: ✅ Fully implemented
  - **What it does**: Generates random 256x256 RGB images using numpy
  - **Purpose**: Testing/development - produces random noise, not real images
  - **Usage**: `--model-name dummy`
  
- **SDXLModel** (`src/vis_ifeval/models/sdxl_model.py`)
  - **Status**: ✅ Fully implemented (optional dependency)
  - **What it does**: Real Stable Diffusion XL image generation via HuggingFace diffusers
  - **Purpose**: Production use - generates actual images from prompts
  - **Requirements**: `torch`, `diffusers`, `transformers` (not in requirements.txt yet)
  - **Usage**: `--model-name sdxl` (requires GPU recommended)

#### 2. **OCR Backend System**
- **TesseractBackend** (`src/vis_ifeval/utils/ocr_backend.py`)
  - **Status**: ✅ Fully implemented
  - **What it does**: Real OCR using pytesseract/Tesseract
  - **Purpose**: Extracts text from images for evaluation
  - **Usage**: Default backend, configured via `VIS_IFEVAL_OCR_BACKEND=tesseract`

- **AdvancedBackend** (placeholder)
  - **Status**: 🔲 Stub/placeholder
  - **What it does**: Placeholder for future Surya/DeepSeek-OCR integration
  - **Purpose**: Will support advanced OCR models later

#### 3. **Constraint Evaluators**

- **TextEvaluator** (`src/vis_ifeval/evaluators/text_eval.py`)
  - **Status**: ✅ Fully implemented
  - **What it does**: Real OCR-based text evaluation using Tesseract
  - **How it works**: 
    - Extracts text from image using OCR backend
    - Computes Character Error Rate (CER) vs target text
    - Converts to score [0,1] using exponential decay
  - **Handles**: `constraint["type"] == "text"`
  - **Real or Dummy**: **REAL** - actually evaluates text in images

- **LabelEvaluator** (`src/vis_ifeval/evaluators/label_eval.py`)
  - **Status**: ✅ Fully implemented
  - **What it does**: Real nutrition label parsing and evaluation
  - **How it works**:
    - Crops label region from image
    - Extracts text via OCR
    - Parses nutrition fields (serving_size, calories, sodium, etc.) using regex
    - Compares parsed values to targets using CER + numeric refinement
  - **Handles**: `constraint["type"] == "table_slot"`
  - **Real or Dummy**: **REAL** - actually parses and evaluates nutrition labels

- **LogicEvaluator** (`src/vis_ifeval/evaluators/logic_eval.py`)
  - **Status**: ✅ Fully implemented
  - **What it does**: Real logic consistency checks (e.g., sodium mg vs %DV)
  - **How it works**:
    - Reuses LabelEvaluator parsing
    - Validates internal consistency (e.g., 50mg sodium = 2% DV using 2300mg daily reference)
    - Computes relative error and converts to score
  - **Handles**: `constraint["type"] == "logic"` with `logic_type: "percent_dv_consistency"`
  - **Real or Dummy**: **REAL** - actually checks logical consistency

- **NegativeEvaluator** (`src/vis_ifeval/evaluators/negative_eval.py`)
  - **Status**: 🔲 Placeholder (structure ready, logic pending)
  - **What it does**: Currently returns 1.0 (assumes success)
  - **Future**: Will use CLIP to check for absence of concepts
  - **Handles**: `constraint["type"] == "negative"`
  - **Real or Dummy**: **DUMMY** - placeholder behavior, CLIP integration needed

- **CompositionEvaluator** (`src/vis_ifeval/evaluators/comp_eval.py`)
  - **Status**: 🔲 Stub
  - **What it does**: Returns 0.0, logs warning
  - **Future**: Will use object detection (GroundingDINO) for count/attribute/spatial/state
  - **Handles**: `constraint["type"] in {"count", "attribute", "spatial", "state"}`
  - **Real or Dummy**: **DUMMY** - not implemented yet

#### 4. **Pipeline Infrastructure**
- **generate_images.py**: ✅ Fully implemented
  - Generates images from prompts
  - Saves images and logs generation metadata
  - Supports W&B logging
  
- **evaluate_constraints.py**: ✅ Fully implemented
  - Loads generated images
  - Evaluates all constraints using EvaluatorRegistry
  - Saves scores to JSONL
  
- **aggregate_metrics.py**: ✅ Fully implemented
  - Computes VIPR (overall and by type/category)
  - Calculates latency statistics
  - Saves metrics to JSON

#### 5. **Supporting Infrastructure**
- **Config system**: ✅ Fully implemented (env vars, W&B config, OCR backend selection)
- **W&B integration**: ✅ Fully implemented (graceful degradation if not available)
- **CLI**: ✅ Fully implemented (argparse with --model-name, --use-wandb flags)
- **IO utilities**: ✅ Fully implemented (JSONL load/save)

---

## What's Dummy vs Real

### DUMMY (Testing/Placeholder)
1. **DummyModel**: Generates random noise images (not real images)
2. **CompositionEvaluator**: Returns 0.0, doesn't actually evaluate
3. **NegativeEvaluator**: Returns 1.0 placeholder, doesn't use CLIP yet
4. **AdvancedBackend**: Placeholder for future OCR backends

### REAL (Production Ready)
1. **SDXLModel**: Real image generation (if dependencies installed)
2. **TesseractBackend**: Real OCR text extraction
3. **TextEvaluator**: Real text evaluation using OCR
4. **LabelEvaluator**: Real nutrition label parsing and evaluation
5. **LogicEvaluator**: Real consistency checking
6. **Pipeline**: Fully functional end-to-end

---

## Current Test Results

### With DummyModel (Random Images)
- **VIPR**: 0.0476 (4.76%) - Expected low score for random images
- **Breakdown**:
  - text: 0.0 (random images have no readable text)
  - table_slot: 0.0 (random images have no nutrition labels)
  - logic: 0.0 (no labels to check consistency)
  - negative: 1.0 (placeholder returns success)
  - count/attribute/spatial/state: 0.0 (stub evaluator)

### Expected with Real Models (SDXL)
- **VIPR**: Should be much higher (depends on model quality)
- **text**: Should score well if text is clearly visible
- **table_slot**: Should score well if nutrition labels are readable
- **logic**: Should score well if labels are parsed correctly
- **negative**: Will need CLIP integration for real evaluation
- **count/attribute/spatial/state**: Will need CompositionEvaluator implementation

---

## Project Structure

```
vis_ifeval/
├── src/vis_ifeval/
│   ├── config.py              ✅ Config with env vars
│   ├── models/
│   │   ├── base_model.py      ✅ Abstract interface
│   │   ├── dummy_model.py     ✅ Random images (dummy)
│   │   └── sdxl_model.py      ✅ Real SDXL (optional)
│   ├── evaluators/
│   │   ├── base.py            ✅ Abstract interface
│   │   ├── text_eval.py       ✅ REAL - OCR text evaluation
│   │   ├── label_eval.py      ✅ REAL - Nutrition label parsing
│   │   ├── logic_eval.py      ✅ REAL - Consistency checks
│   │   ├── negative_eval.py   🔲 DUMMY - CLIP placeholder
│   │   └── comp_eval.py       🔲 DUMMY - Stub
│   ├── runners/
│   │   ├── generate_images.py      ✅ Full pipeline
│   │   ├── evaluate_constraints.py ✅ Full pipeline
│   │   └── aggregate_metrics.py    ✅ Full pipeline
│   └── utils/
│       ├── io.py              ✅ JSONL utilities
│       ├── ocr_backend.py     ✅ OCR abstraction (Tesseract real, Advanced stub)
│       ├── clip_utils.py      🔲 CLIP placeholder
│       └── wandb_logger.py    ✅ W&B integration
├── prompts/
│   └── prompts.jsonl          ✅ 3 example prompts
├── data/outputs/              ✅ Generated images stored here
├── results/                   ✅ Evaluation results stored here
└── requirements.txt           ✅ Dependencies (wandb, pytesseract, etc.)
```

---

## Prompt Format

Each prompt in `prompts/prompts.jsonl` has:
- `id`: Unique identifier
- `category`: One of "composition", "poster_text", "nutrition_label", "logic_negative"
- `prompt`: Full text description
- `constraints`: Array of constraint objects

### Constraint Types Currently Supported

1. **text** ✅ REAL
   ```json
   {"id": "headline", "type": "text", "target": "SPRING SALE", "region": "label_top"}
   ```

2. **table_slot** ✅ REAL
   ```json
   {"id": "sodium", "type": "table_slot", "field": "sodium", "target": "50 mg"}
   ```

3. **logic** ✅ REAL
   ```json
   {"id": "sodium_consistency", "type": "logic", "logic_type": "percent_dv_consistency", ...}
   ```

4. **negative** 🔲 DUMMY
   ```json
   {"id": "no_sugar", "type": "negative", "concept": "sugar_drink"}
   ```

5. **count/attribute/spatial/state** 🔲 DUMMY
   ```json
   {"id": "mug_count", "type": "count", "object": "blue mug", "target": 3}
   ```

---

## How to Run

### Basic Pipeline (Dummy Model)
```bash
PYTHONPATH=src python -m vis_ifeval.runners.generate_images --model-name dummy
PYTHONPATH=src python -m vis_ifeval.runners.evaluate_constraints --model-name dummy
PYTHONPATH=src python -m vis_ifeval.runners.aggregate_metrics --model-name dummy
```

### With W&B Logging
```bash
export VIS_IFEVAL_USE_WANDB=1
# Then run pipeline as above
```

### With SDXL (if dependencies installed)
```bash
pip install torch diffusers transformers
PYTHONPATH=src python -m vis_ifeval.runners.generate_images --model-name sdxl
# ... rest of pipeline
```

---

## Next Steps / TODO

### High Priority
1. **Implement CompositionEvaluator** - Use GroundingDINO or similar for:
   - Object counting
   - Attribute detection
   - Spatial relationship reasoning
   - State detection

2. **Integrate CLIP in NegativeEvaluator** - Use open_clip_torch to:
   - Encode images and text concepts
   - Compute similarity scores
   - Return low score if forbidden concept is present

3. **Add more prompts** - Expand `prompts.jsonl` with more test cases

### Medium Priority
4. **Integrate Advanced OCR** - Add Surya or DeepSeek-OCR backends
5. **Add SD3 model** - Similar to SDXL hook
6. **Improve label parsing** - More robust regex/ML-based parsing

### Low Priority
7. **Add more logic types** - Beyond percent_dv_consistency
8. **Performance optimization** - Caching, batching
9. **Better error handling** - More graceful failures

---

## Key Files to Understand

1. **EvaluatorRegistry** (`evaluators/__init__.py`): Routes constraints to appropriate evaluators
2. **OCR Backend** (`utils/ocr_backend.py`): Abstraction for text extraction
3. **Config** (`config.py`): Centralized configuration management
4. **Runners**: Three-step pipeline (generate → evaluate → aggregate)

---

## Dependencies Status

### Installed & Working
- ✅ pillow, numpy, tqdm
- ✅ pytesseract (requires system Tesseract)
- ✅ python-Levenshtein
- ✅ wandb
- ✅ matplotlib

### Optional (for SDXL)
- 🔲 torch
- 🔲 diffusers
- 🔲 transformers

### Future (for advanced features)
- 🔲 open_clip_torch (for CLIP-based negative evaluation)
- 🔲 groundingdino (for composition evaluation)
- 🔲 surya-ocr / deepseek-ocr (for advanced OCR)

---

## Summary for ChatGPT

**Current State**: The benchmark is **fully functional** with a working end-to-end pipeline. 

**What works for real**:
- Text evaluation (OCR-based)
- Nutrition label parsing and evaluation
- Logic consistency checks
- SDXL image generation (if dependencies installed)

**What's dummy/placeholder**:
- DummyModel (random images for testing)
- CompositionEvaluator (stub, returns 0.0)
- NegativeEvaluator (placeholder, returns 1.0, needs CLIP)
- Advanced OCR backends (placeholder structure)

**The pipeline runs successfully** and produces metrics. With real image models, you get meaningful VIPR scores. The dummy model is intentionally producing low scores (random images don't satisfy constraints), which is expected behavior.

