# Repository Structure

## Overview

This repository has been reorganized for clarity and maintainability. All files are properly categorized and named.

## Directory Structure

```
vinit_benchmark/
├── README.md                    # Main README
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
├── .gitignore                  # Git ignore rules
│
├── src/                        # Source code
│   └── vis_ifeval/
│       ├── evaluators/         # Evaluation modules
│       ├── models/             # Model implementations
│       ├── runners/            # Evaluation runners
│       └── utils/              # Utility functions
│
├── prompts/                    # Prompt files
│   ├── prompts.jsonl
│   ├── prompts_character_consistency.jsonl
│   ├── prompts_sketch_to_render.jsonl
│   └── prompts_test.jsonl
│
├── scripts/                    # Executable scripts
│   ├── evaluation/             # Evaluation scripts
│   │   ├── run_full_evaluation.py
│   │   ├── run_evaluation_dalle3.py
│   │   ├── run_evaluation_nano_banana.py
│   │   ├── run_csp_evaluation.py
│   │   ├── test_special_evaluators.py
│   │   ├── test_new_evaluators.py
│   │   └── test_all_evaluators.py
│   │
│   ├── analysis/               # Analysis scripts
│   │   ├── merge_results.py
│   │   ├── create_visualizations.py
│   │   ├── create_case_studies.py
│   │   └── analyze_errors.py
│   │
│   ├── paper/                  # Paper generation scripts
│   │   ├── generate_pdf_report.py
│   │   ├── generate_paper_assets.py
│   │   ├── compile_paper.sh
│   │   ├── create_overleaf_package.sh
│   │   ├── install_latex.sh
│   │   └── add_cost_section.py
│   │
│   └── utils/                  # Utility scripts
│       └── run_all_models.py
│
├── docs/                       # Documentation
│   ├── evaluation/             # Evaluation documentation
│   │   ├── evaluation_logic.txt
│   │   └── evaluation_metrics.txt
│   │
│   ├── technical/              # Technical documentation
│   │   ├── ARCHITECTURE.md
│   │   ├── API_MODELS_GUIDE.md
│   │   ├── FOUNDER_TECHNICAL_DOC.md
│   │   ├── HLD.md
│   │   ├── REAL_MODEL_GUIDE.md
│   │   └── RENDERING.md
│   │
│   ├── guides/                 # User guides
│   │   ├── VISUALIZATION_GUIDE.md
│   │   ├── INSTALL_AND_COMPILE.md
│   │   ├── QUICK_START.md
│   │   ├── README_IEEE_PAPER.md
│   │   └── README_PAPER.md
│   │
│   └── reports/                # Reports and summaries
│       ├── models_performance_summary.txt
│       ├── COMPLETION_SUMMARY.md
│       ├── SUBMISSION_CHECKLIST.md
│       ├── FINAL_SUBMISSION_README.md
│       ├── CURRENT_STATUS.md
│       └── PROJECT_STATUS.md
│
├── paper/                      # Paper files
│   ├── paper.md                # Main paper (Markdown)
│   ├── paper_ieee.tex          # IEEE LaTeX version
│   ├── comprehensive_report.tex # Full report LaTeX
│   │
│   ├── assets/                 # Paper visualizations
│   │   ├── figures/            # Charts and graphs (10 files)
│   │   └── case_studies/       # Case study images (3 files)
│   │
│   └── data/                   # Paper data files
│       ├── comparison_data.json
│       ├── cost_analysis.json
│       └── error_analysis.json
│
├── results/                    # Evaluation results
│   ├── gpt_image1/
│   │   └── results.json
│   ├── nano_banana/
│   │   └── results.json
│   ├── dalle3/
│   │   └── results.json
│   └── comparison_data.json
│
└── submission/                 # Final submission package
    ├── README.md
    ├── paper.md
    ├── paper.tex
    ├── results/                # Result JSON files
    └── paper_assets/           # Submission assets
```

## File Naming Conventions

### Scripts
- Evaluation scripts: `run_evaluation_*.py` or `test_*.py`
- Analysis scripts: `create_*.py` or `analyze_*.py`
- Paper scripts: `generate_*.py` or `compile_*.sh`

### Documentation
- Evaluation: `evaluation_*.txt`
- Technical: `*.md` (ARCHITECTURE, API_MODELS_GUIDE, etc.)
- Guides: `*_GUIDE.md` or `README_*.md`
- Reports: `*_SUMMARY.md` or `*_SUMMARY.txt`

### Results
- Format: `{model_name}/results.json`
- Comparison: `comparison_data.json`

### Paper Assets
- Figures: Descriptive names (e.g., `overall_pass_rate_comparison.png`)
- Case studies: `case_study_{number}_{description}.png`

## Key Files

### Main Entry Points
- `scripts/evaluation/run_full_evaluation.py` - Main evaluation script
- `scripts/paper/generate_pdf_report.py` - Generate PDF report
- `scripts/analysis/create_visualizations.py` - Create all visualizations

### Documentation
- `README.md` - Main repository README
- `docs/evaluation/evaluation_logic.txt` - Evaluation methodology
- `docs/evaluation/evaluation_metrics.txt` - Metrics documentation
- `docs/reports/models_performance_summary.txt` - Performance summary

### Results
- `results/gpt_image1/results.json` - GPT Image 1 results
- `results/nano_banana/results.json` - Nano Banana results
- `results/dalle3/results.json` - DALL-E 3 results

## Excluded Files (.gitignore)

- `venv/` - Virtual environment
- `__pycache__/` - Python cache
- `*.log` - Log files
- `data/outputs/` - Generated images
- `*.pdf`, `*.zip` - Large generated files
- `weights/`, `*.pt`, `*.pth` - Model weights
- `deepseek_ocr_repo/` - External repos

## Usage

### Running Evaluations
```bash
# Full evaluation
python scripts/evaluation/run_full_evaluation.py

# Specific model
python scripts/evaluation/run_evaluation_dalle3.py
```

### Generating Paper
```bash
# Generate PDF
python scripts/paper/generate_pdf_report.py

# Create visualizations
python scripts/analysis/create_visualizations.py
```

### Viewing Results
```bash
# Performance summary
cat docs/reports/models_performance_summary.txt

# Results JSON
cat results/gpt_image1/results.json
```

