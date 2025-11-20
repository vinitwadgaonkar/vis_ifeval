# Visual Instruction Following Evaluation Benchmark - Paper Assets

This directory contains all assets for the benchmark paper submission.

## Structure

```
paper_assets/
├── comparison_data.json          # Complete comparison data
├── comparison_table.md           # Markdown comparison table
├── comparison_table.tex          # LaTeX comparison table
├── error_analysis.json           # Error analysis data
├── error_analysis_report.md      # Error analysis report
├── detailed_error_analysis.json  # Detailed error patterns
├── figures/                      # Visualizations
│   ├── pass_rate_by_type_comparison.png
│   ├── avg_score_by_type_comparison.png
│   └── overall_pass_rate_comparison.png
└── case_studies/                 # Case study images
    ├── case_study_01_text_005.png
    ├── case_study_02_comp_001.png
    ├── case_study_03_csp_01_numbers_row.png
    └── case_studies_metadata.json
```

## Paper Documents

- `paper.md` - Complete paper in Markdown format
- `paper.tex` - LaTeX version for PDF generation
- `evaluation_logic.txt` - Detailed evaluation methodology

## Generating PDF

To generate PDF from LaTeX:

```bash
pdflatex paper.tex
bibtex paper  # if using bibliography
pdflatex paper.tex
pdflatex paper.tex
```

Or convert Markdown to PDF:

```bash
pandoc paper.md -o paper.pdf --pdf-engine=xelatex
```

## Key Results Summary

- **Total Prompts**: 47
- **Total Constraints**: 167-172
- **Models Evaluated**: 3 (GPT Image 1, Nano Banana, DALL-E 3)
- **Best Overall Performance**: GPT Image 1 (58.1% pass rate)
- **Strongest Areas**: Spatial (100%), Character Consistency (100%), CSP (95-100%)
- **Weakest Areas**: Text Rendering (12-13%), Counting (17-31%)

## Evaluation Results Location

- GPT Image 1: `data/outputs/full_evaluation/results.json`
- Nano Banana: `data/outputs/full_evaluation_openrouter/results.json`
- DALL-E 3: `data/outputs/full_evaluation_dalle/results.json`

## Generated Images

All generated images are stored in:
- `data/outputs/full_evaluation/*.png` (GPT Image 1)
- `data/outputs/full_evaluation_openrouter/*.png` (Nano Banana)
- `data/outputs/full_evaluation_dalle/*.png` (DALL-E 3)

## Reproducing Results

See main README.md for instructions on running evaluations.

