# Visual Instruction Following Evaluation Benchmark - Submission Package

## Contents

- `paper.md` - Complete benchmark paper (Markdown)
- `paper.tex` - LaTeX version for PDF generation
- `evaluation_logic.txt` - Detailed evaluation methodology
- `paper_assets/` - All figures, tables, and case studies
- `results/` - Evaluation results in JSON format
- `SUBMISSION_CHECKLIST.md` - Submission checklist

## Generating PDF

```bash
# From Markdown
pandoc paper.md -o paper.pdf --pdf-engine=xelatex

# From LaTeX
pdflatex paper.tex
```

## Key Results

- **Models Evaluated**: 3 (GPT Image 1, Nano Banana, DALL-E 3)
- **Total Prompts**: 47
- **Total Constraints**: 167-172
- **Best Performance**: GPT Image 1 (58.1% pass rate)
- **Most Cost-Effective**: Nano Banana ($0.24 for 47 images)

See paper.md for complete details.
