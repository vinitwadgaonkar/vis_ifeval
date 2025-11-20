# Installing LaTeX and Compiling IEEE Paper

## Installation Instructions

### Option 1: Full LaTeX Installation (Recommended)

```bash
sudo apt-get update
sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-latex-recommended texlive-fonts-recommended texlive-bibtex-extra texlive-publishers
```

This installs all necessary packages including IEEE format support.

### Option 2: Minimal Installation (Faster, ~500MB)

```bash
sudo apt-get update
sudo apt-get install -y texlive-latex-base texlive-latex-recommended texlive-publishers
```

### Option 3: Using Tlmgr (TeX Live Manager)

If you have a TeX Live installation:
```bash
tlmgr install ieeetran
```

## Compiling the Paper

After LaTeX is installed, compile the IEEE paper:

```bash
cd /home/csgrad/vinitwad/vinit_benchmark
pdflatex paper_ieee.tex
pdflatex paper_ieee.tex  # Run twice for references
```

Or use the provided script:
```bash
./compile_paper.sh
```

## Verifying Installation

Check if LaTeX is installed:
```bash
pdflatex --version
```

## Troubleshooting

If you get "IEEEtran.cls not found":
- Install texlive-publishers: `sudo apt-get install texlive-publishers`

If images don't appear:
- Ensure all PNG files are in `paper_assets/figures/` and `paper_assets/case_studies/`
- Check file paths in the .tex file

If compilation fails:
- Check `paper_ieee.log` for error messages
- Ensure all required packages are installed

## Alternative: Online LaTeX Compilers

If you cannot install LaTeX locally, you can use:
- Overleaf (https://www.overleaf.com) - Upload paper_ieee.tex and images
- ShareLaTeX - Similar online service

## File Structure for Compilation

```
vinit_benchmark/
├── paper_ieee.tex
├── paper_assets/
│   ├── figures/
│   │   ├── overall_pass_rate_comparison.png
│   │   ├── pass_rate_by_type_comparison.png
│   │   └── avg_score_by_type_comparison.png
│   └── case_studies/
│       ├── case_study_01_text_005.png
│       ├── case_study_02_comp_001.png
│       └── case_study_03_csp_01_numbers_row.png
```

