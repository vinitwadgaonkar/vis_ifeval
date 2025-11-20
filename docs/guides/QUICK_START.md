# Quick Start - Compile IEEE Paper

## Install LaTeX (One-time setup)

```bash
sudo apt-get update
sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-latex-recommended texlive-fonts-recommended texlive-bibtex-extra texlive-publishers
```

## Compile Paper

```bash
cd /home/csgrad/vinitwad/vinit_benchmark
pdflatex paper_ieee.tex
pdflatex paper_ieee.tex
```

Output: `paper_ieee.pdf`

## Alternative: Use Online (No Installation)

1. Go to https://www.overleaf.com
2. Upload `paper_ieee.tex` and all images from `paper_assets/`
3. Click "Compile"

## Files Ready

✅ paper_ieee.tex - Complete IEEE LaTeX document
✅ All visualizations in paper_assets/
✅ All case studies included
✅ All tables formatted
✅ Ready to compile!
