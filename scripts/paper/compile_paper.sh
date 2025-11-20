#!/bin/bash
# Compile IEEE LaTeX paper to PDF

cd "$(dirname "$0")"

echo "Compiling IEEE LaTeX paper..."

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "ERROR: pdflatex not found!"
    echo "Please install LaTeX first:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-latex-recommended texlive-fonts-recommended texlive-bibtex-extra texlive-publishers"
    exit 1
fi

# Compile LaTeX document (run twice for references)
echo "Running pdflatex (first pass)..."
pdflatex -interaction=nonstopmode paper_ieee.tex > /dev/null 2>&1

echo "Running pdflatex (second pass for references)..."
pdflatex -interaction=nonstopmode paper_ieee.tex > /dev/null 2>&1

# Check if PDF was created
if [ -f "paper_ieee.pdf" ]; then
    echo "✅ PDF created successfully: paper_ieee.pdf"
    ls -lh paper_ieee.pdf
else
    echo "❌ PDF compilation failed. Check paper_ieee.log for errors."
    if [ -f "paper_ieee.log" ]; then
        echo "Last 20 lines of log:"
        tail -20 paper_ieee.log
    fi
    exit 1
fi

