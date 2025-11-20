#!/bin/bash
# Install LaTeX packages (run with appropriate permissions)

echo "Installing LaTeX packages..."

# Check if we can install
if command -v apt-get &> /dev/null; then
    echo "Using apt-get to install LaTeX..."
    # Note: This may require sudo
    echo "To install LaTeX, run:"
    echo "sudo apt-get update"
    echo "sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-latex-recommended texlive-fonts-recommended texlive-bibtex-extra texlive-publishers texlive-latex-extra"
    echo ""
    echo "For IEEE format, also install:"
    echo "sudo apt-get install -y texlive-publishers"
else
    echo "apt-get not available. Please install LaTeX manually."
fi

echo ""
echo "After installation, compile with:"
echo "pdflatex paper_ieee.tex"
echo "pdflatex paper_ieee.tex  # Run twice for references"

