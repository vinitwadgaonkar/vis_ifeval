#!/bin/bash
# Create a package ready for Overleaf upload

cd "$(dirname "$0")"
PACKAGE_DIR="overleaf_package"
mkdir -p "$PACKAGE_DIR"

echo "Creating Overleaf package..."

# Copy LaTeX file
cp paper_ieee.tex "$PACKAGE_DIR/"

# Copy all images to a single directory (Overleaf needs them in the same folder or subfolder)
mkdir -p "$PACKAGE_DIR/figures"
cp paper_assets/figures/*.png "$PACKAGE_DIR/figures/" 2>/dev/null
cp paper_assets/case_studies/*.png "$PACKAGE_DIR/figures/" 2>/dev/null

# Update paths in LaTeX file for Overleaf
sed 's|paper_assets/figures/|figures/|g' paper_ieee.tex | \
sed 's|paper_assets/case_studies/|figures/|g' > "$PACKAGE_DIR/paper_ieee.tex"

# Create README for Overleaf
cat > "$PACKAGE_DIR/README.txt" << 'EOF'
OVERLEAF UPLOAD INSTRUCTIONS
============================

1. Go to https://www.overleaf.com
2. Create a new project
3. Upload paper_ieee.tex
4. Upload all PNG files from the figures/ folder
5. Click "Compile" (Recompile button)

The paper should compile successfully!

Files to upload:
- paper_ieee.tex
- All PNG files from figures/ folder
EOF

echo "✅ Overleaf package created in: $PACKAGE_DIR"
echo "   - paper_ieee.tex (with updated paths)"
echo "   - figures/ (all images)"
echo ""
echo "Upload this folder to Overleaf to compile!"
ls -lh "$PACKAGE_DIR"

