#!/bin/bash

# Setup script for MacBook
# Run this after cloning the repository

set -e

echo "🚀 Setting up vis_ifeval on MacBook..."
echo ""

# Check Python version
echo "📋 Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+ first."
    echo "   Install via Homebrew: brew install python@3.10"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Found Python $PYTHON_VERSION"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo "📥 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    echo "✅ Requirements installed"
else
    echo "⚠️  requirements.txt not found, skipping..."
fi

# Install package in editable mode
if [ -f "pyproject.toml" ]; then
    echo "📦 Installing package in editable mode..."
    pip install -e . --quiet
    echo "✅ Package installed"
else
    echo "⚠️  pyproject.toml not found, skipping package installation..."
fi

# Install additional dependencies for Mac
echo "🍎 Installing Mac-specific dependencies..."

# PyTorch (will use MPS on Apple Silicon)
echo "  - Installing PyTorch..."
pip install torch torchvision torchaudio --quiet

# Additional ML dependencies
echo "  - Installing ML dependencies..."
pip install groundingdino-py insightface onnxruntime ultralytics --quiet

echo "✅ Mac-specific dependencies installed"

# Create .env file template
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file template..."
    cat > .env << 'EOF'
# API Keys (replace with your actual keys)
OPENAI_API_KEY=your-openai-api-key-here
OPENROUTER_API_KEY=your-openrouter-api-key-here

# OCR Backend
VIS_IFEVAL_OCR_BACKEND=deepseek
EOF
    echo "✅ .env file created (please add your API keys)"
else
    echo "✅ .env file already exists"
fi

# Create weights directory
mkdir -p weights
echo "✅ Created weights directory"

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python3 -c "import vis_ifeval; print('✅ vis_ifeval package imported successfully')" 2>/dev/null || echo "⚠️  Package import check skipped"

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Add your API keys to .env file:"
echo "      nano .env"
echo ""
echo "   2. Activate the virtual environment:"
echo "      source venv/bin/activate"
echo ""
echo "   3. Test the installation:"
echo "      python3 -c \"from vis_ifeval.evaluators import EvaluatorRegistry; print('✅ Ready!')\""
echo ""
echo "   4. Run evaluations:"
echo "      python3 scripts/utils/run_all_models.py --help"
echo ""

