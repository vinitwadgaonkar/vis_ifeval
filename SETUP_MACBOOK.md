# Setup Guide for MacBook

## Quick Start

### 1. Clone the Repository

```bash
cd ~/Desktop  # or wherever you want the project
git clone https://github.com/vinitwadgaonkar/vis_ifeval.git
cd vis_ifeval
```

### 2. Install Python (if needed)

Check if Python 3.10+ is installed:
```bash
python3 --version
```

If not installed, install via Homebrew:
```bash
brew install python@3.10
```

### 3. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 5. Install System Dependencies

#### For DeepSeek OCR:
```bash
# Install PyTorch (CPU or MPS for Apple Silicon)
pip install torch torchvision torchaudio

# For Apple Silicon Macs (M1/M2/M3), use MPS backend:
# PyTorch will automatically use MPS if available
```

#### For GroundingDINO (Spatial Evaluator):
```bash
# Install additional dependencies
pip install groundingdino-py
```

#### For InsightFace (Character Consistency):
```bash
pip install insightface onnxruntime
```

#### For YOLOv8 (Character Detection):
```bash
pip install ultralytics
```

### 6. Download Model Weights

The models will be downloaded automatically on first use, but you can pre-download:

```bash
# Create weights directory
mkdir -p weights

# DeepSeek OCR will download automatically
# GroundingDINO will download automatically
# InsightFace models will download automatically
```

### 7. Set Up API Keys

Create a `.env` file in the project root:

```bash
cat > .env << EOF
OPENAI_API_KEY=your-openai-api-key-here
OPENROUTER_API_KEY=your-openrouter-api-key-here
VIS_IFEVAL_OCR_BACKEND=deepseek
EOF
```

Or export them in your shell:
```bash
export OPENAI_API_KEY='your-openai-api-key-here'
export OPENROUTER_API_KEY='your-openrouter-api-key-here'
export VIS_IFEVAL_OCR_BACKEND='deepseek'
```

Add to your `~/.zshrc` or `~/.bash_profile`:
```bash
echo 'export OPENAI_API_KEY="your-key-here"' >> ~/.zshrc
echo 'export OPENROUTER_API_KEY="your-key-here"' >> ~/.zshrc
echo 'export VIS_IFEVAL_OCR_BACKEND="deepseek"' >> ~/.zshrc
source ~/.zshrc
```

### 8. Verify Installation

```bash
python3 -c "import vis_ifeval; print('✅ Package installed')"
python3 -c "from vis_ifeval.evaluators import EvaluatorRegistry; print('✅ Evaluators loaded')"
```

### 9. Test Run

```bash
# Test a simple evaluation
python3 scripts/utils/run_all_models.py --help
```

## Project Structure

```
vis_ifeval/
├── src/vis_ifeval/          # Main package
│   ├── evaluators/          # All evaluators
│   ├── models/              # Model implementations
│   └── utils/               # Utilities (OCR, etc.)
├── scripts/                 # Scripts
│   ├── analysis/            # Analysis scripts
│   ├── paper/               # Paper generation
│   └── utils/               # Utility scripts
├── prompts/                 # Test prompts
├── results/                 # Evaluation results
├── paper/                   # Paper assets
├── docs/                    # Documentation
└── submission/              # Submission materials
```

## Common Issues

### Issue: PyTorch MPS (Apple Silicon) Support
If you have an M1/M2/M3 Mac, PyTorch should automatically use MPS. If not:
```bash
pip install torch torchvision torchaudio
```

### Issue: Missing System Libraries
```bash
# Install via Homebrew
brew install libpng jpeg libtiff freetype
```

### Issue: Permission Errors
```bash
# Make scripts executable
chmod +x scripts/**/*.sh
```

### Issue: Model Download Fails
Models are downloaded to `~/.cache/` by default. Ensure you have:
- Internet connection
- Sufficient disk space (~5-10GB for all models)
- Write permissions in home directory

## Running Evaluations

### Single Model Evaluation
```bash
export OPENAI_API_KEY='your-key'
python3 scripts/utils/run_all_models.py --model gpt-image-1
```

### Full Evaluation Suite
```bash
# See available options
python3 scripts/utils/run_all_models.py --help
```

## Data Transfer

If you need to transfer evaluation results from the server:

1. **From Server** (on the Linux machine):
```bash
cd /home/csgrad/vinitwad/vinit_benchmark
tar -czf results.tar.gz results/
# Then download via scp or GitHub
```

2. **To MacBook**:
```bash
# Via SCP (if you have SSH access)
scp user@server:/path/to/results.tar.gz ~/Downloads/
cd vis_ifeval
tar -xzf ~/Downloads/results.tar.gz
```

Or simply pull the latest from GitHub if results are committed.

## Next Steps

1. ✅ Clone repository
2. ✅ Set up environment
3. ✅ Install dependencies
4. ✅ Configure API keys
5. ✅ Test installation
6. ✅ Run evaluations

## Support

If you encounter issues:
1. Check the error message
2. Verify all dependencies are installed
3. Ensure API keys are set correctly
4. Check Python version (3.10+ required)

