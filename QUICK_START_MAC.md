# Quick Start for MacBook

## One-Command Setup

```bash
# Clone and setup
git clone https://github.com/vinitwadgaonkar/vis_ifeval.git
cd vis_ifeval
chmod +x setup_macbook.sh
./setup_macbook.sh
```

## Manual Setup (Step by Step)

### 1. Clone Repository
```bash
git clone https://github.com/vinitwadgaonkar/vis_ifeval.git
cd vis_ifeval
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 4. Install ML Dependencies
```bash
pip install torch torchvision torchaudio
pip install groundingdino-py insightface onnxruntime ultralytics
```

### 5. Configure API Keys
```bash
# Create .env file
cat > .env << EOF
OPENAI_API_KEY=your-key-here
OPENROUTER_API_KEY=your-key-here
VIS_IFEVAL_OCR_BACKEND=deepseek
EOF

# Or export in shell
export OPENAI_API_KEY='your-key-here'
export OPENROUTER_API_KEY='your-key-here'
```

### 6. Test Installation
```bash
python3 -c "from vis_ifeval.evaluators import EvaluatorRegistry; print('✅ Ready!')"
```

## Transfer Results from Server (Optional)

If you want to copy evaluation results from the server:

```bash
# On your MacBook, use SCP:
scp -r user@server:/home/csgrad/vinitwad/vinit_benchmark/results ./results

# Or download from GitHub if results are committed
git pull origin main
```

## Run Evaluations

```bash
# Activate environment
source venv/bin/activate

# Set API keys
export OPENAI_API_KEY='your-key'
export OPENROUTER_API_KEY='your-key'

# Run evaluation
python3 scripts/utils/run_all_models.py --model gpt-image-1
```

## Troubleshooting

### Python Not Found
```bash
brew install python@3.10
```

### Permission Denied
```bash
chmod +x setup_macbook.sh
```

### Import Errors
```bash
# Reinstall package
pip install -e . --force-reinstall
```

### MPS (Apple Silicon) Issues
PyTorch should automatically detect and use MPS on M1/M2/M3 Macs. If not:
```bash
pip install torch torchvision torchaudio --upgrade
```

## Project Structure

- `src/vis_ifeval/` - Main package code
- `scripts/` - Evaluation and analysis scripts
- `prompts/` - Test prompts
- `results/` - Evaluation results
- `paper/` - Paper assets
- `docs/` - Documentation

## Need Help?

See `SETUP_MACBOOK.md` for detailed instructions.

