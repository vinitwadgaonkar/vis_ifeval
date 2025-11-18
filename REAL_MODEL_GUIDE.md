# Using Real Image Models

## Quick Start

### SDXL Model (Recommended for testing)

```bash
# Install dependencies
pip install torch diffusers transformers accelerate

# Generate images (will download model ~7GB on first run)
PYTHONPATH=src python -m vis_ifeval.runners.generate_images \
    --model-name sdxl \
    --prompts-path prompts/prompts_test.jsonl

# Evaluate
PYTHONPATH=src python -m vis_ifeval.runners.evaluate_constraints \
    --model-name sdxl \
    --prompts-path prompts/prompts_test.jsonl

# Aggregate
PYTHONPATH=src python -m vis_ifeval.runners.aggregate_metrics \
    --model-name sdxl
```

### SD3 Model

```bash
PYTHONPATH=src python -m vis_ifeval.runners.generate_images \
    --model-name sd3 \
    --prompts-path prompts/prompts_test.jsonl
```

### FLUX Model

```bash
PYTHONPATH=src python -m vis_ifeval.runners.generate_images \
    --model-name flux \
    --prompts-path prompts/prompts_test.jsonl
```

### OpenAI DALL-E

```bash
# Set API key
export OPENAI_API_KEY=your_key_here

PYTHONPATH=src python -m vis_ifeval.runners.generate_images \
    --model-name openai \
    --prompts-path prompts/prompts_test.jsonl
```

## Notes

- **First run**: Models will download (SDXL ~7GB, SD3 ~5GB, FLUX ~24GB)
- **GPU recommended**: CPU works but is very slow
- **Memory**: Ensure sufficient RAM/VRAM
- **Time**: Real models take 10-60 seconds per image depending on hardware

## Testing with Limited Prompts

Create a test prompt file:

```bash
# Take first 2 prompts
head -2 prompts/prompts.jsonl > prompts/prompts_test.jsonl
```

Then use `--prompts-path prompts/prompts_test.jsonl` for faster testing.
