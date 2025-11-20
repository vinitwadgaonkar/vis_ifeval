# API-Based Model Integration Guide

This project now supports multiple API-based image generation services, so you don't need to run models locally!

## Available API Models

### 1. **OpenAI DALL-E** (`openai`)
- **API Key**: `OPENAI_API_KEY`
- **Get key**: https://platform.openai.com/api-keys
- **Cost**: Pay per image
- **Speed**: Very fast (~5-10 seconds)
- **Usage**:
  ```bash
  export OPENAI_API_KEY=your_key_here
  PYTHONPATH=src python -m vis_ifeval.runners.generate_images \
      --model-name openai \
      --prompts-path prompts/prompts_test.jsonl
  ```

### 2. **NovelAI** (`novelai`)
- **API Key**: `NOVELAI_API_KEY`
- **Get key**: https://novelai.net/ (requires subscription)
- **Cost**: Subscription-based
- **Speed**: Fast (~10-20 seconds)
- **Usage**:
  ```bash
  export NOVELAI_API_KEY=your_key_here
  PYTHONPATH=src python -m vis_ifeval.runners.generate_images \
      --model-name novelai \
      --prompts-path prompts/prompts_test.jsonl
  ```

### 3. **Banana.dev** (`banana`)
- **API Key**: `BANANA_API_KEY`
- **Model Key**: `BANANA_MODEL_KEY`
- **Get keys**: https://banana.dev/ (deploy your own model)
- **Cost**: Pay per inference
- **Speed**: Fast (~10-30 seconds)
- **Usage**:
  ```bash
  export BANANA_API_KEY=your_api_key
  export BANANA_MODEL_KEY=your_model_key
  PYTHONPATH=src python -m vis_ifeval.runners.generate_images \
      --model-name banana \
      --prompts-path prompts/prompts_test.jsonl
  ```

### 4. **Replicate** (`replicate`)
- **API Token**: `REPLICATE_API_TOKEN`
- **Get token**: https://replicate.com/account/api-tokens
- **Cost**: Pay per inference
- **Speed**: Fast (~10-30 seconds)
- **Models**: SDXL, FLUX, SD3, and many others
- **Usage**:
  ```bash
  export REPLICATE_API_TOKEN=your_token_here
  PYTHONPATH=src python -m vis_ifeval.runners.generate_images \
      --model-name replicate \
      --prompts-path prompts/prompts_test.jsonl
  ```

### 5. **Stability AI API** (`stability-api`)
- **API Key**: `STABILITY_API_KEY`
- **Get key**: https://platform.stability.ai/account/keys
- **Cost**: Pay per image
- **Speed**: Fast (~10-20 seconds)
- **Models**: SDXL, SD3, and other Stability models
- **Usage**:
  ```bash
  export STABILITY_API_KEY=your_key_here
  PYTHONPATH=src python -m vis_ifeval.runners.generate_images \
      --model-name stability-api \
      --prompts-path prompts/prompts_test.jsonl
  ```

## Quick Start

1. **Choose a service** and get your API key/token
2. **Set the environment variable**:
   ```bash
   export OPENAI_API_KEY=sk-...  # or whichever service you're using
   ```
3. **Run the pipeline**:
   ```bash
   PYTHONPATH=src python -m vis_ifeval.runners.generate_images \
       --model-name openai \
       --prompts-path prompts/prompts_test.jsonl
   ```

## Running Multiple Models

Use the automated script to test multiple API models:

```bash
PYTHONPATH=src python scripts/run_all_models.py \
    --models openai,novelai,replicate \
    --prompts-path prompts/prompts_test.jsonl \
    --limit-prompts 5
```

## Cost Comparison

| Service | Cost Model | Typical Cost/Image |
|---------|-----------|-------------------|
| OpenAI DALL-E | Pay per image | $0.04 - $0.12 |
| NovelAI | Subscription | Included |
| Banana.dev | Pay per inference | $0.01 - $0.05 |
| Replicate | Pay per inference | $0.002 - $0.01 |
| Stability AI | Pay per image | $0.04 - $0.08 |

## Advantages of API Models

✅ **No local GPU required** - Run on any machine  
✅ **Fast setup** - Just set an API key  
✅ **No model downloads** - No 7GB+ downloads  
✅ **Always up-to-date** - Latest model versions  
✅ **Scalable** - Handle many requests easily  

## Notes

- All API models require internet connection
- API keys should be kept secure (use environment variables)
- Some services have rate limits
- Costs can add up with many images - monitor usage
- Local models (SDXL, SD3, FLUX) still available if you prefer

