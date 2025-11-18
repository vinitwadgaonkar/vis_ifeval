#!/bin/bash
# Test script for Gemini model - run after quota resets

export GEMINI_API_KEY=AIzaSyCwbpJTvTccCvU4HgZZO7G36N0Fy5_VIoY

echo "Testing Gemini model with 5 prompts..."
PYTHONPATH=src python -m vis_ifeval.runners.generate_images \
    --model-name gemini \
    --prompts-path prompts/prompts_gemini_test.jsonl \
    --output-dir data/outputs

echo "Checking generated images..."
ls -lh data/outputs/gemini/ 2>/dev/null || echo "No images generated yet"

echo "Done!"
