#!/bin/bash
# Quick test script for vis_ifeval pipeline

set -e

echo "Testing vis_ifeval pipeline..."
echo ""

# Step 1: Generate
echo "1. Generating images..."
PYTHONPATH=src python -m vis_ifeval.runners.generate_images --model-name dummy > /dev/null 2>&1
echo "   ✓ Images generated"

# Step 2: Evaluate
echo "2. Evaluating constraints..."
PYTHONPATH=src python -m vis_ifeval.runners.evaluate_constraints --model-name dummy > /dev/null 2>&1
echo "   ✓ Constraints evaluated"

# Step 3: Aggregate
echo "3. Aggregating metrics..."
PYTHONPATH=src python -m vis_ifeval.runners.aggregate_metrics --model-name dummy > /dev/null 2>&1
echo "   ✓ Metrics aggregated"

# Show results
echo ""
echo "Results:"
python -c "import json; d=json.load(open('results/metrics_dummy.json')); print(f\"  VIPR: {d['vipr']:.4f}\"); print(f\"  Total: {d['total_constraints']} constraints\"); print(f\"  Satisfied: {d['total_satisfied']}\")"

echo ""
echo "✓ All tests passed!"
