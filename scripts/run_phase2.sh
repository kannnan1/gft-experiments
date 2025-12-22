#!/bin/bash
# Run all Phase 2 CLIP experiments
# ImageNet → Sketch domain adaptation

set -e  # Exit on error

echo "=========================================="
echo "GiFT Experiments - Phase 2"
echo "CLIP Domain Adaptation"
echo "=========================================="

# Seeds for statistical validity
SEEDS="42 123 456"

# Base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "Working directory: $BASE_DIR"
echo "Seeds: $SEEDS"
echo ""

# C1.2-C1.3: ImageNet → Sketch
echo "=========================================="
echo "Running C1.2-C1.3: ImageNet → Sketch"
echo "=========================================="

for seed in $SEEDS; do
    echo "Running C1.2 (CLIP LoRA r=16) with seed $seed..."
    python scripts/run_experiment.py --config configs/phase2/c1_clip_sketch_lora.yaml --seed $seed
    
    echo "Running C1.3 (CLIP GFT r=16) with seed $seed..."
    python scripts/run_experiment.py --config configs/phase2/c1_clip_sketch_gft.yaml --seed $seed
done

echo ""
echo "=========================================="
echo "Phase 2 experiments complete!"
echo "=========================================="
echo ""
echo "Results saved to: $BASE_DIR/results/"
echo "Logs: $BASE_DIR/results/logs/"
echo "Checkpoints: $BASE_DIR/results/checkpoints/"
echo ""
echo "To aggregate results, run:"
echo "  python scripts/aggregate_results.py"
