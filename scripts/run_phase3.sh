#!/bin/bash
# Run all Phase 3 BLIP experiments
# COCO → Medical captions

set -e  # Exit on error

echo "=========================================="
echo "GiFT Experiments - Phase 3"
echo "BLIP Caption Adaptation"
echo "=========================================="

# Seeds for statistical validity
SEEDS="42 123 456"

# Base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "Working directory: $BASE_DIR"
echo "Seeds: $SEEDS"
echo ""

# B1.2-B1.3: COCO → Medical
echo "=========================================="
echo "Running B1.2-B1.3: COCO → Medical"
echo "=========================================="

for seed in $SEEDS; do
    echo "Running B1.2 (BLIP LoRA r=16) with seed $seed..."
    python scripts/run_experiment.py --config configs/phase3/b1_blip_medical_lora.yaml --seed $seed
    
    echo "Running B1.3 (BLIP GFT r=16) with seed $seed..."
    python scripts/run_experiment.py --config configs/phase3/b1_blip_medical_gft.yaml --seed $seed
done

echo ""
echo "=========================================="
echo "Phase 3 experiments complete!"
echo "=========================================="
echo ""
echo "Results saved to: $BASE_DIR/results/"
echo "Logs: $BASE_DIR/results/logs/"
echo "Checkpoints: $BASE_DIR/results/checkpoints/"
echo ""
echo "To aggregate results, run:"
echo "  python scripts/aggregate_results.py"
