#!/bin/bash
# Run all Phase 1 ResNet CIFAR-10 experiments
# This script runs experiments overnight on VDI GPU machine

set -e  # Exit on error

echo "=========================================="
echo "GiFT Experiments - Phase 1"
echo "ResNet-18 on CIFAR-10"
echo "=========================================="

# Seeds for statistical validity
SEEDS="42 123 456"

# Base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "Working directory: $BASE_DIR"
echo "Seeds: $SEEDS"
echo ""

# R1.4-R1.5: Rank 16 experiments
echo "=========================================="
echo "Running R1.4-R1.5: Rank 16 experiments"
echo "=========================================="

for seed in $SEEDS; do
    echo "Running R1.4 (LoRA r=16) with seed $seed..."
    python scripts/run_experiment.py --config configs/phase1/r1_resnet_lora_r16.yaml --seed $seed
    
    echo "Running R1.5 (GFT r=16) with seed $seed..."
    python scripts/run_experiment.py --config configs/phase1/r1_resnet_gft_r16.yaml --seed $seed
done

# R2.1-R2.2: Animal/Vehicle split
echo "=========================================="
echo "Running R2.1-R2.2: Animal/Vehicle split"
echo "=========================================="

for seed in $SEEDS; do
    echo "Running R2.1 (LoRA Animal/Vehicle) with seed $seed..."
    python scripts/run_experiment.py --config configs/phase1/r2_animal_vehicle_lora.yaml --seed $seed
    
    echo "Running R2.2 (GFT Animal/Vehicle) with seed $seed..."
    python scripts/run_experiment.py --config configs/phase1/r2_animal_vehicle_gft.yaml --seed $seed
done

echo ""
echo "=========================================="
echo "Phase 1 experiments complete!"
echo "=========================================="
echo ""
echo "Results saved to: $BASE_DIR/results/"
echo "Logs: $BASE_DIR/results/logs/"
echo "Checkpoints: $BASE_DIR/results/checkpoints/"
echo ""
echo "To aggregate results, run:"
echo "  python scripts/aggregate_results.py"
