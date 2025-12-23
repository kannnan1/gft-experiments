#!/bin/bash
# Complete Phase 1: All ResNet CIFAR-10/100 experiments
# This includes R1.x (core), R2.x (alternative splits), and R3.x (CIFAR-100)

set -e  # Exit on error

echo "=========================================="
echo "GiFT Experiments - COMPLETE PHASE 1"
echo "ResNet on CIFAR-10 and CIFAR-100"
echo "=========================================="

# Seeds for statistical validity
SEEDS="42"

# Base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "Working directory: $BASE_DIR"
echo "Seeds: $SEEDS"
echo ""
echo "This will run:"
echo "  - R1.1-R1.7: Core comparison (Even/Odd split)"
echo "  - R2.1-R2.6: Alternative splits (Animal/Vehicle, >5 vs <5, Living/Non-living)"
echo "  - R3.2-R3.5: ResNet-50 on CIFAR-100"
echo ""
echo "Total: 18 experiment configurations × 3 seeds = 54 runs"
echo "Estimated time: ~12-15 hours on GPU"
echo ""

# ============================================
# R2.x: Alternative Splits
# ============================================

echo ""
echo "=========================================="
echo "R2.x: Alternative Splits (Semantic Validity)"
echo "=========================================="

# R2.1-R2.2: Animal/Vehicle
echo "Running R2.1-R2.2 (Animal/Vehicle)..."
#for seed in $SEEDS; do
#    python scripts/run_experiment.py --config configs/phase1/r2_animal_vehicle_lora.yaml --seed $seed
#    python scripts/run_experiment.py --config configs/phase1/r2_animal_vehicle_gft.yaml --seed $seed
#done

# R2.3-R2.4: >5 vs <5
echo "Running R2.3-R2.4 (>5 vs <5)..."
for seed in $SEEDS; do
#    python scripts/run_experiment.py --config configs/phase1/r2_3_gt5_lt5_lora.yaml --seed $seed
    python scripts/run_experiment.py --config configs/phase1/r2_4_gt5_lt5_gft.yaml --seed $seed
done

# R2.5-R2.6: Living/Non-living
echo "Running R2.5-R2.6 (Living/Non-living)..."
for seed in $SEEDS; do
    python scripts/run_experiment.py --config configs/phase1/r2_5_living_nonliving_lora.yaml --seed $seed
    python scripts/run_experiment.py --config configs/phase1/r2_6_living_nonliving_gft.yaml --seed $seed
done

# ============================================
# Aggregate Results
# ============================================

echo ""
echo "=========================================="
echo "Aggregating Phase 1 Results..."
echo "=========================================="

python scripts/aggregate_results.py

echo ""
echo "=========================================="
echo "PHASE 1 COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to: $BASE_DIR/results/"
echo "Tables: $BASE_DIR/results/tables/"
echo "Logs: $BASE_DIR/results/logs/"
echo "Checkpoints: $BASE_DIR/results/checkpoints/"
echo ""
echo "Next steps:"
echo "  1. Review aggregated results in results/tables/"
echo "  2. Check wandb dashboard for detailed metrics"
echo "  3. Run analysis scripts for geometric metrics"
echo "  4. Generate paper figures"
echo "  5. Proceed to Phase 2 (CLIP) when ready"
