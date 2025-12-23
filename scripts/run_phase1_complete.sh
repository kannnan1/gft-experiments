#!/bin/bash
# Complete Phase 1: All ResNet CIFAR-10/100 experiments
# This includes R1.x (core), R2.x (alternative splits), and R3.x (CIFAR-100)

set -e  # Exit on error

echo "=========================================="
echo "GiFT Experiments - COMPLETE PHASE 1"
echo "ResNet on CIFAR-10 and CIFAR-100"
echo "=========================================="

# Seeds for statistical validity
SEEDS="42 123 456"

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
# R1.x: Core Comparison (Even/Odd)
# ============================================

echo "=========================================="
echo "R1.x: Core Comparison (Even/Odd Split)"
echo "=========================================="

# R1.1: Baseline
echo "Running R1.1 (Baseline)..."
for seed in $SEEDS; do
    python scripts/run_experiment.py --config configs/phase1/r1_1_baseline.yaml --seed $seed
done

# R1.2: LoRA r=8 (DONE - can skip if already run)
echo "Running R1.2 (LoRA r=8)..."
for seed in $SEEDS; do
    python scripts/run_experiment.py --config configs/phase1/r1_2_lora_r8.yaml --seed $seed
done

# R1.3: GFT r=8 (DONE - can skip if already run)
echo "Running R1.3 (GFT r=8)..."
for seed in $SEEDS; do
    python scripts/run_experiment.py --config configs/phase1/r1_3_gft_r8.yaml --seed $seed
done

# R1.4: LoRA r=16
echo "Running R1.4 (LoRA r=16)..."
for seed in $SEEDS; do
    python scripts/run_experiment.py --config configs/phase1/r1_4_lora_r16.yaml --seed $seed
done

# R1.5: GFT r=16
echo "Running R1.5 (GFT r=16)..."
for seed in $SEEDS; do
    python scripts/run_experiment.py --config configs/phase1/r1_5_gft_r16.yaml --seed $seed
done

# R1.6: Adapter
echo "Running R1.6 (Adapter)..."
for seed in $SEEDS; do
    python scripts/run_experiment.py --config configs/phase1/r1_6_adapter.yaml --seed $seed
done

# R1.7: Full Fine-Tuning
echo "Running R1.7 (Full FT)..."
for seed in $SEEDS; do
    python scripts/run_experiment.py --config configs/phase1/r1_7_full_ft.yaml --seed $seed
done

# ============================================
# R2.x: Alternative Splits
# ============================================

echo ""
echo "=========================================="
echo "R2.x: Alternative Splits (Semantic Validity)"
echo "=========================================="

# R2.1-R2.2: Animal/Vehicle
echo "Running R2.1-R2.2 (Animal/Vehicle)..."
for seed in $SEEDS; do
    python scripts/run_experiment.py --config configs/phase1/r2_animal_vehicle_lora.yaml --seed $seed
    python scripts/run_experiment.py --config configs/phase1/r2_animal_vehicle_gft.yaml --seed $seed
done

# R2.3-R2.4: >5 vs <5
echo "Running R2.3-R2.4 (>5 vs <5)..."
for seed in $SEEDS; do
    python scripts/run_experiment.py --config configs/phase1/r2_3_gt5_lt5_lora.yaml --seed $seed
    python scripts/run_experiment.py --config configs/phase1/r2_4_gt5_lt5_gft.yaml --seed $seed
done

# R2.5-R2.6: Living/Non-living
echo "Running R2.5-R2.6 (Living/Non-living)..."
for seed in $SEEDS; do
    python scripts/run_experiment.py --config configs/phase1/r2_5_living_nonliving_lora.yaml --seed $seed
    python scripts/run_experiment.py --config configs/phase1/r2_6_living_nonliving_gft.yaml --seed $seed
done

# ============================================
# R3.x: ResNet-50 on CIFAR-100
# ============================================

echo ""
echo "=========================================="
echo "R3.x: ResNet-50 on CIFAR-100 (Scaling)"
echo "=========================================="

# R3.2-R3.3: Rank 16
echo "Running R3.2-R3.3 (ResNet-50, r=16)..."
for seed in $SEEDS; do
    python scripts/run_experiment.py --config configs/phase1/r3_2_resnet50_cifar100_lora_r16.yaml --seed $seed
    python scripts/run_experiment.py --config configs/phase1/r3_3_resnet50_cifar100_gft_r16.yaml --seed $seed
done

# R3.4-R3.5: Rank 32
echo "Running R3.4-R3.5 (ResNet-50, r=32)..."
for seed in $SEEDS; do
    python scripts/run_experiment.py --config configs/phase1/r3_4_resnet50_cifar100_lora_r32.yaml --seed $seed
    python scripts/run_experiment.py --config configs/phase1/r3_5_resnet50_cifar100_gft_r32.yaml --seed $seed
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
