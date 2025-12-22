#!/bin/bash
# Master script to run ALL GiFT experiments (Phase 1, 2, and 3)
# WARNING: This will take a LONG time (multiple days on GPU)

set -e  # Exit on error

echo "=========================================="
echo "GiFT Experiments - ALL PHASES"
echo "Complete Experimental Suite"
echo "=========================================="

# Base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "Working directory: $BASE_DIR"
echo ""
echo "This will run:"
echo "  - Phase 1: ResNet CIFAR-10 experiments"
echo "  - Phase 2: CLIP domain adaptation experiments"
echo "  - Phase 3: BLIP captioning experiments"
echo ""
echo "Estimated total time: 3-5 days on single GPU"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# Phase 1: ResNet CIFAR-10
echo ""
echo "=========================================="
echo "Starting Phase 1: ResNet CIFAR-10"
echo "=========================================="
bash scripts/run_phase1.sh

# Phase 2: CLIP
echo ""
echo "=========================================="
echo "Starting Phase 2: CLIP Domain Adaptation"
echo "=========================================="
bash scripts/run_phase2.sh

# Phase 3: BLIP
echo ""
echo "=========================================="
echo "Starting Phase 3: BLIP Captioning"
echo "=========================================="
bash scripts/run_phase3.sh

# Aggregate all results
echo ""
echo "=========================================="
echo "Aggregating all results..."
echo "=========================================="
python scripts/aggregate_results.py

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to: $BASE_DIR/results/"
echo "Tables: $BASE_DIR/results/tables/"
echo "Logs: $BASE_DIR/results/logs/"
echo "Checkpoints: $BASE_DIR/results/checkpoints/"
echo ""
echo "Check wandb dashboard for detailed metrics:"
echo "  https://wandb.ai/<your-username>/gft_experiments"
