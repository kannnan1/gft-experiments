# GiFT Experiments

Geometric Fine-Tuning (GiFT) experiments for measuring and mitigating catastrophic forgetting in parameter-efficient fine-tuning.

## Overview

This repository contains a complete experimental framework for comparing GiFT against LoRA and other PEFT methods across multiple architectures and tasks. The framework includes:

- **Comprehensive logging** with Weights & Biases integration
- **Progress tracking** with real-time metrics display
- **Automatic checkpointing** and experiment management
- **Geometric metrics** computation (distance preservation, CKA, SV divergence)
- **Multi-seed execution** for statistical validity
- **Batch execution scripts** for overnight runs

## Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd gft_experiments

# Install dependencies
pip install -r requirements.txt

# Set up wandb (optional but recommended)
export WANDB_API_KEY=<your-wandb-api-key>
wandb login
```

### Running a Single Experiment

```bash
# Run a single experiment with default config
python scripts/run_experiment.py --config configs/phase1/r1_resnet_lora_r16.yaml

# Run with specific seed
python scripts/run_experiment.py --config configs/phase1/r1_resnet_lora_r16.yaml --seed 42

# Resume from checkpoint
python scripts/run_experiment.py --config configs/phase1/r1_resnet_lora_r16.yaml --resume results/checkpoints/R1.4_seed42/last.pt
```

### Running Batch Experiments

```bash
# Run all Phase 1 experiments (overnight)
bash scripts/run_phase1_complete.sh
```

### Aggregating Results

```bash
# Aggregate results across all seeds
python scripts/aggregate_results.py
```

Results will be saved to `results/tables/aggregated_results.csv` and `aggregated_results.md`.

## Repository Structure

```
gft_experiments/
├── models/                    # Model implementations
│   ├── lora_linear.py        # LoRA implementation
│   ├── geometric_linear.py   # GFT implementation
│   └── model_factory.py      # Model creation utilities
├── experiments/               # Experiment implementations
│   ├── base_experiment.py    # Base experiment class
│   └── resnet_cifar10.py     # ResNet CIFAR-10 experiments
├── utils/                     # Utilities
│   ├── logger.py             # Logging with wandb
│   ├── progress.py           # Progress bars
│   ├── checkpoint.py         # Checkpoint management
│   ├── metrics.py            # Metrics computation
│   └── data_utils.py         # Dataset utilities
├── configs/                   # Configuration files
│   ├── base_config.yaml      # Base configuration
│   └── phase1/               # Phase 1 configs
├── scripts/                   # Execution scripts
│   ├── run_experiment.py     # Main runner
│   ├── run_phase1_complete.sh         # Batch Phase 1
│   └── aggregate_results.py  # Results aggregation
└── results/                   # Results directory
    ├── logs/                 # Training logs
    ├── checkpoints/          # Model checkpoints
    ├── tables/               # Result tables
    └── plots/                # Plots and visualizations
```

## Configuration

Experiments are configured via YAML files. See `configs/base_config.yaml` for all available options.

### Key Configuration Sections

- **experiment**: Name, ID, seeds
- **model**: Architecture, method (lora/gft/full_ft), rank
- **data**: Dataset, task type, batch size
- **training**: Epochs, learning rate, optimizer, scheduler
- **logging**: Wandb settings, logging intervals
- **evaluation**: What metrics to compute
- **paths**: Data and output directories

## Experiments

### Phase 1: ResNet on CIFAR-10

| Exp ID | Method | Rank | Task | Config File |
|--------|--------|------|------|-------------|
| R1.4 | LoRA | 16 | Even/Odd | `configs/phase1/r1_resnet_lora_r16.yaml` |
| R1.5 | GFT | 16 | Even/Odd | `configs/phase1/r1_resnet_gft_r16.yaml` |
| R2.1 | LoRA | 8 | Animal/Vehicle | `configs/phase1/r2_animal_vehicle_lora.yaml` |
| R2.2 | GFT | 8 | Animal/Vehicle | `configs/phase1/r2_animal_vehicle_gft.yaml` |

## Metrics Tracked

### Training Metrics (per epoch)
- Training loss and accuracy
- Validation loss and accuracy
- Learning rate
- Training time

### Catastrophic Forgetting Metrics
- Base task accuracy (before adaptation)
- Adaptation task accuracy
- Retention accuracy (base task after adaptation)
- Forgetting percentage
- Absolute accuracy drop
- Retention rate

### Geometric Metrics (optional)
- Distance preservation (Spearman ρ)
- Singular value divergence (KL)
- CKA similarity (layer-wise)
- Orthogonality measure (GFT only)

### Computational Metrics
- Total parameters
- Trainable parameters
- Training time per epoch
- GPU memory usage

## Machine Setup

For running experiments overnight on a VDI GPU machine:

1. **Clone repository on VDI**:
   ```bash
   git clone <repo-url>
   cd gft_experiments
   ```

2. **Setup environment**:
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Set wandb API key
   export WANDB_API_KEY=<your-key>
   ```

3. **Run batch experiments**:
   ```bash
   # Make script executable
   chmod +x scripts/run_phase1.sh
   
   # Run in background with nohup
   nohup bash scripts/run_phase1.sh > phase1_output.log 2>&1 &
   ```

4. **Monitor progress**:
   ```bash
   # Check log file
   tail -f phase1_output.log
   
   # Check wandb dashboard
   # Visit: https://wandb.ai/<your-username>/gft_experiments
   ```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in config
- Use gradient accumulation
- Use smaller model or rank

### Wandb Not Logging
- Check API key: `echo $WANDB_API_KEY`
- Login: `wandb login`
- Disable wandb: Set `use_wandb: false` in config

### Data Download Issues
- Check internet connection
- Manually download to `./data/` directory
- Verify disk space

## Citation

If you use this code, please cite:

```bibtex
@article{gift2025,
  title={Geometric Fine-Tuning (GFT): Decoupling Feature
Metrics and Decision Geometry via Trainable Rotor
Fields},
  author={Agus Sudjianto, Kannan V. and others},
  journal={arXiv preprint},
  year={2025}
}
```

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue or contact [your-email].
