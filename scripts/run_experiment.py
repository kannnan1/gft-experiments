#!/usr/bin/env python3
"""Main experiment runner script."""

import argparse
import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.resnet_cifar10 import ResNetCIFAR10Experiment, ResNetCIFAR10TwoStageExperiment
from experiments.clip_adaptation import CLIPAdaptationExperiment
from experiments.blip_captioning import BLIPCaptioningExperiment


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_experiment(config_path: str, seed: int = None, resume: str = None):
    """Run a single experiment.
    
    Args:
        config_path: Path to configuration file
        seed: Random seed (overrides config if provided)
        resume: Path to checkpoint to resume from
    """
    # Load config
    config = load_config(config_path)
    
    # Override seed if provided
    if seed is not None:
        config['experiment']['seeds'] = [seed]
    
    # Determine experiment type based on config
    architecture = config['model']['architecture']
    
    if architecture == 'clip':
        experiment_class = CLIPAdaptationExperiment
    elif architecture == 'blip':
        experiment_class = BLIPCaptioningExperiment
    elif config['data']['base_task'] == 'cifar10':
        # Use two-stage experiment for catastrophic forgetting measurement
        if config['evaluation']['eval_base_task']:
            experiment_class = ResNetCIFAR10TwoStageExperiment
        else:
            experiment_class = ResNetCIFAR10Experiment
    else:
        raise ValueError(f"Unsupported architecture/task: {architecture}/{config['data']['base_task']}")
    
    # Run experiment for each seed
    for exp_seed in config['experiment']['seeds']:
        print(f"\n{'='*60}")
        print(f"Running experiment: {config['experiment']['name']}")
        print(f"Experiment ID: {config['experiment']['exp_id']}")
        print(f"Seed: {exp_seed}")
        print(f"{'='*60}\n")
        
        # Create and run experiment
        experiment = experiment_class(config, seed=exp_seed)
        experiment.setup()
        experiment.run()
        
        print(f"\n{'='*60}")
        print(f"Experiment complete: {config['experiment']['exp_id']} (seed {exp_seed})")
        print(f"{'='*60}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run GiFT experiments')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Run experiment
    run_experiment(args.config, args.seed, args.resume)


if __name__ == '__main__':
    main()
