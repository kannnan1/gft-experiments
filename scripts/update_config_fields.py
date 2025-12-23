
import os
import yaml
from pathlib import Path

def update_configs(configs_dir):
    config_path = Path(configs_dir)
    yaml_files = list(config_path.glob('**/*.yaml'))
    
    for yaml_file in yaml_files:
        print(f"Updating {yaml_file}...")
        
        with open(yaml_file, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        in_training = False
        has_skip = False
        has_checkpoint = False
        
        # Detect if they already exist
        for line in lines:
            stripped = line.strip()
            if stripped == 'training:':
                in_training = True
            elif in_training and stripped.startswith('skip_stage1:'):
                has_skip = True
            elif in_training and stripped.startswith('stage1_checkpoint:'):
                has_checkpoint = True
            elif in_training and (not line.startswith('  ') and stripped != '' and not stripped.startswith('#') and stripped != 'training:'):
                in_training = False
        
        # Reset and process
        in_training = False
        for i, line in enumerate(lines):
            new_lines.append(line)
            stripped = line.strip()
            
            if stripped == 'training:':
                in_training = True
            elif in_training and (i + 1 == len(lines) or (not lines[i+1].startswith('  ') and lines[i+1].strip() != '' and not lines[i+1].strip().startswith('#'))):
                # We are at the end of the training block
                if not has_skip:
                    new_lines.append("  skip_stage1: false\n")
                if not has_checkpoint:
                    new_lines.append("  stage1_checkpoint: \"\"\n")
                in_training = False
                
        with open(yaml_file, 'w') as f:
            f.writelines(new_lines)

if __name__ == "__main__":
    update_configs('/Users/kannanvenkataramanan/Documents/Agus/gft_experiments/configs')
