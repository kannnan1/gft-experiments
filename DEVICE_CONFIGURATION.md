# Device Configuration Guide

## Overview

The GiFT experiments repository supports multiple device types for maximum compatibility across different systems:
- **CUDA** (NVIDIA GPUs)
- **MPS** (Apple Silicon M1/M2/M3)
- **CPU** (fallback)

## Configuration

### Option 1: Auto-Detection (Default)

By default, the system will automatically detect and use the best available device:

```yaml
# In config file (or omit entirely)
device: "auto"
```

**Detection priority**:
1. CUDA (if available)
2. MPS (if available on Apple Silicon)
3. CPU (fallback)

### Option 2: Explicit Device Selection

Specify the device explicitly in your config file:

```yaml
# Force CUDA
device: "cuda"

# Force MPS (Apple Silicon)
device: "mps"

# Force CPU
device: "cpu"
```

## Usage Examples

### Running on NVIDIA GPU (CUDA)

```bash
# Auto-detect (will use CUDA if available)
python scripts/run_experiment.py --config configs/phase1/r1_2_lora_r8.yaml

# Or explicitly specify
# Edit config file: device: "cuda"
```

### Running on Apple Silicon (MPS)

```bash
# Auto-detect (will use MPS on M1/M2/M3 Macs)
python scripts/run_experiment.py --config configs/phase1/r1_2_lora_r8.yaml

# Or explicitly in config:
# device: "mps"
```

### Running on CPU

```bash
# Explicitly specify CPU in config
# device: "cpu"
python scripts/run_experiment.py --config configs/phase1/r1_2_lora_r8.yaml
```

## Device-Specific Considerations

### CUDA (NVIDIA GPUs)

**Recommended for**:
- Production experiments
- Large-scale training
- Fastest performance

**Batch sizes**:
- ResNet-18: 128-256
- ResNet-50: 64-128
- CLIP: 32-64
- BLIP: 16-32

**Expected speed**: ~30 min per ResNet-18 experiment

### MPS (Apple Silicon)

**Recommended for**:
- Development and testing
- Small-scale experiments
- M1/M2/M3 Macs

**Batch sizes** (reduce by ~50% vs CUDA):
- ResNet-18: 64-128
- ResNet-50: 32-64
- CLIP: 16-32
- BLIP: 8-16

**Expected speed**: ~45-60 min per ResNet-18 experiment

**Note**: MPS support requires PyTorch 1.12+ and macOS 12.3+

### CPU

**Recommended for**:
- Testing only
- Very small experiments
- Debugging

**Batch sizes** (reduce significantly):
- ResNet-18: 16-32
- ResNet-50: 8-16

**Expected speed**: ~2-4 hours per ResNet-18 experiment

**Not recommended for production experiments**

## Troubleshooting

### CUDA Issues

**"CUDA out of memory"**:
```yaml
# Reduce batch size in config
data:
  batch_size: 64  # or 32
```

**"CUDA not available"**:
- Check CUDA installation: `nvidia-smi`
- Verify PyTorch CUDA build: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall PyTorch with CUDA support

### MPS Issues

**"MPS not available"**:
- Requires macOS 12.3+
- Requires PyTorch 1.12+
- Check: `python -c "import torch; print(torch.backends.mps.is_available())"`

**"MPS out of memory"**:
- Reduce batch size
- Close other applications
- MPS shares memory with system

### CPU Performance

**Very slow training**:
- This is expected on CPU
- Consider using cloud GPU (Google Colab, AWS, etc.)
- Or use smaller model/dataset for testing

## VDI GPU Machine Setup

For running on VDI GPU machine:

```bash
# 1. Check available device
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# 2. Run experiments (will auto-detect CUDA)
bash scripts/run_phase1_complete.sh

# 3. Monitor GPU usage
watch -n 1 nvidia-smi
```

## Performance Comparison

| Device | ResNet-18 (50 epochs) | ResNet-50 (50 epochs) | CLIP (10 epochs) |
|--------|----------------------|----------------------|------------------|
| CUDA (V100) | ~30 min | ~60 min | ~4 hours |
| CUDA (RTX 3090) | ~25 min | ~50 min | ~3 hours |
| MPS (M1 Max) | ~50 min | ~90 min | ~6 hours |
| CPU (16 cores) | ~3 hours | ~6 hours | ~24 hours |

## Best Practices

1. **Development**: Use MPS or small CUDA GPU
2. **Testing**: Use CPU with small batch sizes
3. **Production**: Use CUDA on VDI/cloud GPU
4. **Overnight runs**: Always use CUDA for full experiment suite

## Configuration Template

```yaml
# Recommended settings for different devices

# CUDA (VDI/Cloud)
device: "auto"  # Will detect CUDA
data:
  batch_size: 128
  num_workers: 4

# MPS (Apple Silicon)
device: "mps"
data:
  batch_size: 64
  num_workers: 2

# CPU (Testing only)
device: "cpu"
data:
  batch_size: 16
  num_workers: 2
training:
  epochs: 5  # Reduce for testing
```
