# GiFT Experiments - Phase 2 & 3 Setup Guide

## Phase 2: CLIP Domain Adaptation

### Dataset Requirements

#### ImageNet Validation Set
1. Download ImageNet validation set (50,000 images)
2. Organize in standard format:
   ```
   data/imagenet/
   ├── val/
   │   ├── n01440764/
   │   ├── n01443537/
   │   └── ...
   ```

#### ImageNet-Sketch
1. Download from: https://github.com/HaohanWang/ImageNet-Sketch
2. Extract to:
   ```
   data/imagenet-sketch/
   ├── sketch/
   │   ├── n01440764/
   │   ├── n01443537/
   │   └── ...
   ```

### Running CLIP Experiments

```bash
# Single experiment
python scripts/run_experiment.py --config configs/phase2/c1_clip_sketch_lora.yaml

# All Phase 2 experiments
bash scripts/run_phase2.sh
```

### Expected Results (from experimental plan)
- **C1.2 (LoRA)**: ~14.7% forgetting on ImageNet zero-shot
- **C1.3 (GFT)**: ~5.9% forgetting on ImageNet zero-shot
- **GFT Advantage**: ~2.5x better retention

---

## Phase 3: BLIP Captioning

### Dataset Requirements

#### COCO Captions
1. Download COCO 2017:
   ```bash
   wget http://images.cocodataset.org/zips/train2017.zip
   wget http://images.cocodataset.org/zips/val2017.zip
   wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   ```

2. Extract to:
   ```
   data/coco/
   ├── train2017/
   ├── val2017/
   └── annotations/
       ├── captions_train2017.json
       └── captions_val2017.json
   ```

#### Medical Image Captions
- **Note**: Medical caption datasets require special access
- Options:
  - ROCO (Radiology Objects in COntext)
  - MIMIC-CXR (requires credentialing)
  - IU X-Ray (Indiana University)

### Running BLIP Experiments

```bash
# Single experiment
python scripts/run_experiment.py --config configs/phase3/b1_blip_medical_lora.yaml

# All Phase 3 experiments
bash scripts/run_phase3.sh
```

### Expected Results (from experimental plan)
- **B1.2 (LoRA)**: ~23.3% CIDEr degradation
- **B1.3 (GFT)**: ~10.0% CIDEr degradation
- **GFT Advantage**: ~2.3x better retention

---

## Caption Metrics Installation

For proper caption evaluation, install pycocoevalcap:

```bash
pip install pycocoevalcap
# or
pip install git+https://github.com/salaniz/pycocoevalcap.git
```

This provides:
- **CIDEr**: Consensus-based metric (primary)
- **BLEU-4**: N-gram overlap
- **METEOR**: Semantic similarity
- **ROUGE-L**: Longest common subsequence

---

## Implementation Notes

### CLIP Experiments
- Vision encoder is adapted with LoRA/GFT
- Text encoder remains frozen
- Zero-shot evaluation on ImageNet measures retention
- Fine-tuning on target domain (Sketch) measures adaptation

### BLIP Experiments
- Vision encoder is adapted with LoRA/GFT
- Text decoder can optionally be adapted (not implemented yet)
- Caption quality measured with CIDEr, BLEU, METEOR
- Qualitative analysis shows LoRA captions become generic

---

## Computational Requirements

### Phase 2 (CLIP)
- **GPU Memory**: ~8-10 GB
- **Time per experiment**: ~4-6 hours (10 epochs)
- **Total Phase 2**: ~20-30 hours

### Phase 3 (BLIP)
- **GPU Memory**: ~10-12 GB
- **Time per experiment**: ~6-8 hours (20 epochs)
- **Total Phase 3**: ~30-40 hours

### Total for All Phases
- **Estimated time**: 3-5 days on single GPU
- **Storage**: ~50-100 GB (including datasets)

---

## Troubleshooting

### CLIP Issues
- **Out of memory**: Reduce batch size to 32 or 16
- **Slow training**: Reduce image resolution or use smaller CLIP model
- **Dataset not found**: Check ImageNet/Sketch paths in config

### BLIP Issues
- **Out of memory**: Reduce batch size to 16 or 8
- **Caption metrics error**: Install pycocoevalcap
- **Slow generation**: Reduce num_beams or max_length

### General
- **Wandb not logging**: Check API key with `echo $WANDB_API_KEY`
- **Import errors**: Ensure transformers installed: `pip install transformers>=4.30.0`
- **CUDA errors**: Check CUDA version matches PyTorch build

---

## Next Steps

After running experiments:

1. **Aggregate results**:
   ```bash
   python scripts/aggregate_results.py
   ```

2. **Check wandb dashboard** for detailed metrics and visualizations

3. **Generate paper figures** (see analysis/ directory)

4. **Run ablation studies** (Phase 4) for deeper analysis
