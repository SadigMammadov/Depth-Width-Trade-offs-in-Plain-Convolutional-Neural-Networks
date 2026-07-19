# COMP5329 Deep Learning — Assignment 2 Project Plan

## Research Hypothesis

> A fundamental question in convolutional neural network design is how to optimally
> allocate a fixed parameter budget between model depth and width. We believe deeper
> models can capture hierarchical compositional features while wider models can encode
> richer input representations. Thus we hypothesize that the optimal depth-to-width ratio
> for various tasks is based around that axis. We investigate this hypothesis systematically
> on CIFAR-100, a challenging fine-grained classification benchmark with 100 semantic
> categories. We construct a controlled family of CNN architectures that span a range of
> depth-width configurations while holding the total parameter count constant, enabling a
> fair and direct comparison across architectural choices. In addition to classification
> accuracy, we analyze training dynamics such as loss landscapes, convergence behavior,
> and gradient flow to understand why certain configurations succeed or fail. Our main
> purpose is to provide both empirical evidence and mechanistic insight into the depth-width
> trade-off, offering principled guidance for CNN architecture design under resource
> constraints.

---

## Key Design Decisions

- **Dataset:** CIFAR-100 (60,000 images, 32×32, 100 classes, 500 train + 100 test per class)
- **Framework:** PyTorch (Google Colab)
- **Fixed parameter budget:** ~1.5M parameters (±5% tolerance per model)
- **Architecture style:** VGG-style plain CNN (Conv → BN → ReLU blocks, 3 spatial downsampling stages)
- **Number of models:** 6 models spanning the depth-width spectrum

---

## Model Family (all ≈ 1.5M parameters)

Width is **constant across all stages** (no channel doubling) so depth and width
are the only architectural variables — no confounds from growing channel sizes.

| Model | Depth | Width | Params     | Stage depths | Character |
|-------|-------|-------|------------|--------------|-----------|
| M1    | 4     | 233   | 1,497,358  | (1, 1, 2)    | Very shallow, very wide |
| M2    | 8     | 153   | 1,496,746  | (2, 2, 4)    | Shallow, wide |
| M3    | 12    | 122   | 1,492,038  | (4, 4, 4)    | Medium balanced |
| M4    | 16    | 105   | 1,505,170  | (5, 5, 6)    | Deep, narrow |
| M5    | 20    | 93    | 1,494,610  | (6, 6, 8)    | Very deep, narrow |
| M6    | 28    | 78    | 1,492,786  | (9, 9, 10)   | Extremely deep, very narrow |

Stage depths = layers per stage (Stage1, Stage2, Stage3).
Width solved per model via: P = 9·3·W + (D-1)·9·W² + D·2·W + W·100 + 100 = 1,500,000 ± 5%.

---

## Training Protocol (identical for all 6 models)

- **Optimizer:** SGD with momentum 0.9, weight decay 5e-4
- **LR schedule:** Cosine annealing decay
- **Epochs:** 200
- **Batch size:** 128
- **Data augmentation:** RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize
- **Logged per epoch:** train loss, val loss, val top-1 accuracy, val top-5 accuracy, per-layer gradient L2 norms
- **Checkpoints:** Save best val accuracy checkpoint per model

---

## Project Phases

### Phase 0 — Environment Setup
- [ ] Install dependencies in Colab (torch, torchvision, matplotlib, numpy, cv2/PIL)
- [ ] Download and verify CIFAR-100 dataset
- [ ] Write `DataLoader` with standard augmentation pipeline
- [ ] Write `count_params(model)` utility
- [ ] Write training loop with logging (loss, accuracy, gradient norms per layer)
- [ ] Test full pipeline end-to-end on a small dummy model before scaling up

### Phase 1 — Architecture Family Design
- [ ] Implement parameterizable `VGGStyleCNN(depth, base_width)` class in PyTorch
- [ ] Instantiate all 6 models (M1–M6) and verify parameter counts ≈ 1.5M each
- [ ] Do a single forward pass on a dummy batch to confirm shapes

### Phase 2 — Training (compute-heavy)
- [ ] Train all 6 models with identical hyperparameters
- [ ] Run models sequentially in Colab (save checkpoints to Google Drive after each)
- [ ] Estimated time: ~2–4 hours per model on Colab GPU
- [ ] Save: training logs (CSV), final checkpoint, best checkpoint per model

### Phase 3 — Image Complexity Scoring 
- [ ] For each image in the CIFAR-100 test set (10,000 images), compute Laplacian variance
  as the sharpness score: `cv2.Laplacian(img, cv2.CV_64F).var()`
- [ ] Sort all scores and split into 3 tiers by percentile:
  - Low complexity: bottom 33% (smooth/blurry images)
  - Medium complexity: middle 33%
  - High complexity: top 33% (sharp/edge-rich images)
- [ ] Save tier assignments as a dict: `{image_index: tier_label}`
- [ ] For each trained model, compute per-tier accuracy on the test set

**Theoretical motivation:** Deeper models extract hierarchical edge→part→object features,
so they may outperform on high-edge-density images. Wider models capture broader
channel-level correlations and may handle simpler images more robustly. This analysis
turns a performance comparison into mechanistic insight.

### Phase 4 — Analysis & Visualization
- [ ] **Main result:** Accuracy vs. depth-width ratio curve (all 6 models)
- [ ] **Convergence:** Val accuracy vs. epoch for all 6 models on one plot
- [ ] **Per-tier accuracy table:** rows = models, columns = complexity tier
- [ ] **Gradient flow heatmap:** x-axis = layer index, y-axis = model, color = mean gradient norm
  → reveals vanishing/exploding gradient patterns across depth
- [ ] **Per-class analysis:** For best deep model and best wide model, compute accuracy
  on each of the 100 classes; group by CIFAR-100 superclass (20 superclasses)
- [ ] **Loss landscape (optional):** Perturb weights along 2D random directions for M1, M3, M6

### Phase 5 — Paper Writing
Structure (6–8 pages, double-column, LaTeX template from course):

1. **Abstract** — already submitted to OpenReview
2. **Introduction** — motivate depth-width question, state hypothesis explicitly
3. **Related Work** — depth vs. width literature:
   - Wide ResNets (Zagoruyko & Komodakis, 2016)
   - ResNet depth scaling (He et al., 2016)
   - EfficientNet compound scaling (Tan & Le, 2019)
   - Gradient flow in deep networks literature
4. **Method** — architecture family design, fixed parameter constraint derivation,
   training protocol, image complexity scoring methodology
5. **Experiments & Results** — main accuracy table, training dynamics plots,
   complexity-stratified analysis, gradient flow heatmap
6. **Discussion** — when does depth win? When does width win? Why?
   Limitations: 32×32 images constrain depth, single dataset
7. **Conclusion**
8. **References**

---

## Division of Labor (suggested)

| Task | Owner |
|------|-------|
| Model implementation + training | You |
| Training loop, logging, checkpointing | You |
| Gradient flow + loss landscape analysis | You |
| Image complexity scoring (Laplacian variance) | Teammate |
| Per-tier accuracy breakdown | Teammate |
| Per-class / per-superclass analysis | Either |
| Paper writing | Both |

---

## File Structure (target)

```
COMP5329_Deep_Learning_As2/
├── PROJECT_PLAN.md          ← this file
├── main.ipynb               ← main Colab notebook
├── models/
│   └── cnn_family.py        ← parameterizable CNN architecture
├── utils/
│   ├── train.py             ← training loop
│   ├── evaluate.py          ← evaluation + per-class accuracy
│   └── complexity.py        ← image complexity scoring
├── results/
│   ├── logs/                ← training CSVs
│   ├── checkpoints/         ← model .pth files
│   └── figures/             ← saved plots
└── paper/                   ← LaTeX source
```

---

## Current Status

- [x] Abstract submitted to OpenReview
- [x] Project plan written
- [x] Phase 0 — Environment setup (smoke test passed)
- [x] Phase 1 — Architecture family (VGGStyleCNN, 6 models, parameter budgets verified)
- [ ] Phase 2 — Training (IN PROGRESS — run Phase 2 cell in main.ipynb)
- [ ] Phase 3 — Image complexity scoring
- [ ] Phase 4 — Analysis
- [ ] Phase 5 — Paper

**Paper deadline:** Sunday, 24 May 2026 at 23:59 (Sydney time)

---

## Notes for New Claude Sessions

- Always read this file first to understand the project state
- The hypothesis and parameter budget are fixed — do not change them
- All 6 models MUST use identical training protocols for results to be comparable
- The image complexity stratification is an ANALYSIS tool, not the primary experiment
- The primary metric is top-1 accuracy on CIFAR-100 test set
- Colab notebook is `main.ipynb` — all code lives there unless modularized