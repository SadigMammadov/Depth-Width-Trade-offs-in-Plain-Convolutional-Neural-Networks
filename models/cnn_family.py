"""
Phase 1 — VGG-style CNN family for the depth-width trade-off study.

Six models spanning the depth-width spectrum, all constrained to ~1.5 M parameters.
Width is constant across all stages (no channel doubling), so depth and width are
the only architectural variables — no confounds from growing channel sizes.

Model   Depth   Width   Params      Stage depths   Character
------  -----   -----   ----------  ------------   --------------------------
M1        4     233     1,497,358   (1, 1, 2)      Very shallow, very wide
M2        8     153     1,496,746   (2, 2, 4)      Shallow, wide
M3       12     122     1,492,038   (4, 4, 4)      Medium balanced
M4       16     105     1,505,170   (5, 5, 6)      Deep, narrow
M5       20      93     1,494,610   (6, 6, 8)      Very deep, narrow
M6       28      78     1,492,786   (9, 9, 10)     Extremely deep, very narrow

Architecture per model:
  Input  (3, 32, 32)
  Stage1 [d1 × ConvBNReLU, width channels] → MaxPool2d  → (width, 16, 16)
  Stage2 [d2 × ConvBNReLU, width channels] → MaxPool2d  → (width,  8,  8)
  Stage3 [d3 × ConvBNReLU, width channels] → GlobalAvgPool → (width,)
  Linear(width, 100)

Parameter budget derivation (per model):
  P = 9·3·W + (D-1)·9·W² + D·2·W + W·100 + 100
    ≈ 9·W²·D  for large D·W²
  Width solved via quadratic for P = 1,500,000.
"""

import torch.nn as nn


# ── Model configurations ──────────────────────────────────────────────────────
# (depth, width) pairs computed to give ≈1.5 M parameters each (within ±5%)
MODEL_CONFIGS = {
    'M1': {'depth':  4, 'width': 233},
    'M2': {'depth':  8, 'width': 153},
    'M3': {'depth': 12, 'width': 122},
    'M4': {'depth': 16, 'width': 105},
    'M5': {'depth': 20, 'width':  93},
    'M6': {'depth': 28, 'width':  78},
}


# ── Building blocks ───────────────────────────────────────────────────────────

class _ConvBNReLU(nn.Module):
    """3×3 Conv (no bias) → BatchNorm → ReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _Stage(nn.Module):
    """Sequential block of ConvBNReLU layers at a fixed spatial resolution."""
    def __init__(self, in_ch: int, out_ch: int, n_layers: int):
        super().__init__()
        blocks = [_ConvBNReLU(in_ch, out_ch)]
        for _ in range(n_layers - 1):
            blocks.append(_ConvBNReLU(out_ch, out_ch))
        # Named as layers.0, layers.1, ... for clean gradient-norm extraction
        self.layers = nn.ModuleList(blocks)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ── Main model ────────────────────────────────────────────────────────────────

class VGGStyleCNN(nn.Module):
    """
    Parameterisable VGG-style CNN for the depth-width trade-off study.

    Args:
        depth:       Total number of conv layers (distributed across 3 stages).
        width:       Number of channels — constant across all stages.
        num_classes: Output dimension (100 for CIFAR-100).
    """

    def __init__(self, depth: int, width: int, num_classes: int = 100):
        super().__init__()
        d1, d2, d3 = _split_depth(depth)

        self.stage1 = _Stage(3,     width, d1)
        self.pool1  = nn.MaxPool2d(2, 2)
        self.stage2 = _Stage(width, width, d2)
        self.pool2  = nn.MaxPool2d(2, 2)
        self.stage3 = _Stage(width, width, d3)
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(width, num_classes)

        # Metadata for analysis
        self.depth        = depth
        self.width        = width
        self.stage_depths = (d1, d2, d3)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool1(self.stage1(x))
        x = self.pool2(self.stage2(x))
        x = self.gap(self.stage3(x))
        x = x.flatten(1)
        return self.classifier(x)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_depth(depth: int) -> tuple:
    """Distribute total depth across 3 stages (min 1 layer per stage)."""
    d1 = max(1, depth // 3)
    d2 = max(1, depth // 3)
    d3 = max(1, depth - d1 - d2)
    return d1, d2, d3


def build_model_family(num_classes: int = 100) -> dict:
    """Instantiate all 6 models and return {name: VGGStyleCNN}."""
    return {
        name: VGGStyleCNN(**cfg, num_classes=num_classes)
        for name, cfg in MODEL_CONFIGS.items()
    }
