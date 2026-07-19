"""
Phase 3 — Image complexity scoring.

Each CIFAR-100 test image (32x32) is scored by its Laplacian variance,
a standard proxy for edge sharpness / texture richness.  Images are then
stratified into three tiers (low / medium / high complexity) by percentile.

Theoretical motivation:
  Deeper CNNs extract hierarchical edge → part → object features, so they may
  outperform on high-edge-density images.  Wider CNNs capture broader
  channel-level correlations and may handle simpler, smoother images better.
  Per-tier accuracy lets us test this mechanistic hypothesis empirically.
"""

import numpy as np
import cv2


def laplacian_variance(img_tensor) -> float:
    """
    Compute Laplacian variance of a single CHW float tensor (values 0–1).
    Higher value = more edges / sharper image.
    """
    # Convert CHW float → HWC uint8 grayscale
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def score_dataset(test_set) -> np.ndarray:
    """
    Score every image in the test set.

    Args:
        test_set: torchvision dataset with ToTensor() transform (no normalisation).

    Returns:
        scores: np.ndarray of shape (N,), one score per image.
    """
    scores = np.zeros(len(test_set), dtype=np.float32)
    for i, (img, _) in enumerate(test_set):
        scores[i] = laplacian_variance(img)
        if i % 2000 == 0:
            print(f'  Scored {i}/{len(test_set)} images...')
    print(f'  Done. Score range: [{scores.min():.2f}, {scores.max():.2f}]')
    return scores


def stratify(scores: np.ndarray, n_tiers: int = 3) -> dict:
    """
    Assign each image to a complexity tier by equal-frequency binning.

    Returns:
        tier_labels: {image_index: tier_id}
            tier_id 0 = lowest complexity, n_tiers-1 = highest
    """
    boundaries = np.percentile(scores, np.linspace(0, 100, n_tiers + 1))
    tier_ids   = np.digitize(scores, boundaries[1:-1])  # 0-indexed tiers
    return {int(i): int(tier_ids[i]) for i in range(len(scores))}


def tier_summary(scores: np.ndarray, tier_labels: dict, n_tiers: int = 3):
    """Print a summary table of tier boundaries and counts."""
    tier_names = ['Low', 'Medium', 'High'] if n_tiers == 3 else [str(i) for i in range(n_tiers)]
    print(f"\n{'Tier':<10} {'Name':<10} {'Count':>8} {'Score range'}")
    print('-' * 45)
    for t in range(n_tiers):
        indices = [i for i, v in tier_labels.items() if v == t]
        s = scores[indices]
        print(f"{t:<10} {tier_names[t]:<10} {len(indices):>8}   "
              f"[{s.min():.2f}, {s.max():.2f}]")
