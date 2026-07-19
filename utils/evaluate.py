import os
import torch
import torch.nn as nn
import numpy as np


# ── CIFAR-100 superclass mapping ──────────────────────────────────────────────
# Maps each of the 20 superclasses to its 5 fine-grained class names.
CIFAR100_SUPERCLASSES = {
    'aquatic mammals':               ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish':                          ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers':                       ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food containers':               ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit & vegetables':            ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household electrical devices':  ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household furniture':           ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects':                       ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large carnivores':              ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large outdoor man-made':        ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large outdoor natural':         ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large herbivores/omnivores':    ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium mammals':                ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect invertebrates':      ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people':                        ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles':                      ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small mammals':                 ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees':                         ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles 1':                    ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles 2':                    ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}


def build_class_to_superclass(class_names: list) -> dict:
    """
    Return {class_idx: superclass_name} using the dataset's class list.
    class_names should be dataset.classes (alphabetically sorted CIFAR-100 names).
    """
    name_to_super = {
        fine: sup
        for sup, fines in CIFAR100_SUPERCLASSES.items()
        for fine in fines
    }
    return {idx: name_to_super[name] for idx, name in enumerate(class_names)
            if name in name_to_super}


# ── Checkpoint helper ─────────────────────────────────────────────────────────

def load_best_checkpoint(model: nn.Module, model_name: str, cfg: dict, device) -> float:
    """
    Load best-val-acc weights into model in-place.
    Returns the val_acc1 stored in the checkpoint.
    """
    ckpt_path = os.path.join(cfg['drive_base'], 'checkpoints', f'{model_name}_best.pth')
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck['state_dict'])
    model.to(device).eval()
    return ck['val_acc1']


# ── Accuracy functions ────────────────────────────────────────────────────────

def per_class_accuracy(
    model: nn.Module,
    test_loader,
    device,
    num_classes: int = 100,
) -> np.ndarray:
    """
    Return array of shape (num_classes,) with top-1 accuracy (0–100) per class.
    """
    correct = np.zeros(num_classes, dtype=int)
    total   = np.zeros(num_classes, dtype=int)

    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs).argmax(1)
            for t, p in zip(targets, preds):
                total[t.item()]   += 1
                correct[t.item()] += int(t == p)

    return np.where(total > 0, 100.0 * correct / total, 0.0)


def per_tier_accuracy(
    model: nn.Module,
    test_loader,
    device,
    tier_labels: dict,
    n_tiers: int = 3,
) -> dict:
    """
    Top-1 accuracy broken down by image-complexity tier.

    tier_labels: {image_index: tier_id}  tier_id in {0, 1, ..., n_tiers-1}
    Returns: {tier_id: accuracy_percent}
    """
    correct = {t: 0 for t in range(n_tiers)}
    total   = {t: 0 for t in range(n_tiers)}
    idx     = 0

    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs).argmax(1)
            for p, t in zip(preds, targets):
                tier = tier_labels.get(idx, -1)
                if tier >= 0:
                    total[tier]   += 1
                    correct[tier] += int(p == t)
                idx += 1

    return {
        tier: (100.0 * correct[tier] / total[tier] if total[tier] > 0 else 0.0)
        for tier in range(n_tiers)
    }


def per_superclass_accuracy(
    model: nn.Module,
    test_loader,
    device,
    class_to_superclass: dict,
) -> dict:
    """
    Top-1 accuracy grouped by CIFAR-100 superclass (20 groups of 5 classes).

    class_to_superclass: {class_idx: superclass_name}
    Returns: {superclass_name: accuracy_percent}
    """
    correct = {}
    total   = {}

    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs).argmax(1)
            for p, t in zip(preds, targets):
                sup = class_to_superclass.get(t.item(), 'unknown')
                total[sup]   = total.get(sup, 0) + 1
                correct[sup] = correct.get(sup, 0) + int(p == t)

    return {
        sup: (100.0 * correct[sup] / total[sup] if total[sup] > 0 else 0.0)
        for sup in total
    }
