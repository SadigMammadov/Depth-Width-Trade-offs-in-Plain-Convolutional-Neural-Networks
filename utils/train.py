import os
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from tqdm.auto import tqdm


# ── Utilities ─────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_budget(model: nn.Module, name: str, cfg: dict) -> tuple:
    """Print and return (n_params, within_budget) for this model."""
    n  = count_params(model)
    lo = cfg['param_budget'] * (1 - cfg['param_tol'])
    hi = cfg['param_budget'] * (1 + cfg['param_tol'])
    ok = lo <= n <= hi
    tag = 'OK' if ok else 'OUT OF RANGE'
    print(f'{name:<8}  {n:>10,} params  [{tag}]  '
          f"(target {cfg['param_budget']:,} +/-{cfg['param_tol']*100:.0f}%)")
    return n, ok


def is_training_complete(model_name: str, cfg: dict) -> bool:
    """Return True if a full training log already exists for this model."""
    log_csv = os.path.join(cfg['drive_base'], 'logs', f'{model_name}_log.csv')
    if not os.path.exists(log_csv):
        return False
    try:
        return len(pd.read_csv(log_csv)) >= cfg['epochs']
    except Exception:
        return False


# ── Inner loop helpers ────────────────────────────────────────────────────────

def _train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = correct = total = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch:03d}/{total_epochs}',
                leave=True, unit='batch')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        out  = model(inputs)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct    += out.argmax(1).eq(targets).sum().item()
        total      += inputs.size(0)
        pbar.set_postfix(loss=f'{total_loss/total:.4f}', acc=f'{100*correct/total:.1f}%')
    return total_loss / total, 100.0 * correct / total


def _eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct1 = correct5 = total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            out = model(inputs)
            total_loss += criterion(out, targets).item() * inputs.size(0)
            total      += inputs.size(0)
            correct1   += out.argmax(1).eq(targets).sum().item()
            top5 = out.topk(5, dim=1).indices
            correct5   += top5.eq(targets.unsqueeze(1)).any(dim=1).sum().item()
    return total_loss / total, 100.0 * correct1 / total, 100.0 * correct5 / total


def _capture_grad_norms(model) -> dict:
    """L2 norm of each parameter's gradient after the most recent backward pass."""
    return {
        name: param.grad.norm(2).item()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


# ── Main training function ────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    model_name: str,
    train_loader,
    test_loader,
    device,
    cfg: dict,
    resume: bool = True,
    log_every: int = 10,
    ckpt_every: int = 50,
) -> tuple:
    """
    Full training run for one model with resume support and incremental saves.

    Drive output layout:
      checkpoints/<model_name>_best.pth     best val-acc weights
      checkpoints/<model_name>_latest.pth   periodic snapshot for resume
      logs/<model_name>_log.csv             per-epoch metrics (written every log_every epochs)
      logs/<model_name>_gradnorms.json      gradient norms (written at end)

    Args:
        resume:     If True, try to resume from _latest.pth if it exists.
        log_every:  Write log CSV to Drive every N epochs (guards against disconnects).
        ckpt_every: Save _latest.pth for resume every N epochs.

    Returns:
        (log, best_acc)  where log is a list of per-epoch metric dicts.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    ckpt_dir    = os.path.join(cfg['drive_base'], 'checkpoints')
    log_dir     = os.path.join(cfg['drive_base'], 'logs')
    best_path   = os.path.join(ckpt_dir, f'{model_name}_best.pth')
    latest_path = os.path.join(ckpt_dir, f'{model_name}_latest.pth')
    log_csv     = os.path.join(log_dir,  f'{model_name}_log.csv')
    log_json    = os.path.join(log_dir,  f'{model_name}_gradnorms.json')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    best_acc    = 0.0
    log         = []   # per-epoch metric rows (no grad norms — those go to grad_log)
    grad_log    = []   # per-epoch gradient norm dicts
    start_epoch = 1

    # ── Attempt resume from latest checkpoint ──
    if resume and os.path.exists(latest_path):
        try:
            ck = torch.load(latest_path, map_location=device)
            model.load_state_dict(ck['state_dict'])
            optimizer.load_state_dict(ck['optimizer_state'])
            scheduler.load_state_dict(ck['scheduler_state'])
            start_epoch = ck['epoch'] + 1
            best_acc    = ck.get('best_acc', 0.0)
            if os.path.exists(log_csv):
                log = pd.read_csv(log_csv).to_dict('records')
            print(f'Resumed {model_name} from epoch {ck["epoch"]} '
                  f'(best val acc so far: {best_acc:.2f}%)')
        except Exception as e:
            print(f'Resume failed ({e}) — starting {model_name} from scratch.')
            start_epoch = 1
            log = []

    if start_epoch > cfg['epochs']:
        print(f'{model_name} already fully trained ({cfg["epochs"]} epochs).')
        return log, best_acc

    print(f"\n{'='*64}")
    print(f"  {model_name}   ({count_params(model):,} params)   "
          f"epochs {start_epoch} -> {cfg['epochs']}")
    print(f"{'='*64}")

    for epoch in range(start_epoch, cfg['epochs'] + 1):
        t0 = time.time()

        tr_loss, tr_acc              = _train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, cfg['epochs'])
        grad_norms                   = _capture_grad_norms(model)
        val_loss, val_acc1, val_acc5 = _eval_one_epoch(model, test_loader, criterion, device)
        scheduler.step()

        # Best checkpoint
        is_best = val_acc1 > best_acc
        if is_best:
            best_acc = val_acc1
            torch.save({
                'epoch': epoch, 'model_name': model_name,
                'state_dict': model.state_dict(), 'val_acc1': val_acc1,
            }, best_path)

        row = dict(
            epoch=epoch,
            tr_loss=round(tr_loss,   4), tr_acc=round(tr_acc,    3),
            val_loss=round(val_loss,  4), val_acc1=round(val_acc1, 3),
            val_acc5=round(val_acc5, 3), time_s=round(time.time() - t0, 1),
        )
        log.append(row)
        grad_log.append({'epoch': epoch, 'grad_norms': grad_norms})

        best_marker = ' *' if is_best else ''
        print(f"Ep {epoch:3d}/{cfg['epochs']}  "
              f"TrL {tr_loss:.4f} TrAcc {tr_acc:.1f}%  "
              f"ValL {val_loss:.4f} Val1 {val_acc1:.1f}% Val5 {val_acc5:.1f}%  "
              f"{row['time_s']}s{best_marker}")

        # Incremental Drive saves (protects against Colab disconnect)
        if epoch % log_every == 0 or epoch == cfg['epochs']:
            pd.DataFrame(log).to_csv(log_csv, index=False)

        if epoch % ckpt_every == 0:
            torch.save({
                'epoch':           epoch,
                'model_name':      model_name,
                'state_dict':      model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_acc':        best_acc,
            }, latest_path)

    # Final saves
    pd.DataFrame(log).to_csv(log_csv, index=False)
    with open(log_json, 'w') as f:
        json.dump(grad_log, f)

    print(f"\nBest val acc : {best_acc:.2f}%")
    print(f"Checkpoint   : {best_path}")
    print(f"Log          : {log_csv}")
    return log, best_acc
