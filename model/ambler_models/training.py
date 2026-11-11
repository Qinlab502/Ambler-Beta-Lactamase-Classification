"""
Training utilities for model training, evaluation, and K-fold cross-validation.
"""
import os
import json
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, classification_report, confusion_matrix
from datetime import datetime


def _move_batch_to_device(batch, device):
    """Move a batch to the specified device."""
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate_epoch(model, dataloader, device):
    """
    Evaluate one epoch.

    Returns:
        loss_mean, acc, f1m, mcc, y_true, y_pred
    """
    model.eval()
    losses = []
    y_true, y_pred = [], []

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        outputs = model(**batch)
        loss = outputs['loss']
        logits = outputs['logits']

        losses.append(loss.item())
        preds = torch.argmax(logits, dim=-1)

        y_true.extend(batch["labels"].detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    loss_mean = float(np.mean(losses)) if losses else 0.0
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    f1m = f1_score(y_true, y_pred, average="macro") if y_true else 0.0
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0

    return loss_mean, acc, f1m, mcc, y_true, y_pred


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_clip=1.0):
    """
    Train for one epoch.

    Returns:
        loss_mean, acc, f1m, mcc
    """
    model.train()
    losses = []
    y_true, y_pred = [], []

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs['loss']
        logits = outputs['logits']

        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        preds = torch.argmax(logits.detach(), dim=-1)

        y_true.extend(batch["labels"].detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    loss_mean = float(np.mean(losses)) if losses else 0.0
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    f1m = f1_score(y_true, y_pred, average="macro") if y_true else 0.0
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0

    return loss_mean, acc, f1m, mcc


def run_kfold_training(
    dataset,
    fold_splits,
    model_class,
    model_name,
    device,
    model_kwargs=None,
    num_labels=5,
    out_dir="runs",
    epochs=50,
    batch_size=32,
    num_workers=12,
    lr=1e-4,
    max_patience=10,
    weight_decay=0.01,
    warmup_ratio=0.1,
    grad_clip=1.0,
    seed=42,
):
    """
    Train with K-fold cross-validation.

    Args:
        dataset: dataset instance
        fold_splits: indices for each fold
        model_class: model class (AmblerMLPModel, AmblerCNNModel, or AmblerLSTMModel)
        model_name: path to the ESM2 model
        device: training device
        model_kwargs: additional model arguments (dict)
        num_labels: number of classes
        out_dir: output directory
        epochs: number of training epochs
        batch_size: batch size
        num_workers: number of data loader workers
        lr: learning rate
        max_patience: early stopping patience
        weight_decay: weight decay
        warmup_ratio: warmup ratio
        grad_clip: gradient clipping threshold
        seed: random seed

    Returns:
        summaries: summary metrics for each fold
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    summaries = []
    if model_kwargs is None:
        model_kwargs = {}

    labels = np.array(dataset.labels, dtype=np.int64)

    for fold, (train_idx, val_idx) in enumerate(fold_splits, start=1):
        print("\n" + "=" * 80)
        print(f"Fold {fold}/{len(fold_splits)}")
        print("=" * 80)

        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )

        model = model_class(
            model_name=model_name,
            num_labels=num_labels,
            **model_kwargs
        ).to(device)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

        total_steps = epochs * len(train_loader)
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        best_f1 = -1.0
        best_state = None
        history = []
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc, tr_f1, tr_mcc = train_epoch(
                model, train_loader, optimizer, scheduler, device, grad_clip=grad_clip
            )
            va_loss, va_acc, va_f1, va_mcc, _, _ = evaluate_epoch(model, val_loader, device)

            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[Fold {fold} | Epoch {epoch:02d}/{epochs:02d}] "
                f"lr={lr_now:.2e} | "
                f"train: loss={tr_loss:.4f} acc={tr_acc:.4f} f1m={tr_f1:.4f} mcc={tr_mcc:.4f} | "
                f"val: loss={va_loss:.4f} acc={va_acc:.4f} f1m={va_f1:.4f} mcc={va_mcc:.4f}"
            )

            history.append({
                "epoch": epoch,
                "lr": lr_now,
                "train": {"loss": tr_loss, "acc": tr_acc, "f1_macro": tr_f1, "mcc": tr_mcc},
                "val": {"loss": va_loss, "acc": va_acc, "f1_macro": va_f1, "mcc": va_mcc},
            })

            if va_f1 > best_f1:
                best_f1 = va_f1
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                print(f"  --> Best Model saved (val_f1: {best_f1:.4f})")
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                print(f"\n[Fold {fold}] Early stopping at epoch {epoch}. Best Val F1: {best_f1:.4f}")
                break

        # Save fold results.
        model.load_state_dict(best_state)
        va_loss, va_acc, va_f1, va_mcc, y_true, y_pred = evaluate_epoch(model, val_loader, device)
        report = classification_report(y_true, y_pred, digits=4)
        cm = confusion_matrix(y_true, y_pred)

        fold_tag = f"fold{fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        fold_dir = os.path.join(out_dir, fold_tag)
        os.makedirs(fold_dir, exist_ok=True)

        torch.save(best_state, os.path.join(fold_dir, "best_model.pt"))
        with open(os.path.join(fold_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        with open(os.path.join(fold_dir, "val_report.txt"), "w", encoding="utf-8") as f:
            f.write(report)
            f.write("\n\nConfusion matrix:\n")
            f.write(np.array2string(cm))

        print("\n[Best checkpoint VAL report]")
        print(report)
        print("Confusion matrix:\n", cm)

        summaries.append({
            "fold": fold,
            "best_val_f1_macro": float(best_f1),
            "best_checkpoint_dir": fold_dir,
            "best_val": {"loss": float(va_loss), "acc": float(va_acc), "f1_macro": float(va_f1), "mcc": float(va_mcc)},
        })

        # Clear GPU memory.
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    mean_f1 = float(np.mean([s["best_val_f1_macro"] for s in summaries]))
    std_f1 = float(np.std([s["best_val_f1_macro"] for s in summaries]))
    print("\n" + "=" * 80)
    print(f"K-fold mean best val f1_macro: {mean_f1:.4f} ± {std_f1:.4f}")
    print("=" * 80)

    with open(os.path.join(out_dir, "all_fold_summaries.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    return summaries
