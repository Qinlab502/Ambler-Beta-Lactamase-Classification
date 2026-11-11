"""
Training utilities for SVM and Random Forest models using K-fold cross-validation.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    classification_report, confusion_matrix
)
from datetime import datetime


def run_ml_kfold_training(
    dataset,
    fold_splits,
    model_class,
    model_name,
    device,
    model_kwargs=None,
    num_labels=5,
    out_dir="runs",
    batch_size=32,
    num_workers=12,
    seed=42,
):
    """
    Train a traditional machine learning model with K-fold cross-validation.

    Args:
        dataset: dataset instance
        fold_splits: indices for each fold
        model_class: model class (AmblerSVMModel or AmblerRFModel)
        model_name: path to the ESM2 model
        device: compute device
        model_kwargs: additional model arguments (dict)
        num_labels: number of classes
        out_dir: output directory
        batch_size: batch size
        num_workers: number of data loader workers
        seed: random seed

    Returns:
        summaries: summary metrics for each fold
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    summaries = []
    if model_kwargs is None:
        model_kwargs = {}

    for fold, (train_idx, val_idx) in enumerate(fold_splits, start=1):
        print("\n" + "=" * 80)
        print(f"Fold {fold}/{len(fold_splits)}")
        print("=" * 80)

        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )

        # Create the model.
        model = model_class(
            model_name=model_name,
            num_labels=num_labels,
            **model_kwargs
        )

        # Train (feature extraction + ML classifier fitting).
        print(f"Training {model_class.__name__}...")
        model.fit(train_loader, device)

        # Evaluate.
        print("Evaluating...")
        y_true, y_pred, y_proba = model.predict(val_loader, device)

        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average='macro')
        mcc = matthews_corrcoef(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=4)
        cm = confusion_matrix(y_true, y_pred)

        print(f"Fold {fold} Results:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Macro: {f1m:.4f}")
        print(f"  MCC: {mcc:.4f}")
        print(report)

        # Save results.
        fold_tag = f"fold{fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        fold_dir = os.path.join(out_dir, fold_tag)
        os.makedirs(fold_dir, exist_ok=True)

        # Save the model.
        model.save(os.path.join(fold_dir, "model.pkl"))

        # Save the report.
        with open(os.path.join(fold_dir, "val_report.txt"), "w", encoding="utf-8") as f:
            f.write(report)
            f.write("\n\nConfusion matrix:\n")
            f.write(np.array2string(cm))

        # Save prediction results.
        np.savez(
            os.path.join(fold_dir, "predictions.npz"),
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba
        )

        summaries.append({
            "fold": fold,
            "val_f1_macro": float(f1m),
            "val_acc": float(acc),
            "val_mcc": float(mcc),
            "checkpoint_dir": fold_dir,
        })

        # Clean up.
        del model
        torch.cuda.empty_cache()

    # Aggregate metrics.
    mean_f1 = float(np.mean([s["val_f1_macro"] for s in summaries]))
    std_f1 = float(np.std([s["val_f1_macro"] for s in summaries]))
    mean_acc = float(np.mean([s["val_acc"] for s in summaries]))
    std_acc = float(np.std([s["val_acc"] for s in summaries]))
    mean_mcc = float(np.mean([s["val_mcc"] for s in summaries]))
    std_mcc = float(np.std([s["val_mcc"] for s in summaries]))

    print("\n" + "=" * 80)
    print(f"K-fold Summary:")
    print(f"  F1 Macro: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  MCC: {mean_mcc:.4f} ± {std_mcc:.4f}")
    print("=" * 80)

    with open(os.path.join(out_dir, "all_fold_summaries.json"), "w", encoding="utf-8") as f:
        json.dump({
            "summaries": summaries,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "mean_mcc": mean_mcc,
            "std_mcc": std_mcc,
        }, f, ensure_ascii=False, indent=2)

    return summaries
