"""
Train a Random Forest model for Ambler classification.
Uses ESM2 embeddings + Random Forest classifier.
"""
import os
import torch
from ambler_models import (
    AmblerRFModel,
    prepare_data,
    run_ml_kfold_training
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)


def main():
    # Configuration
    MODEL_NAME = os.path.join(PROJECT_ROOT, "pretrained_model", "esm2_t33_650M_UR50D")
    FASTA_PATH = os.path.join(PROJECT_ROOT, "data", "ambler_dataset.fasta")
    OUT_DIR = os.path.join(BASE_DIR, "runs_rf")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")
    print("=" * 80)
    print("Training Random Forest Model for Ambler Classification")
    print("=" * 80)

    # Prepare data
    dataset, fold_splits, tokenizer = prepare_data(
        fasta_path=FASTA_PATH,
        model_name=MODEL_NAME,
        max_length=512,
        n_splits=5,
        random_state=42
    )

    # Train
    summaries = run_ml_kfold_training(
        dataset=dataset,
        fold_splits=fold_splits,
        model_class=AmblerRFModel,
        model_name=MODEL_NAME,
        device=DEVICE,
        model_kwargs={
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "n_jobs": -1,
        },
        num_labels=5,
        out_dir=OUT_DIR,
        batch_size=32,
        seed=42
    )

    print("\nTraining completed!")
    print(f"Results saved to: {OUT_DIR}")

    return summaries


if __name__ == "__main__":
    main()
