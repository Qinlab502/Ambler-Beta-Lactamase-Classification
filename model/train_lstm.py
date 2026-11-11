"""
Train an LSTM model for Ambler classification.
"""
import os
import torch
from ambler_models import (
    AmblerLSTMModel,
    prepare_data,
    run_kfold_training
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)


def main():
    # Configuration
    MODEL_NAME = os.path.join(PROJECT_ROOT, "pretrained_model", "esm2_t33_650M_UR50D")
    FASTA_PATH = os.path.join(PROJECT_ROOT, "data", "ambler_dataset.fasta")
    OUT_DIR = os.path.join(BASE_DIR, "runs_lstm")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")
    print("=" * 80)
    print("Training LSTM Model for Ambler Classification")
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
    summaries = run_kfold_training(
        dataset=dataset,
        fold_splits=fold_splits,
        model_class=AmblerLSTMModel,
        model_name=MODEL_NAME,
        device=DEVICE,
        model_kwargs={
            "lstm_hidden": 256,
            "num_layers": 2,
            "dropout": 0.3,
            "bidirectional": True
        },
        num_labels=5,
        out_dir=OUT_DIR,
        epochs=50,
        batch_size=16,
        lr=1e-4,
        warmup_ratio=0.1,
        max_patience=10,
        seed=42
    )

    print("\nTraining completed!")
    print(f"Results saved to: {OUT_DIR}")

    return summaries


if __name__ == "__main__":
    main()
