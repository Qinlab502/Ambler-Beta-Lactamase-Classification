"""
Ensemble prediction script.
Combines MLP, CNN, LSTM, SVM and RF models for prediction.
"""
import os
import argparse
import torch
from transformers import EsmTokenizer
from ambler_models import (
    EnsemblePredictor
)


# ==================== Configuration ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_NAME = os.path.join(PROJECT_ROOT, "pretrained_model", "esm2_t33_650M_UR50D")
OUTPUT_DIR = os.path.join(BASE_DIR, "ensemble_results")

# Training output directory of each model
BASE_DIRS = {
    'mlp': os.path.join(BASE_DIR, 'runs_mlp'),
    'cnn': os.path.join(BASE_DIR, 'runs_cnn'),
    'lstm': os.path.join(BASE_DIR, 'runs_lstm'),
    'svm': os.path.join(BASE_DIR, 'runs_svm'),
    'rf': os.path.join(BASE_DIR, 'runs_rf')
}

# Label names (Ambler classification)
LABEL_NAMES = ['X', 'A', 'B', 'C', 'D']

# Ensemble method: 'soft_vote' or 'hard_vote'
ENSEMBLE_METHOD = 'soft_vote'

# Model weights (None means equal weights)
WEIGHTS = None


def predict_from_fasta(
    model_name: str,
    fasta_path: str,
    base_dirs: dict,
    output_dir: str,
    method: str = 'soft_vote',
    weights: dict = None,
    max_length: int = 512,
    batch_size: int = 32,
    num_labels: int = 5,
    label_names: list = None,
    fold_idx: int = None
):
    """
    Predict new data from a FASTA file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Ensemble Prediction from FASTA File")
    print("=" * 80)
    print(f"Input FASTA: {fasta_path}")
    print(f"Output directory: {output_dir}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = EsmTokenizer.from_pretrained(model_name)

    # Create predictor
    predictor = EnsemblePredictor(
        model_name=model_name,
        num_labels=num_labels,
        device=device,
        label_names=label_names
    )

    # Load models
    print("\nLoading models...")
    predictor.load_models_from_base_dirs(base_dirs, fold_idx=fold_idx)

    # Predict
    output_path = os.path.join(output_dir, "predictions")
    results = predictor.predict_from_fasta(
        fasta_path=fasta_path,
        tokenizer=tokenizer,
        method=method,
        weights=weights,
        max_length=max_length,
        batch_size=batch_size,
        output_path=output_path
    )

    # Print result summary
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    print(f"Total sequences: {len(results['sequence_names'])}")
    print(f"\nPrediction distribution:")
    for label, count in sorted(results['prediction_distribution'].items()):
        print(f"  {label}: {count} ({count/len(results['sequence_names'])*100:.1f}%)")

    print(f"\nResults saved to: {output_path}.json and {output_path}.tsv")

    return results


def predict_from_sequences(
    model_name: str,
    sequences: list,
    sequence_names: list,
    base_dirs: dict,
    output_dir: str,
    method: str = 'soft_vote',
    weights: dict = None,
    max_length: int = 512,
    batch_size: int = 32,
    num_labels: int = 5,
    label_names: list = None,
    fold_idx: int = None
):
    """
    Predict from a list of sequences.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Ensemble Prediction from Sequences")
    print("=" * 80)
    print(f"Number of sequences: {len(sequences)}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = EsmTokenizer.from_pretrained(model_name)

    # Create predictor
    predictor = EnsemblePredictor(
        model_name=model_name,
        num_labels=num_labels,
        device=device,
        label_names=label_names
    )

    # Load models
    print("\nLoading models...")
    predictor.load_models_from_base_dirs(base_dirs, fold_idx=fold_idx)

    # Predict
    results = predictor.predict_from_sequences(
        sequences=sequences,
        sequence_names=sequence_names,
        tokenizer=tokenizer,
        method=method,
        weights=weights,
        max_length=max_length,
        batch_size=batch_size
    )

    # Print result summary
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    print(f"Total sequences: {len(results['sequence_names'])}")
    print(f"\nPrediction distribution:")
    for label, count in sorted(results['prediction_distribution'].items()):
        print(f"  {label}: {count} ({count/len(results['sequence_names'])*100:.1f}%)")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "predictions")
    predictor._save_predictions(results, output_path)

    return results


def main():
    parser = argparse.ArgumentParser(description='Ensemble Prediction for Ambler Classification')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input FASTA file')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help='Output directory')
    parser.add_argument('--model_name', type=str, default=MODEL_NAME,
                        help='Path to the ESM2 model')
    parser.add_argument('--method', type=str, default='soft_vote',
                        choices=['soft_vote', 'hard_vote'],
                        help='Ensemble method')
    parser.add_argument('--fold', type=int, default=None,
                        help='Which fold to use (None = auto-select the best)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--mlp_dir', type=str, default=BASE_DIRS['mlp'],
                        help='MLP model directory')
    parser.add_argument('--cnn_dir', type=str, default=BASE_DIRS['cnn'],
                        help='CNN model directory')
    parser.add_argument('--lstm_dir', type=str, default=BASE_DIRS['lstm'],
                        help='LSTM model directory')
    parser.add_argument('--svm_dir', type=str, default=BASE_DIRS['svm'],
                        help='SVM model directory')
    parser.add_argument('--rf_dir', type=str, default=BASE_DIRS['rf'],
                        help='RF model directory')

    args = parser.parse_args()

    # Update model directories
    base_dirs = {
        'mlp': args.mlp_dir,
        'cnn': args.cnn_dir,
        'lstm': args.lstm_dir,
        'svm': args.svm_dir,
        'rf': args.rf_dir
    }

    # Check available models
    available_models = {}
    for name, path in base_dirs.items():
        if os.path.exists(path):
            available_models[name] = path
            print(f"✓ {name.upper()}: {path}")
        else:
            print(f"✗ {name.upper()}: {path} (not found)")
    base_dirs = available_models

    if not base_dirs:
        print("\nError: No model directories found!")
        print("Please train models first using:")
        print("  python train_mlp.py")
        print("  python train_cnn.py")
        print("  python train_lstm.py")
        print("  python train_svm.py")
        print("  python train_rf.py")
        return

    # Predict
    predict_from_fasta(
        model_name=args.model_name,
        fasta_path=args.input,
        base_dirs=base_dirs,
        output_dir=args.output,
        method=args.method,
        max_length=512,
        batch_size=args.batch_size,
        label_names=LABEL_NAMES,
        fold_idx=args.fold
    )


if __name__ == "__main__":
    main()