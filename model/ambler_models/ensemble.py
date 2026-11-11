"""
Ensemble prediction module.
Combines MLP, CNN, LSTM, SVM, and RF models using soft or hard voting.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class EnsemblePredictor:
    """
    Ensemble predictor for MLP, CNN, LSTM, SVM, and RF models.
    """
    def __init__(
        self,
        model_name: str,
        num_labels: int = 5,
        device: torch.device = None,
        label_names: list = None
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_names = label_names or ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']

        # Model instances.
        self.mlp_model = None
        self.cnn_model = None
        self.lstm_model = None
        self.svm_model = None
        self.rf_model = None

        # Names of the loaded models.
        self.loaded_models = []

    def load_models(
        self,
        mlp_path: str = None,
        cnn_path: str = None,
        lstm_path: str = None,
        svm_path: str = None,
        rf_path: str = None
    ):
        """
        Load all available models.
        """
        from ambler_models import (
            AmblerMLPModel, AmblerCNNModel, AmblerLSTMModel,
            AmblerSVMModel, AmblerRFModel
        )

        self.loaded_models = []

        # Load deep learning models.
        if mlp_path:
            model_file = self._get_model_file(mlp_path, 'best_model.pt')
            if model_file and os.path.exists(model_file):
                self.mlp_model = AmblerMLPModel(
                    model_name=self.model_name,
                    num_labels=self.num_labels
                ).to(self.device)
                self.mlp_model.load_state_dict(
                    torch.load(model_file, map_location=self.device, weights_only=True)
                )
                self.mlp_model.eval()
                self.loaded_models.append('mlp')
                print(f"✓ Loaded MLP model from {model_file}")
            else:
                print(f"✗ MLP model not found: {mlp_path}")

        if cnn_path:
            model_file = self._get_model_file(cnn_path, 'best_model.pt')
            if model_file and os.path.exists(model_file):
                self.cnn_model = AmblerCNNModel(
                    model_name=self.model_name,
                    num_labels=self.num_labels
                ).to(self.device)
                self.cnn_model.load_state_dict(
                    torch.load(model_file, map_location=self.device, weights_only=True)
                )
                self.cnn_model.eval()
                self.loaded_models.append('cnn')
                print(f"✓ Loaded CNN model from {model_file}")
            else:
                print(f"✗ CNN model not found: {cnn_path}")

        if lstm_path:
            model_file = self._get_model_file(lstm_path, 'best_model.pt')
            if model_file and os.path.exists(model_file):
                self.lstm_model = AmblerLSTMModel(
                    model_name=self.model_name,
                    num_labels=self.num_labels
                ).to(self.device)
                self.lstm_model.load_state_dict(
                    torch.load(model_file, map_location=self.device, weights_only=True)
                )
                self.lstm_model.eval()
                self.loaded_models.append('lstm')
                print(f"✓ Loaded LSTM model from {model_file}")
            else:
                print(f"✗ LSTM model not found: {lstm_path}")

        # Load traditional machine learning models.
        if svm_path:
            model_file = self._get_model_file(svm_path, 'model.pkl')
            if model_file and os.path.exists(model_file):
                self.svm_model = AmblerSVMModel(
                    model_name=self.model_name,
                    num_labels=self.num_labels
                )
                self.svm_model.load(model_file)
                self.loaded_models.append('svm')
                print(f"✓ Loaded SVM model from {model_file}")
            else:
                print(f"✗ SVM model not found: {svm_path}")

        if rf_path:
            model_file = self._get_model_file(rf_path, 'model.pkl')
            if model_file and os.path.exists(model_file):
                self.rf_model = AmblerRFModel(
                    model_name=self.model_name,
                    num_labels=self.num_labels
                )
                self.rf_model.load(model_file)
                self.loaded_models.append('rf')
                print(f"✓ Loaded RF model from {model_file}")
            else:
                print(f"✗ RF model not found: {rf_path}")

        print(f"\nTotal models loaded: {len(self.loaded_models)} ({', '.join(self.loaded_models)})")

        if not self.loaded_models:
            raise ValueError("No models loaded! Please check model paths.")

    def _get_model_file(self, path: str, model_name: str) -> str:
        """Resolve the model file path."""
        if os.path.isfile(path):
            return path
        elif os.path.isdir(path):
            return os.path.join(path, model_name)
        return None

    def load_models_from_base_dirs(self, base_dirs: dict, fold_idx: int = None):
        """
        Load models from their base directories.
        """
        model_paths = {}

        for model_type, base_dir in base_dirs.items():
            if not os.path.exists(base_dir):
                print(f"Warning: {base_dir} not found")
                continue

            if fold_idx is not None:
                fold_dirs = [d for d in os.listdir(base_dir) if d.startswith(f'fold{fold_idx}')]
            else:
                summary_path = os.path.join(base_dir, "all_fold_summaries.json")
                if os.path.exists(summary_path):
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)
                    if 'summaries' in summary:
                        best_fold = max(summary['summaries'],
                                        key=lambda x: x.get('best_val_f1_macro', x.get('val_f1_macro', 0)))
                    else:
                        best_fold = max(summary,
                                        key=lambda x: x.get('best_val_f1_macro', x.get('val_f1_macro', 0)))
                    fold_num = best_fold['fold']
                    fold_dirs = [d for d in os.listdir(base_dir) if d.startswith(f'fold{fold_num}')]
                else:
                    fold_dirs = [d for d in os.listdir(base_dir) if d.startswith('fold')]

            if fold_dirs:
                model_paths[model_type] = os.path.join(base_dir, sorted(fold_dirs)[0])

        self.load_models(
            mlp_path=model_paths.get('mlp'),
            cnn_path=model_paths.get('cnn'),
            lstm_path=model_paths.get('lstm'),
            svm_path=model_paths.get('svm'),
            rf_path=model_paths.get('rf')
        )

        return model_paths

    def _get_dl_model_probs(self, model, dataloader, show_progress=True):
        """Get prediction probabilities from a deep learning model."""
        all_probs = []

        iterator = tqdm(dataloader, desc="DL model inference", disable=not show_progress)
        with torch.no_grad():
            for batch in iterator:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = model(input_ids, attention_mask)
                logits = outputs['logits']
                probs = torch.softmax(logits, dim=-1)

                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def _get_ml_model_probs(self, model, dataloader, show_progress=True):
        """Get prediction probabilities from a traditional ML model."""
        _, _, probs = model.predict(dataloader, self.device)
        return probs

    def predict_ensemble(
        self,
        dataloader,
        method: str = 'soft_vote',
        weights: dict = None,
        show_progress: bool = True,
        return_individual: bool = True
    ):
        """
        Make ensemble predictions.
        """
        model_probs = {}
        model_preds = {}

        # Collect predictions from each available model.
        if self.mlp_model is not None:
            probs = self._get_dl_model_probs(self.mlp_model, dataloader, show_progress)
            model_probs['mlp'] = probs
            model_preds['mlp'] = np.argmax(probs, axis=-1)
            if show_progress:
                print(f"MLP predictions done")

        if self.cnn_model is not None:
            probs = self._get_dl_model_probs(self.cnn_model, dataloader, show_progress)
            model_probs['cnn'] = probs
            model_preds['cnn'] = np.argmax(probs, axis=-1)
            if show_progress:
                print(f"CNN predictions done")

        if self.lstm_model is not None:
            probs = self._get_dl_model_probs(self.lstm_model, dataloader, show_progress)
            model_probs['lstm'] = probs
            model_preds['lstm'] = np.argmax(probs, axis=-1)
            if show_progress:
                print(f"LSTM predictions done")

        if self.svm_model is not None:
            probs = self._get_ml_model_probs(self.svm_model, dataloader, show_progress)
            model_probs['svm'] = probs
            model_preds['svm'] = np.argmax(probs, axis=-1)
            if show_progress:
                print(f"SVM predictions done")

        if self.rf_model is not None:
            probs = self._get_ml_model_probs(self.rf_model, dataloader, show_progress)
            model_probs['rf'] = probs
            model_preds['rf'] = np.argmax(probs, axis=-1)
            if show_progress:
                print(f"RF predictions done")

        if not model_probs:
            raise ValueError("No models loaded!")

        # Use equal weights by default.
        if weights is None:
            weights = {name: 1.0 for name in model_probs.keys()}

        # Combine model predictions.
        if method == 'soft_vote':
            total_weight = sum(weights.get(name, 1.0) for name in model_probs.keys())
            ensemble_proba = np.zeros_like(list(model_probs.values())[0])
            for name, probs in model_probs.items():
                w = weights.get(name, 1.0)
                ensemble_proba += w * probs
            ensemble_proba /= total_weight
            y_pred = np.argmax(ensemble_proba, axis=-1)

        elif method == 'hard_vote':
            all_preds = np.stack([model_preds[name] for name in model_preds.keys()], axis=0)
            num_samples = all_preds.shape[1]
            num_classes = self.num_labels

            y_pred = np.zeros(num_samples, dtype=np.int64)
            ensemble_proba = np.zeros((num_samples, num_classes), dtype=np.float64)

            total_weight = sum(weights.get(name, 1.0) for name in model_preds.keys())

            for i in range(num_samples):
                votes = {}
                for j, name in enumerate(model_preds.keys()):
                    pred = all_preds[j, i]
                    w = weights.get(name, 1.0)
                    votes[pred] = votes.get(pred, 0) + w

                y_pred[i] = max(votes.keys(), key=lambda x: votes[x])

                for cls in range(num_classes):
                    ensemble_proba[i, cls] = votes.get(cls, 0) / total_weight

        else:
            raise ValueError(f"Unknown method: {method}")

        model_results = None
        if return_individual:
            model_results = {
                name: {
                    'probs': model_probs[name],
                    'preds': model_preds[name]
                }
                for name in model_probs.keys()
            }

        return y_pred, ensemble_proba, model_results

    def predict_from_fasta(
        self,
        fasta_path: str,
        tokenizer,
        method: str = 'soft_vote',
        weights: dict = None,
        max_length: int = 512,
        batch_size: int = 32,
        num_workers: int = 4,
        output_path: str = None
    ):
        """
        Make predictions from a FASTA file.
        """
        from ambler_models import get_fasta_dict, PredictDataset

        print(f"Loading sequences from {fasta_path}...")
        fasta_dict = get_fasta_dict(fasta_path)
        sequence_names = list(fasta_dict.keys())
        sequences = list(fasta_dict.values())

        dataset = PredictDataset(sequences, sequence_names, tokenizer, max_length=max_length)

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )

        print(f"Predicting {len(dataset)} sequences...")
        print(f"Using models: {', '.join(self.loaded_models)}")
        print(f"Ensemble method: {method}")

        y_pred, y_proba, model_results = self.predict_ensemble(
            dataloader, method=method, weights=weights,
            show_progress=True, return_individual=True
        )

        results = {
            'sequence_names': sequence_names,
            'sequences': sequences,
            'predictions': y_pred.tolist(),
            'predicted_labels': [self.label_names[p] for p in y_pred],
        }

        if y_proba is not None:
            results['probabilities'] = y_proba.tolist()
            results['confidence'] = y_proba.max(axis=1).tolist()

        if model_results:
            results['individual_predictions'] = {
                name: {
                    'predictions': model_results[name]['preds'].tolist(),
                    'predicted_labels': [self.label_names[p] for p in model_results[name]['preds']]
                }
                for name in model_results.keys()
            }

        pred_counts = {}
        for p in y_pred:
            label = self.label_names[p]
            pred_counts[label] = pred_counts.get(label, 0) + 1
        results['prediction_distribution'] = pred_counts

        if output_path:
            self._save_predictions(results, output_path)

        return results

    def predict_from_sequences(
        self,
        sequences: list,
        sequence_names: list = None,
        tokenizer=None,
        method: str = 'soft_vote',
        weights: dict = None,
        max_length: int = 512,
        batch_size: int = 32
    ):
        """
        Make predictions from a list of sequences.
        """
        from ambler_models import PredictDataset

        if sequence_names is None:
            sequence_names = [f"seq_{i}" for i in range(len(sequences))]

        dataset = PredictDataset(sequences, sequence_names, tokenizer, max_length=max_length)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available()
        )

        print(f"Predicting {len(sequences)} sequences...")
        y_pred, y_proba, model_results = self.predict_ensemble(
            dataloader, method=method, weights=weights,
            show_progress=True, return_individual=True
        )

        results = {
            'sequence_names': sequence_names,
            'sequences': sequences,
            'predictions': y_pred.tolist(),
            'predicted_labels': [self.label_names[p] for p in y_pred],
        }

        if y_proba is not None:
            results['probabilities'] = y_proba.tolist()
            results['confidence'] = y_proba.max(axis=1).tolist()

        if model_results:
            results['individual_predictions'] = {
                name: {
                    'predictions': model_results[name]['preds'].tolist(),
                    'predicted_labels': [self.label_names[p] for p in model_results[name]['preds']]
                }
                for name in model_results.keys()
            }

        pred_counts = {}
        for p in y_pred:
            label = self.label_names[p]
            pred_counts[label] = pred_counts.get(label, 0) + 1
        results['prediction_distribution'] = pred_counts

        return results

    def _save_predictions(self, results: dict, output_path: str):
        """Save prediction results."""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        json_path = output_path if output_path.endswith('.json') else output_path + '.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {json_path}")

        tsv_path = json_path.replace('.json', '.tsv')
        with open(tsv_path, 'w', encoding='utf-8') as f:
            headers = ['Sequence_Name', 'Predicted_Class', 'Predicted_Label', 'Confidence']
            f.write('\t'.join(headers) + '\n')

            for i, name in enumerate(results['sequence_names']):
                pred_class = results['predictions'][i]
                pred_label = results['predicted_labels'][i]
                confidence = results.get('confidence', [1.0] * len(results['sequence_names']))[i]
                f.write(f"{name}\t{pred_class}\t{pred_label}\t{confidence:.4f}\n")
        print(f"Results saved to {tsv_path}")


def find_best_fold_models(base_dirs: dict):
    """Find the best fold directory for each model."""
    best_models = {}

    for model_type, base_dir in base_dirs.items():
        if not os.path.exists(base_dir):
            print(f"Warning: {base_dir} not found")
            continue

        fold_dirs = [d for d in os.listdir(base_dir) if d.startswith('fold')]
        if not fold_dirs:
            print(f"Warning: No fold directories in {base_dir}")
            continue

        summary_path = os.path.join(base_dir, "all_fold_summaries.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)

            if 'summaries' in summary:
                best_fold = max(summary['summaries'],
                                key=lambda x: x.get('best_val_f1_macro', x.get('val_f1_macro', 0)))
            else:
                best_fold = max(summary,
                                key=lambda x: x.get('best_val_f1_macro', x.get('val_f1_macro', 0)))
            fold_num = best_fold['fold']

            for fold_dir in fold_dirs:
                if fold_dir.startswith(f'fold{fold_num}'):
                    best_models[model_type] = os.path.join(base_dir, fold_dir)
                    break
        else:
            best_models[model_type] = os.path.join(base_dir, sorted(fold_dirs)[0])

        print(f"Best {model_type}: {best_models.get(model_type, 'N/A')}")

    return best_models