"""
Data processing module.
"""
import torch
from transformers import EsmTokenizer
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from collections import Counter
import numpy as np


def get_fasta_dict(fasta_file):
    """Read a FASTA file and return a {header: sequence} dict."""
    fasta_dict = {}
    with open(fasta_file, 'r') as infile:
        for line in infile:
            if line.startswith(">"):
                head = line.strip().replace(">", "")
                fasta_dict[head] = ''
            else:
                fasta_dict[head] += line.replace("\n", "")
    return fasta_dict


def get_label(fasta_dict):
    """Extract the Ambler class label from the FASTA header."""
    label_map = {
        "X": 0,
        "A": 1,
        "B1": 2,
        "B2": 2,
        "B3": 2,
        "C": 3,
        "D": 4,
    }

    labels = []
    for header in fasta_dict.keys():
        last_field = header.rsplit("|", 1)[-1]
        label = label_map.get(last_field, label_map["X"])
        labels.append(label)

    return labels


class SeqDataset(torch.utils.data.Dataset):
    """Sequence dataset with labels (used for training and evaluation)."""
    def __init__(self, fasta_dict, tokenizer, max_length=512):
        super().__init__()
        self.fasta_dict = fasta_dict
        self.names = [name for name in fasta_dict.keys()]
        self.labels = get_label(fasta_dict)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        seq_name = self.names[idx]
        selected_seq = self.fasta_dict[seq_name]
        label = self.labels[idx]

        encoded = self.tokenizer(
            selected_seq,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.names)


class PredictDataset(torch.utils.data.Dataset):
    """Prediction dataset without labels (used for inference only)."""
    def __init__(self, sequences, sequence_names, tokenizer, max_length=512):
        super().__init__()
        self.sequences = sequences
        self.names = sequence_names
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        encoded = self.tokenizer(
            seq,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(0, dtype=torch.long)  # placeholder, unused during prediction
        }

    def __len__(self):
        return len(self.sequences)


def prepare_data(fasta_path, model_name, max_length=512, n_splits=5, random_state=42):
    """
    Prepare the dataset and data loaders.

    Args:
        fasta_path: path to the FASTA file
        model_name: path to the ESM2 model
        max_length: maximum sequence length
        n_splits: number of folds for K-fold cross-validation
        random_state: random seed

    Returns:
        dataset: SeqDataset instance
        fold_splits: list of index arrays for the K folds
        tokenizer: EsmTokenizer instance
    """
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    fasta_dict = get_fasta_dict(fasta_path)
    dataset = SeqDataset(fasta_dict, tokenizer, max_length=max_length)

    labels = np.array(dataset.labels, dtype=np.int64)
    indices = np.arange(len(dataset))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_splits = list(skf.split(indices, labels))

    print(f"Dataset size: {len(dataset)}")
    print(f"Fold sizes: {[(len(tr), len(va)) for tr, va in fold_splits]}")

    for fold, (train_idx, val_idx) in enumerate(fold_splits, start=1):
        c_tr = Counter(labels[train_idx].tolist())
        c_va = Counter(labels[val_idx].tolist())
        print(f"Fold {fold} | Train: {len(train_idx)} | Val: {len(val_idx)}")
        print(f"  Train Dist: {[c_tr.get(i, 0) for i in range(5)]}")
        print(f"  Val   Dist: {[c_va.get(i, 0) for i in range(5)]}")

    return dataset, fold_splits, tokenizer
