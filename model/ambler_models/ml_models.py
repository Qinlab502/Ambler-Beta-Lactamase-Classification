"""
Traditional machine learning models.
Use ESM2 embeddings with SVM and Random Forest classifiers.
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
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from tqdm import tqdm
from .backbone import ESM2Backbone
from .ml_heads import SVMHead, RFHead


class AmblerSVMModel:
    """
    ESM2 feature extraction with an SVM classifier.
    Used for Ambler classification.
    """
    def __init__(
        self,
        model_name: str,
        num_labels: int = 5,
        C: float = 1.0,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        random_state: int = 42
    ):
        self.backbone = ESM2Backbone(model_name)
        self.num_labels = num_labels
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state
        self.classifier = None

    def _extract_features(self, dataloader, device):
        """
        Extract ESM2 features.
        Returns:
            features: [N, hidden_dim]
            labels: [N]
        """
        self.backbone.to(device)
        self.backbone.eval()

        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']

                # Get ESM2 outputs with CLS and EOS removed.
                hidden_state, mask = self.backbone(input_ids, attention_mask)

                # Mean pooling
                mask_expanded = mask.unsqueeze(-1).float()
                sum_emb = (hidden_state * mask_expanded).sum(dim=1)
                valid_len = mask_expanded.sum(dim=1).clamp(min=1e-9)
                pooled = sum_emb / valid_len  # [B, hidden_dim]

                all_features.append(pooled.cpu().numpy())
                all_labels.append(labels.numpy())

        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        return features, labels

    def fit(self, train_loader, device):
        """
        Train the model.
        Args:
            train_loader: training data loader
            device: compute device
        """
        # Extract features.
        X_train, y_train = self._extract_features(train_loader, device)

        # Train the SVM.
        self.classifier = SVMHead(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            probability=True,
            random_state=self.random_state
        )
        self.classifier.fit(X_train, y_train)

        return self

    def predict(self, dataloader, device):
        """
        Make predictions.
        Returns:
            y_true, y_pred, y_proba
        """
        X_test, y_true = self._extract_features(dataloader, device)
        y_pred = self.classifier.predict(X_test)
        y_proba = self.classifier.predict_proba(X_test)

        return y_true, y_pred, y_proba

    def save(self, path: str):
        """Save the model."""
        self.classifier.save(path)

    def load(self, path: str):
        """Load the model."""
        if self.classifier is None:
            self.classifier = SVMHead()
        self.classifier.load(path)


class AmblerRFModel:
    """
    ESM2 feature extraction with a Random Forest classifier.
    Used for Ambler classification.
    """
    def __init__(
        self,
        model_name: str,
        num_labels: int = 5,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.backbone = ESM2Backbone(model_name)
        self.num_labels = num_labels
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.classifier = None

    def _extract_features(self, dataloader, device):
        """
        Extract ESM2 features.
        Returns:
            features: [N, hidden_dim]
            labels: [N]
        """
        self.backbone.to(device)
        self.backbone.eval()

        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']

                # Get ESM2 outputs with CLS and EOS removed.
                hidden_state, mask = self.backbone(input_ids, attention_mask)

                # Mean pooling
                mask_expanded = mask.unsqueeze(-1).float()
                sum_emb = (hidden_state * mask_expanded).sum(dim=1)
                valid_len = mask_expanded.sum(dim=1).clamp(min=1e-9)
                pooled = sum_emb / valid_len  # [B, hidden_dim]

                all_features.append(pooled.cpu().numpy())
                all_labels.append(labels.numpy())

        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        return features, labels

    def fit(self, train_loader, device):
        """
        Train the model.
        Args:
            train_loader: training data loader
            device: compute device
        """
        # Extract features.
        X_train, y_train = self._extract_features(train_loader, device)

        # Train the Random Forest classifier.
        self.classifier = RFHead(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        self.classifier.fit(X_train, y_train)

        return self

    def predict(self, dataloader, device):
        """
        Make predictions.
        Returns:
            y_true, y_pred, y_proba
        """
        X_test, y_true = self._extract_features(dataloader, device)
        y_pred = self.classifier.predict(X_test)
        y_proba = self.classifier.predict_proba(X_test)

        return y_true, y_pred, y_proba

    def feature_importance(self):
        """Return feature importances."""
        if self.classifier is not None:
            return self.classifier.feature_importance()
        return None

    def save(self, path: str):
        """Save the model."""
        self.classifier.save(path)

    def load(self, path: str):
        """Load the model."""
        if self.classifier is None:
            self.classifier = RFHead()
        self.classifier.load(path)
