"""
Ambler classification model package.
Includes MLP, CNN, and LSTM deep learning models, SVM and RF traditional
machine learning models, and ensemble prediction.

Usage:
    # Deep learning models
    from ambler_models import AmblerMLPModel, AmblerCNNModel, AmblerLSTMModel

    # Traditional machine learning models
    from ambler_models import AmblerSVMModel, AmblerRFModel

    # Ensemble prediction
    from ambler_models import EnsemblePredictor

    # Data processing
    from ambler_models import prepare_data, SeqDataset, PredictDataset

    # Training
    from ambler_models import run_kfold_training, run_ml_kfold_training
"""

from .backbone import ESM2Backbone, ESM2BackboneSimple
from .mlp_head import MLPHead
from .cnn_head import CNNHead
from .lstm_head import LSTMHead
from .ml_heads import SVMHead, RFHead
from .mlp_model import AmblerMLPModel
from .cnn_model import AmblerCNNModel
from .lstm_model import AmblerLSTMModel
from .ml_models import AmblerSVMModel, AmblerRFModel
from .data import SeqDataset, PredictDataset, prepare_data, get_fasta_dict, get_label
from .training import run_kfold_training, train_epoch, evaluate_epoch
from .ml_training import run_ml_kfold_training
from .ensemble import EnsemblePredictor, find_best_fold_models

__all__ = [
    # Backbone
    'ESM2Backbone',
    'ESM2BackboneSimple',
    # Deep Learning Heads
    'MLPHead',
    'CNNHead',
    'LSTMHead',
    # Machine Learning Heads
    'SVMHead',
    'RFHead',
    # Deep Learning Models
    'AmblerMLPModel',
    'AmblerCNNModel',
    'AmblerLSTMModel',
    # Machine Learning Models
    'AmblerSVMModel',
    'AmblerRFModel',
    # Data
    'SeqDataset',
    'PredictDataset',
    'prepare_data',
    'get_fasta_dict',
    'get_label',
    # Training
    'run_kfold_training',
    'run_ml_kfold_training',
    'train_epoch',
    'evaluate_epoch',
    # Ensemble
    'EnsemblePredictor',
    'find_best_fold_models',
]