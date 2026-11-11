"""
Full LSTM model.
ESM2 backbone + LSTM classification head.
Used for Ambler classification.
"""
import torch
import torch.nn as nn
from .backbone import ESM2Backbone
from .lstm_head import LSTMHead


class AmblerLSTMModel(nn.Module):
    """
    ESM2 backbone + LSTM classification head.
    CLS and EOS tokens are removed automatically.
    """
    def __init__(
        self,
        model_name: str,
        num_labels: int = 5,
        lstm_hidden: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        self.backbone = ESM2Backbone(model_name)
        hidden_dim = self.backbone.esm.config.hidden_size

        self.head = LSTMHead(
            hidden_dim=hidden_dim,
            num_labels=num_labels,
            lstm_hidden=lstm_hidden,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            labels: [B] (optional)
        Returns:
            dict with 'loss' and 'logits'
        """
        # The backbone already removes CLS and EOS tokens
        hidden_state, mask = self.backbone(input_ids, attention_mask)
        logits = self.head(hidden_state, mask)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}
