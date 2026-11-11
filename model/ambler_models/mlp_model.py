"""
Full MLP model.
ESM2 backbone + MLP classification head.
Used for Ambler classification.
"""
import torch
import torch.nn as nn
from .backbone import ESM2Backbone
from .mlp_head import MLPHead


class AmblerMLPModel(nn.Module):
    """
    ESM2 backbone + MLP classification head.
    CLS and EOS tokens are removed automatically.
    """
    def __init__(
        self,
        model_name: str,
        num_labels: int = 5,
        dropout: float = 0.3
    ):
        super().__init__()
        self.backbone = ESM2Backbone(model_name)
        hidden_dim = self.backbone.esm.config.hidden_size

        self.head = MLPHead(
            hidden_dim=hidden_dim,
            num_labels=num_labels,
            dropout=dropout
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
