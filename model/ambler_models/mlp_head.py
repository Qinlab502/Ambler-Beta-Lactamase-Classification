"""
MLP head module.
Aggregates the sequence representation with mean pooling (CLS and EOS already removed).
"""
import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """
    MLP classification head.
    Input: last_hidden_state [B, seq_len, hidden_dim] (CLS and EOS already removed)
    Output: logits [B, num_labels]
    """
    def __init__(
        self,
        hidden_dim: int,
        num_labels: int,
        dropout: float = 0.3
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )

    def mean_pooling(self, hidden_state, attention_mask):
        """
        Mean pooling over the sequence.
        Note: the input is expected to have CLS and EOS tokens already removed.
        Args:
            hidden_state: [B, seq_len, hidden_dim]
            attention_mask: [B, seq_len]
        Returns:
            pooled: [B, hidden_dim]
        """
        mask = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
        sum_emb = (hidden_state * mask).sum(dim=1)   # [B, hidden_dim]
        valid_len = mask.sum(dim=1).clamp(min=1e-9)   # [B, 1]
        return sum_emb / valid_len                    # [B, hidden_dim]

    def forward(self, hidden_state, attention_mask=None):
        """
        Args:
            hidden_state: [B, seq_len, hidden_dim]
            attention_mask: [B, seq_len]
        Returns:
            logits: [B, num_labels]
        """
        if attention_mask is not None:
            pooled = self.mean_pooling(hidden_state, attention_mask)
        else:
            pooled = hidden_state.mean(dim=1)

        logits = self.classifier(pooled)
        return logits
