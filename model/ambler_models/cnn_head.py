"""
CNN head module.
Multi-scale 1D-CNN classification head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNHead(nn.Module):
    """
    Multi-scale 1D-CNN classification head.
    Input: last_hidden_state [B, seq_len, hidden_dim] (CLS and EOS already removed)
    Output: logits [B, num_labels]
    """
    def __init__(
        self,
        hidden_dim: int,
        num_labels: int,
        num_filters: int = 256,
        kernel_sizes: list = [3, 5, 7, 11],
        dropout: float = 0.3
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes

        # One independent Conv1d per kernel size
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=hidden_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                    padding=k // 2  # same padding keeps the length unchanged
                ),
                nn.BatchNorm1d(num_filters),
                nn.GELU(),
            )
            for k in kernel_sizes
        ])

        # Classifier applied to the concatenated outputs of all kernels
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_filters * len(kernel_sizes), 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_labels)
        )

    def forward(self, hidden_state, attention_mask=None):
        """
        Args:
            hidden_state: [B, seq_len, hidden_dim]
            attention_mask: [B, seq_len]
        Returns:
            logits: [B, num_labels]
        """
        # Conv1d expects input in [B, channels, length] format
        x = hidden_state.transpose(1, 2)  # [B, hidden_dim, seq_len]

        pooled_outputs = []
        for conv in self.convs:
            feat = conv(x)  # [B, num_filters, seq_len]

            # If a mask is given, set padded positions to a very small value before max pooling
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(1).float()  # [B, 1, seq_len]
                feat = feat * mask + (1 - mask) * (-1e9)

            # Global Max Pooling: take the maximum activation of each filter over the sequence
            pooled = F.adaptive_max_pool1d(feat, 1).squeeze(-1)  # [B, num_filters]
            pooled_outputs.append(pooled)

        # Concatenate features from all scales [B, num_filters * len(kernel_sizes)]
        concat = torch.cat(pooled_outputs, dim=-1)
        logits = self.classifier(concat)
        return logits
