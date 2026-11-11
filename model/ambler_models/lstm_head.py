"""
LSTM head module.
Bidirectional LSTM classification head.
"""
import torch
import torch.nn as nn


class LSTMHead(nn.Module):
    """
    Bidirectional LSTM classification head.
    Input: last_hidden_state [B, seq_len, hidden_dim] (CLS and EOS already removed)
    Output: logits [B, num_labels]
    """
    def __init__(
        self,
        hidden_dim: int,
        num_labels: int,
        lstm_hidden: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Bidirectional concatenation doubles the dimension: lstm_hidden * 2
        classifier_input_dim = lstm_hidden * 2 if bidirectional else lstm_hidden
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, 512),
            nn.LayerNorm(512),
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
        # Use pack_padded_sequence to skip padding so the LSTM only sees real residues
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()  # [B] true length of each sequence
            packed = nn.utils.rnn.pack_padded_sequence(
                hidden_state, lengths,
                batch_first=True, enforce_sorted=False
            )
            _, (hn, _) = self.lstm(packed)
        else:
            _, (hn, _) = self.lstm(hidden_state)

        # hn: [num_layers * num_directions, B, lstm_hidden]
        if self.bidirectional:
            # Concatenate the forward and backward hidden states of the last layer
            # Forward: hn[-2], backward: hn[-1]
            last_hidden = torch.cat([hn[-2], hn[-1]], dim=-1)  # [B, lstm_hidden * 2]
        else:
            last_hidden = hn[-1]  # [B, lstm_hidden]

        logits = self.classifier(last_hidden)
        return logits
