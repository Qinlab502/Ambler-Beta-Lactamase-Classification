"""
ESM2 backbone module.
Frozen ESM2 feature extractor that returns last_hidden_state with CLS and EOS removed.
"""
import torch
import torch.nn as nn
from transformers import EsmModel


class ESM2Backbone(nn.Module):
    """
    Frozen ESM2 feature extractor.
    Output: last_hidden_state with CLS and EOS tokens removed.
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)
        self._freeze()

    def _freeze(self):
        """Freeze all ESM2 parameters."""
        for param in self.esm.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
        Returns:
            hidden_state: [B, seq_len-2, hidden_dim]  # CLS and EOS removed
            new_attention_mask: [B, seq_len-2]
        """
        with torch.no_grad():
            outputs = self.esm(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        # [B, seq_len, hidden_dim]
        hidden_state = outputs.last_hidden_state

        # Remove CLS (position 0) and EOS (last valid token of each sequence)
        # attention_mask.sum(dim=1) is the true length of each sequence; EOS sits at that position - 1
        batch_size = hidden_state.size(0)

        # Build a new mask: drop position 0 (CLS) and the EOS position of each sequence
        # First drop the CLS token
        hidden_state_no_cls = hidden_state[:, 1:, :]  # [B, seq_len-1, hidden_dim]
        attention_mask_no_cls = attention_mask[:, 1:]  # [B, seq_len-1]

        # Valid length of each sequence after dropping CLS (original length - 1); EOS is the last valid position
        seq_lengths = attention_mask_no_cls.sum(dim=1)  # [B]

        # Drop the EOS token
        result_hidden = []
        result_mask = []

        for i in range(batch_size):
            valid_len = seq_lengths[i].int().item()
            if valid_len > 0:
                # Keep everything except EOS (i.e. the first valid_len-1 tokens)
                result_hidden.append(hidden_state_no_cls[i, :valid_len-1, :])
                result_mask.append(attention_mask_no_cls[i, :valid_len-1])
            else:
                # Edge case: sequence contains only CLS + EOS and no inner tokens
                result_hidden.append(hidden_state_no_cls[i, :0, :])  # empty sequence
                result_mask.append(attention_mask_no_cls[i, :0])

        # Sequences have different lengths, so pad them
        max_len = max(h.size(0) for h in result_hidden)
        hidden_dim = hidden_state.size(-1)

        padded_hidden = torch.zeros(batch_size, max_len, hidden_dim,
                                    device=hidden_state.device, dtype=hidden_state.dtype)
        padded_mask = torch.zeros(batch_size, max_len,
                                  device=attention_mask.device, dtype=attention_mask.dtype)

        for i in range(batch_size):
            seq_len_i = result_hidden[i].size(0)
            if seq_len_i > 0:
                padded_hidden[i, :seq_len_i, :] = result_hidden[i]
                padded_mask[i, :seq_len_i] = result_mask[i]

        return padded_hidden, padded_mask


class ESM2BackboneSimple(nn.Module):
    """
    Simplified ESM2 backbone (keeps the raw output, does not handle special tokens).
    Intended for heads that handle special tokens themselves.
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)
        self._freeze()

    def _freeze(self):
        for param in self.esm.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Returns:
            hidden_state: [B, seq_len, hidden_dim]
            attention_mask: [B, seq_len]
        """
        with torch.no_grad():
            outputs = self.esm(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        return outputs.last_hidden_state, attention_mask
