"""Adapt two published architecture families to the shared prepared windows."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from spikedistill.models.distillation import Student


class LiDCNN(nn.Module):
    """Li et al. (2018), Fig. 2; use our inputs and mask unobserved cycles."""

    def __init__(self, features, window):
        super().__init__()
        widths = (features, 10, 10, 10, 10, 1)
        self.convs = nn.ModuleList(
            nn.Conv1d(a, b, kernel, padding="same")
            for a, b, kernel in zip(widths[:-1], widths[1:], (10, 10, 10, 10, 3))
        )
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(window, 100),
                                  nn.Tanh(), nn.Linear(100, 1))

    def forward(self, x, mask):
        valid = mask.unsqueeze(1)
        state = x.transpose(1, 2).masked_fill(~valid, 0)
        for layer in self.convs:
            state = layer(state).tanh().masked_fill(~valid, 0)
        return self.head(state.flatten(1)).squeeze(-1)


class AttentionBiGRU(nn.Module):
    """BiGRU and temporal self-attention on observed sensor windows."""

    def __init__(self, features, hidden=64):
        super().__init__()
        self.memory = nn.GRU(features, hidden, num_layers=2, dropout=0.3,
                             batch_first=True, bidirectional=True)
        self.project = nn.Linear(2 * hidden, 64)
        self.attention = nn.MultiheadAttention(64, 4, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(64)
        self.head = nn.Sequential(nn.Linear(64 + features, 64), nn.GELU(),
                                  nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x, mask):
        # pack_padded_sequence expects valid cycles first. Keep their order.
        if bool(mask.all()):
            states, _ = self.memory(x)
            valid = mask
        else:
            order = torch.argsort(~mask, dim=1, stable=True)
            observed = x.gather(1, order.unsqueeze(-1).expand_as(x))
            lengths = mask.sum(1)
            packed = pack_padded_sequence(observed, lengths.cpu(), batch_first=True,
                                          enforce_sorted=False)
            states, _ = self.memory(packed)
            states, _ = pad_packed_sequence(states, batch_first=True, total_length=x.shape[1])
            valid = mask.gather(1, order)
        states = self.project(states)
        attended, _ = self.attention(states, states, states, key_padding_mask=~valid,
                                     need_weights=False)
        pooled = self.norm(states + attended).masked_fill(~valid.unsqueeze(-1), 0).sum(1)
        pooled = pooled / valid.sum(1, keepdim=True)
        return self.head(torch.cat((pooled, x[:, -1]), dim=1)).squeeze(-1)


def build(name, features, window):
    if name == "li_dcnn":
        return LiDCNN(features, window)
    if name == "attention_bigru":
        return AttentionBiGRU(features)
    return Student(features, width=64)
