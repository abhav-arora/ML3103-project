"""Attention teacher signals and a recurrent LIF student."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from spikedistill.models.baselines import HardSpike, Sequence


class Teacher(Sequence):
    """Expose the same BiLSTM states and attention used by the baseline."""
    def __init__(self, features):
        super().__init__(features, "attention_bilstm")

    def signals(self, x, mask):
        order = torch.argsort(~mask, dim=1, stable=True)
        observed = (x * mask.unsqueeze(-1)).gather(1, order.unsqueeze(-1).expand_as(x))
        lengths = mask.sum(1)
        packed = pack_padded_sequence(observed, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.cell(packed)
        states, _ = pad_packed_sequence(output, batch_first=True, total_length=x.size(1))
        valid = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
        weights = self.attention(states).squeeze(-1).masked_fill(~valid, -torch.inf).softmax(1)
        prediction = self.readout((states * weights.unsqueeze(-1)).sum(1)).squeeze(-1)
        inverse = order.argsort(1)
        return prediction, states.gather(1, inverse.unsqueeze(-1).expand_as(states)), weights.gather(1, inverse)


class Student(nn.Module):
    def __init__(self, features, width=48):
        super().__init__()
        self.input = nn.Linear(features, width)
        self.recurrent = nn.Linear(width, width, bias=False)
        self.leak_logits = nn.Parameter(torch.full((width,), float(np.log(9))))
        self.threshold_logits = nn.Parameter(torch.full((width,), float(np.log(np.expm1(1 - 1e-4)))))
        self.attention = nn.Linear(width, 1, bias=False)
        self.readout = nn.Linear(width, 1)

    def signals(self, x, mask):
        currents = self.input(x)
        voltage = currents.new_zeros(currents.size(0), currents.size(2))
        spikes = torch.zeros_like(voltage)
        trace, events = [], []
        leak = self.leak_logits.sigmoid()
        threshold = F.softplus(self.threshold_logits) + 1e-4
        for current, valid in zip(currents.unbind(1), mask.unbind(1)):
            candidate = leak * voltage + current + self.recurrent(spikes)
            fired = HardSpike.apply(candidate - threshold)
            valid = valid.unsqueeze(-1)
            voltage = torch.where(valid, candidate - threshold * fired.detach(), voltage)
            spikes = torch.where(valid, fired, spikes)
            trace.append(torch.where(valid, voltage, 0))
            events.append(torch.where(valid, spikes, 0))
        membrane = torch.stack(trace, dim=1)
        events = torch.stack(events, dim=1)
        attention = self.attention(membrane).squeeze(-1).masked_fill(~mask, -torch.inf).softmax(1)
        prediction = self.readout((events * attention.unsqueeze(-1)).sum(1)).squeeze(-1)
        return prediction, membrane, attention, events

    def forward(self, x, mask):
        return self.signals(x, mask)[0]
