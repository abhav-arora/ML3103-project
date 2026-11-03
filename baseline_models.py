"""RUL regressors. Every forward pass returns a scalar in training target scale."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Dense(nn.Module):
    def __init__(self, window, features, widths):
        super().__init__()
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip((window * features, *widths), widths))
        self.readout = nn.Linear(widths[-1], 1)

    def forward(self, x, mask):
        state = (x * mask.unsqueeze(-1)).flatten(1)
        for layer in self.hidden:
            state = layer(state).relu()
        return self.readout(state).squeeze(-1)


class Sequence(nn.Module):
    def __init__(self, features, kind, width=48):
        super().__init__()
        self.conv = nn.Conv1d(features, width, 3, padding=1) if kind == "cnn_lstm" else None
        cell = {"rnn": nn.RNN, "gru": nn.GRU}.get(kind, nn.LSTM)
        bidirectional = kind in {"bilstm", "attention_bilstm"}
        self.cell = cell(width if self.conv else features, width, batch_first=True, bidirectional=bidirectional)
        size = width * (2 if bidirectional else 1)
        self.attention = nn.Sequential(nn.Linear(size, width), nn.Tanh(), nn.Linear(width, 1)) if kind == "attention_bilstm" else None
        self.readout = nn.Linear(size, 1)

    def forward(self, x, mask):
        # Prepared windows have left padding. Move observed rows first for packing.
        order = torch.argsort(~mask, dim=1, stable=True)
        x = (x * mask.unsqueeze(-1)).gather(1, order.unsqueeze(-1).expand_as(x))
        lengths = mask.sum(1)
        if self.conv is not None:
            x = self.conv(x.transpose(1, 2)).relu().transpose(1, 2)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, state = self.cell(packed)
        if self.attention is not None:
            output, _ = pad_packed_sequence(output, batch_first=True)
            valid = torch.arange(output.size(1), device=x.device)[None, :] < lengths[:, None]
            weights = self.attention(output).squeeze(-1).masked_fill(~valid, -torch.inf).softmax(1)
            representation = (output * weights.unsqueeze(-1)).sum(1)
        else:
            hidden = state[0] if isinstance(state, tuple) else state
            representation = hidden.transpose(0, 1).flatten(1)
        return self.readout(representation).squeeze(-1)


class HardSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage):
        ctx.save_for_backward(voltage)
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, gradient):
        voltage, = ctx.saved_tensors
        return gradient / (1 + 10 * voltage.abs()).square()


class RecurrentLIF(nn.Module):
    """Direct input current, fixed leak/threshold and supervised surrogate gradients."""
    def __init__(self, features, width=48):
        super().__init__()
        self.input = nn.Linear(features, width)
        self.recurrent = nn.Linear(width, width, bias=False)
        self.readout = nn.Linear(width, 1)

    def forward(self, x, mask):
        currents = self.input(x)
        voltage = currents.new_zeros(currents.size(0), currents.size(2))
        spikes = torch.zeros_like(voltage)
        total = torch.zeros_like(voltage)
        for current, valid in zip(currents.unbind(1), mask.unbind(1)):
            potential = 0.9 * voltage + current + self.recurrent(spikes)
            fired = HardSpike.apply(potential - 1)
            valid = valid[:, None]
            voltage = torch.where(valid, potential - fired.detach(), voltage)
            spikes = torch.where(valid, fired, spikes)
            total = total + torch.where(valid, fired, 0)
        return self.readout(total / mask.sum(1, keepdim=True)).squeeze(-1)


class ConvertedSNN(nn.Module):
    """Rate conversion of a ReLU DeepNN using train-only activation calibration."""
    def __init__(self, source, thresholds, steps):
        super().__init__()
        self.source = source
        self.register_buffer("thresholds", thresholds)
        self.steps = steps

    def forward(self, x, mask):
        flat = (x * mask.unsqueeze(-1)).flatten(1)
        first_current = self.source.hidden[0](flat)
        voltage = [flat.new_zeros(flat.size(0), layer.out_features) for layer in self.source.hidden]
        accumulated = torch.zeros_like(voltage[-1])
        for _ in range(self.steps):
            current = first_current
            for index, (layer, threshold) in enumerate(zip(self.source.hidden, self.thresholds)):
                if index:
                    current = layer(activity)
                voltage[index] = voltage[index] + current
                activity = (voltage[index] >= threshold).to(current.dtype) * threshold
                voltage[index] = voltage[index] - activity
            accumulated = accumulated + activity
        return self.source.readout(accumulated / self.steps).squeeze(-1)


@torch.no_grad()
def calibrate_conversion(model, batches, device):
    maxima = torch.zeros(len(model.hidden), device=device)
    for x, mask, _ in batches:
        state = (x.to(device) * mask.to(device).unsqueeze(-1)).flatten(1)
        for index, layer in enumerate(model.hidden):
            state = layer(state).relu()
            maxima[index] = torch.maximum(maxima[index], state.max())
    return maxima.clamp_min(1e-6)


MODELS = ("mean", "deepnn", "widenn", "rnn", "lstm", "gru", "bilstm", "cnn_lstm",
          "attention_bilstm", "small_lstm_kd", "snn", "ann_to_snn")


def make_model(name, window, features):
    if name == "deepnn":
        return Dense(window, features, (128, 96, 64, 32))
    if name == "widenn":
        return Dense(window, features, (256,))
    if name == "snn":
        return RecurrentLIF(features)
    if name == "small_lstm_kd":
        return Sequence(features, "lstm", width=16)
    if name in {"rnn", "lstm", "gru", "bilstm", "cnn_lstm", "attention_bilstm"}:
        return Sequence(features, name)
    raise ValueError(f"Unknown trainable model: {name}")
