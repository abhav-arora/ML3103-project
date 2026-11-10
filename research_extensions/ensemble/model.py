"""Load validation-selected member checkpoints as an executable PyTorch ensemble."""
from pathlib import Path

import torch
from torch import nn

from research_extensions.ensemble.run import make_plan
from spikedistill.models.baselines import make_model
from spikedistill.models.distillation import Student


class Ensemble(nn.Module):
    """Members return normalized RUL. Apply the shared RUL cap once after inference."""
    def __init__(self, members, weights):
        super().__init__()
        values = torch.as_tensor(weights, dtype=torch.float32)
        if (not members or values.shape != (len(members),) or not torch.isfinite(values).all()
                or (values < 0).any() or not torch.isclose(values.sum(), torch.tensor(1.))):
            raise ValueError("Use one nonnegative weight per member, summing to one")
        self.members = nn.ModuleList(members)
        self.register_buffer("weights", values)

    def forward(self, x, mask):
        prediction = torch.stack([model(x, mask) for model in self.members])
        return (prediction * self.weights[:, None]).sum(0)


def load_ensemble(member_folders, plan, weighting="equal", device="cpu"):
    folders = [Path(path) for path in member_folders]
    if make_plan(folders) != plan:
        raise ValueError("Member validation files differ from the fitted plan")
    models = []
    for folder, entry in zip(folders, plan["members"]):
        state = torch.load(folder / f"epoch_{entry['selected_epoch']:04d}.pt",
                           map_location="cpu", weights_only=True)
        config = state["config"]
        if config != entry["config"] or state["epoch"] != entry["selected_epoch"]:
            raise ValueError(f"Checkpoint mismatch in {folder}")
        width = config.get("width", 48)
        features = config.get("features")
        name = config["model"]
        is_spiking_student = name in {"learnable_snn", "spike_output_kd", "spike_temporal_kd", "spikedistill"} or name.startswith("spikedistill_w")
        if is_spiking_student:
            # Legacy student metadata omits feature width. The input projection stores it.
            features = state["model"]["input.weight"].shape[1]
            model = Student(features, width=width)
        else:
            model = make_model(name, config["window"], features)
        model.load_state_dict(state["model"], strict=True)
        models.append(model)
    key = {"equal": "equal_weights", "validation": "inverse_validation_mse_weights"}[weighting]
    return Ensemble(models, plan[key]).to(device).eval()
