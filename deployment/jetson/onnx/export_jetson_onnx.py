"""Export the validation-selected FD001 student and freeze parity inputs."""
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import torch

from spikedistill.models.distillation import Student


class Deployment(torch.nn.Module):
    def __init__(self, model, cap):
        super().__init__()
        self.model = model
        self.cap = cap

    def forward(self, sensor_window, valid_mask):
        return self.model(sensor_window, valid_mask > 0.5) * self.cap


def main(data, student, output):
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((student / "test_metrics.json").read_text())
    epoch = report["selected_epoch"]
    path = student / f"epoch_{epoch:04d}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if (config != report["config"] or checkpoint["epoch"] != epoch
            or (config["model"], config["subset"], config["seed"]) != ("spikedistill", "FD001", 42)):
        raise ValueError("Expected the original validation-selected FD001 seed 42 student")
    with np.load(data / "test.npz", allow_pickle=False) as archive:
        test = {name: archive[name] for name in archive.files}
    x = test["X"].astype(np.float32)
    mask = test["mask"].astype(np.float32)
    if x.shape != (100, 30, 18):
        raise ValueError(f"Unexpected FD001 test shape {x.shape}")
    torch.set_num_threads(1)
    model = Student(x.shape[-1]).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    deployment = Deployment(model, config["cap"]).eval()
    # Match the preparation policy for short windows and exercise the mask path.
    padded_x = np.concatenate((np.repeat(x[:1, -11:-10], 19, axis=1), x[:1, -11:]), axis=1)
    padded_mask = np.zeros((1, 30), dtype=np.float32)
    padded_mask[:, -11:] = 1
    all_x = np.concatenate((x, padded_x))
    all_mask = np.concatenate((mask, padded_mask))
    with torch.no_grad():
        reference = deployment(torch.from_numpy(all_x), torch.from_numpy(all_mask)).numpy()
    with (student / "test_predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: int(row["engine_id"]))
    for column, key in (("engine_id", "engine_id"), ("end_cycle", "end_cycle"),
                        ("target_capped_cycles", "y"), ("target_raw_cycles", "y_raw")):
        np.testing.assert_array_equal([float(row[column]) for row in rows], test[key])
    saved = np.array([float(row["prediction_cycles"]) for row in rows])
    np.testing.assert_allclose(reference[:len(x)], saved, rtol=0, atol=0.001)
    model_file = output / "SpikeDistill_RUL_FD001_seed42.onnx"
    # Static 30-step recurrence with the forward-only spike threshold lowered to ONNX ops.
    torch.onnx.export(deployment, (torch.from_numpy(x[:1]), torch.from_numpy(mask[:1])),
                      str(model_file), input_names=["sensor_window", "valid_mask"],
                      output_names=["rul_cycles"], opset_version=17, dynamo=False)
    graph = onnx.load(model_file)
    if any(node.domain not in ("", "ai.onnx") for node in graph.graph.node):
        raise ValueError("Unexpected custom ONNX operators")
    prep = json.loads((data / "preprocessing.json").read_text())
    metadata = dict(subset="FD001", seed=42, selected_epoch=epoch, precision="FP32",
                    checkpoint_sha256=sha256(path.read_bytes()).hexdigest(), feature_names=prep["feature_names"],
                    opset=17, model="SpikeDistill-RUL", window=30, output_units="cycles",
                    training_target_cap_cycles=config["cap"], pytorch=str(torch.__version__),
                    max_pytorch_to_colab_difference_cycles=float(np.max(np.abs(reference[:len(x)]-saved))))
    onnx.helper.set_model_props(graph, {key: json.dumps(value) for key, value in metadata.items()})
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, model_file)
    (output / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "FD001_preprocessing.json").write_text(json.dumps(prep, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output / "test_vectors.npz", sensor_window=all_x, valid_mask=all_mask,
                        expected_cycles=reference, y_raw=test["y_raw"], y_capped=test["y"],
                        engine_id=test["engine_id"], end_cycle=test["end_cycle"])
    (output / "jetson_inputs").mkdir(exist_ok=True)
    for index, name in ((0, "real"), (len(x), "left_padded")):
        all_x[index].astype("<f4").tofile(output / f"jetson_inputs/{name}_window.bin")
        all_mask[index].astype("<f4").tofile(output / f"jetson_inputs/{name}_mask.bin")
        (output / f"jetson_inputs/{name}_expected_cycles.txt").write_text(f"{reference[index]:.8f}\n")


if __name__ == "__main__":
    main(*(Path(arg) for arg in sys.argv[1:]))
