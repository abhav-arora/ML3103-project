"""Read and integrate Jetson module power without changing board settings."""

from dataclasses import dataclass
from pathlib import Path
import re
from threading import Event, Thread
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class Rail:
    name: str
    voltage: Path
    current: Path

    def watts(self):
        return float(self.voltage.read_text()) * float(self.current.read_text()) / 1e6


def find_rail(name, root=Path("/sys/class/hwmon")):
    for label in sorted(root.glob("hwmon*/in*_label")):
        match = re.fullmatch(r"in(\d+)_label", label.name)
        if match is None:
            continue
        try:
            if label.read_text().strip() != name:
                continue
        except OSError:
            continue
        channel = match[1]
        rail = Rail(name, label.with_name(f"in{channel}_input"),
                    label.with_name(f"curr{channel}_input"))
        try:
            rail.watts()
        except (OSError, ValueError):
            continue
        return rail
    return None


def integrate(samples, start, end):
    """Integrate between run boundaries using bracketing sensor samples."""
    times, watts = np.asarray(samples, dtype=float).T
    inside = (times > start) & (times < end)
    boundaries = np.r_[start, times[inside], end]
    if not times[0] <= start < end <= times[-1] or inside.sum() < 2:
        raise ValueError("Power samples did not cover the measurement interval")
    if not np.isfinite(watts).all() or (watts < 0).any():
        raise ValueError("Power sensor returned an invalid reading")
    levels = np.interp(boundaries, times, watts)
    return float(np.sum(np.diff(boundaries) * (levels[1:] + levels[:-1]) / 2))


def measure(rail, infer, seconds, interval=0.05):
    samples = []
    errors = []
    stop = Event()

    def sample():
        before = perf_counter()
        value = rail.watts()
        samples.append(((before + perf_counter()) / 2, value))

    def poll():
        try:
            while not stop.wait(interval):
                sample()
        except (OSError, ValueError) as error:
            errors.append(str(error))

    try:
        sample()
    except (OSError, ValueError) as error:
        return {"status": str(error), "rail": rail.name}, []
    thread = Thread(target=poll, daemon=True)
    thread.start()
    start = perf_counter()
    count = 0
    try:
        while perf_counter() - start < seconds:
            infer(count)
            count += 1
        end = perf_counter()
    finally:
        stop.set()
        thread.join()
    try:
        sample()
        if errors:
            raise ValueError(errors[0])
        joules = integrate(samples, start, end)
        result = dict(status="measured", rail=rail.name, voltage_path=str(rail.voltage),
                      current_path=str(rail.current), duration_s=end-start, samples=len(samples),
                      requested_poll_interval_s=interval, inference_count=count, joules=joules,
                      mean_w=joules/(end-start), mj_per_inference=1000*joules/count)
    except (OSError, ValueError) as error:
        result = {"status": str(error), "rail": rail.name}
    return result, [(time - start, watts) for time, watts in samples]
