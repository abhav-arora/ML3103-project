#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
trtexec_bin="${TRTEXEC:-trtexec}"
if ! command -v "$trtexec_bin" >/dev/null && [[ -x /usr/src/tensorrt/bin/trtexec ]]; then
    trtexec_bin=/usr/src/tensorrt/bin/trtexec
fi
if ! command -v "$trtexec_bin" >/dev/null; then
    echo "trtexec is missing. Use the TensorRT tools installed with JetPack." >&2
    exit 1
fi
results="$(mktemp -d ./jetson_onnx_results.XXXXXX)"
echo "Writing $results"
if command -v nvpmodel >/dev/null; then
    nvpmodel -q > "$results/power_mode.txt" 2>&1 || true
fi
if [[ -r /etc/nv_tegra_release ]]; then
    cp /etc/nv_tegra_release "$results/jetson_linux.txt"
fi
"$trtexec_bin" --onnx=SpikeDistill_RUL_FD001_seed42.onnx \
    --saveEngine="$results/spikedistill_fd001_fp32.plan" --noTF32 --skipInference \
    2>&1 | tee "$results/build.log"
for example in real left_padded; do
    printf 'Expected %s output in cycles: ' "$example"
    cat "jetson_inputs/${example}_expected_cycles.txt"
    "$trtexec_bin" --loadEngine="$results/spikedistill_fd001_fp32.plan" \
        --loadInputs="sensor_window:jetson_inputs/${example}_window.bin,valid_mask:jetson_inputs/${example}_mask.bin" \
        --iterations=1 --duration=0 --warmUp=0 --dumpOutput \
        2>&1 | tee "$results/${example}_smoke.log"
done
"$trtexec_bin" --loadEngine="$results/spikedistill_fd001_fp32.plan" \
    --loadInputs=sensor_window:jetson_inputs/real_window.bin,valid_mask:jetson_inputs/real_mask.bin \
    --iterations=1000 --duration=0 --warmUp=1000 --percentile=95 \
    --exportTimes="$results/times.json" 2>&1 | tee "$results/benchmark.log"
