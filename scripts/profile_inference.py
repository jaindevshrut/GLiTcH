import time
import statistics
import sys
import os
import torch
torch.backends.quantized.engine = "qnnpack"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.detector import build_model


def profile_cpu_latency(model, input_tensor, runs=200, warmup=20):
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
        for _ in range(runs):
            s = time.time()
            _ = model(input_tensor)
            times.append((time.time() - s) * 1000.0)
    avg = sum(times) / len(times)
    p90 = sorted(times)[int(0.9 * len(times)) - 1]
    p99 = sorted(times)[int(0.99 * len(times)) - 1]
    return avg, p90, p99


def main():
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = build_model(n_mels=64, base_ch=16, num_classes=2)
    model.eval()
    x = torch.randn(1, 1, 64, 1000)
    avg, p90, p99 = profile_cpu_latency(model, x)
    print("fp32_avg_ms", round(avg, 3))
    print("fp32_p90_ms", round(p90, 3))
    print("fp32_p99_ms", round(p99, 3))
    try:
        qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        qavg, qp90, qp99 = profile_cpu_latency(qmodel, x)
        print("dq_int8_avg_ms", round(qavg, 3))
        print("dq_int8_p90_ms", round(qp90, 3))
        print("dq_int8_p99_ms", round(qp99, 3))
    except Exception as e:
        print("quantization_error", str(e))
    sw = model.stream_scores(x, window_frames=128, hop_frames=64)
    print("stream_windows", sw.shape[1])
    savg, sp90, sp99 = profile_cpu_latency(model, x[:, :, :, :128])
    print("window_fp32_avg_ms", round(savg, 3))


if __name__ == "__main__":
    main()
