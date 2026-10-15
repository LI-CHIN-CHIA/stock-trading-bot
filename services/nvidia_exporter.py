#!/usr/bin/env python3
"""Prometheus exporter for NVIDIA GPU metrics via nvidia-smi."""
import subprocess
import time
import os
from prometheus_client import start_http_server, Gauge, Info

PORT = int(os.getenv("PORT", "9878"))
INTERVAL = int(os.getenv("INTERVAL", "10"))

# Gauges
gpu_util        = Gauge("nvidia_gpu_utilization_percent",    "GPU core utilization %", ["gpu", "name"])
mem_used        = Gauge("nvidia_gpu_memory_used_bytes",      "GPU VRAM used (bytes)",  ["gpu", "name"])
mem_total       = Gauge("nvidia_gpu_memory_total_bytes",     "GPU VRAM total (bytes)", ["gpu", "name"])
mem_free        = Gauge("nvidia_gpu_memory_free_bytes",      "GPU VRAM free (bytes)",  ["gpu", "name"])
temperature     = Gauge("nvidia_gpu_temperature_celsius",    "GPU temperature °C",     ["gpu", "name"])
power_draw      = Gauge("nvidia_gpu_power_draw_watts",       "GPU power draw (W)",     ["gpu", "name"])
power_limit     = Gauge("nvidia_gpu_power_limit_watts",      "GPU power limit (W)",    ["gpu", "name"])
fan_speed       = Gauge("nvidia_gpu_fan_speed_percent",      "GPU fan speed %",        ["gpu", "name"])
sm_clock        = Gauge("nvidia_gpu_sm_clock_mhz",          "GPU SM clock MHz",       ["gpu", "name"])
mem_clock       = Gauge("nvidia_gpu_memory_clock_mhz",      "GPU memory clock MHz",   ["gpu", "name"])


def query():
    fields = ",".join([
        "index", "name",
        "utilization.gpu", "memory.used", "memory.total", "memory.free",
        "temperature.gpu", "power.draw", "power.limit",
        "fan.speed", "clocks.sm", "clocks.mem",
    ])
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 12:
            continue
        idx, name = parts[0], parts[1]
        labels = [idx, name]

        def f(v):
            try:
                return float(v) if v not in ("N/A", "[N/A]", "") else 0.0
            except ValueError:
                return 0.0

        gpu_util.labels(*labels).set(f(parts[2]))
        mem_used.labels(*labels).set(f(parts[3]) * 1024 * 1024)
        mem_total.labels(*labels).set(f(parts[4]) * 1024 * 1024)
        mem_free.labels(*labels).set(f(parts[5]) * 1024 * 1024)
        temperature.labels(*labels).set(f(parts[6]))
        power_draw.labels(*labels).set(f(parts[7]))
        power_limit.labels(*labels).set(f(parts[8]))
        fan_speed.labels(*labels).set(f(parts[9]))
        sm_clock.labels(*labels).set(f(parts[10]))
        mem_clock.labels(*labels).set(f(parts[11]))


if __name__ == "__main__":
    start_http_server(PORT)
    print(f"nvidia_exporter listening on :{PORT}", flush=True)
    while True:
        try:
            query()
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
        time.sleep(INTERVAL)
