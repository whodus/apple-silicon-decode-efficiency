#!/usr/bin/env python3
"""
Is LLM decoding memory-bandwidth bound on Apple silicon?

Runs the SAME model at three precisions (4-bit, 8-bit, bf16) with MLX.
Same architecture and same math per token, but different bytes per weight.
If decode speed scales with weight size rather than staying flat, decode
is limited by memory bandwidth, not compute.

Setup (Apple silicon Mac, Python 3.9+):
    pip install mlx-lm
Run:
    python3 mlx_bandwidth_bench.py
First run downloads ~4.5 GB of models total. Close other heavy apps first.
Results are also saved to results/<chip>.json.
"""
import json
import os
import platform
import re
import shutil
import subprocess
import sys

MODELS = [
    ("mlx-community/Qwen2.5-1.5B-Instruct-4bit", 4),
    ("mlx-community/Qwen2.5-1.5B-Instruct-8bit", 8),
    ("mlx-community/Qwen2.5-1.5B-Instruct-bf16", 16),
]
PARAMS = 1.54e9  # Qwen2.5-1.5B parameter count (approx.)
PROMPT = "Explain how a CPU cache works, in about three paragraphs."
MAX_TOKENS = 256
RUNS = 2  # first run includes warm-up; keep the best


def chip_name():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def gen_cmd(model):
    exe = shutil.which("mlx_lm.generate")
    base = [exe] if exe else [sys.executable, "-m", "mlx_lm.generate"]
    return base + ["--model", model, "--prompt", PROMPT,
                   "--max-tokens", str(MAX_TOKENS)]


def run_once(model):
    out = subprocess.run(gen_cmd(model), capture_output=True, text=True)
    text = out.stdout + out.stderr
    gen = re.search(r"Generation:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec", text)
    pre = re.search(r"Prompt:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec", text)
    mem = re.search(r"Peak memory:\s*([\d.]+)\s*GB", text)
    if not gen:
        print(f"\nCould not parse output for {model}. Raw output:\n{text[-1500:]}")
        return None
    return {
        "decode_tps": float(gen.group(2)),
        "prefill_tps": float(pre.group(2)) if pre else float("nan"),
        "peak_gb": float(mem.group(1)) if mem else float("nan"),
    }


def save(rows, chip):
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", re.sub(r"[^A-Za-z0-9]+", "-", chip).strip("-").lower() + ".json")
    with open(path, "w") as f:
        json.dump({"chip": chip, "macos": platform.mac_ver()[0], "prompt": PROMPT,
                   "max_tokens": MAX_TOKENS, "runs": RUNS, "params": PARAMS,
                   "models": [m for m, _ in MODELS], "rows": rows}, f, indent=2)
    print(f"Saved {path}")


def main():
    chip = chip_name()
    print(f"Chip: {chip}")
    print("Look up your chip's memory bandwidth (e.g. 'M3 memory bandwidth') "
          "to compare with the last column.\n")
    rows = []
    for model, bits in MODELS:
        print(f"Running {model} ...", flush=True)
        results = [r for r in (run_once(model) for _ in range(RUNS)) if r]
        if not results:
            continue
        best = max(results, key=lambda r: r["decode_tps"])
        weight_gb = PARAMS * bits / 8 / 1e9
        best["bits"] = bits
        best["weight_gb"] = weight_gb
        # Each decoded token reads (roughly) every weight once:
        best["eff_bw"] = best["decode_tps"] * weight_gb
        rows.append(best)

    print("\n{:>5} {:>10} {:>12} {:>13} {:>9} {:>16}".format(
        "bits", "weights GB", "decode tok/s", "prefill tok/s", "peak GB",
        "~effective GB/s"))
    for r in rows:
        print("{:>5} {:>10.2f} {:>12.1f} {:>13.1f} {:>9.2f} {:>16.0f}".format(
            r["bits"], r["weight_gb"], r["decode_tps"], r["prefill_tps"],
            r["peak_gb"], r["eff_bw"]))
    save(rows, chip)

    print("""
How to read this:
- decode tok/s should drop roughly as weights GB grows (about 2x per step).
  Same compute, more bytes -> slower. That means decode is bandwidth bound.
- ~effective GB/s = tok/s x weight size. If it lands in the same ballpark
  across rows and near your chip's spec bandwidth, that's the proof.
  (4-bit has a little extra overhead for quantization scales, so it's approximate.)
- prefill tok/s is much higher: the whole prompt is processed in parallel,
  so weights are reused across many tokens -> more compute bound.
""")


if __name__ == "__main__":
    main()