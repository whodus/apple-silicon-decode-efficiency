#!/usr/bin/env python3
"""
Plot decode tok/s and effective bandwidth vs. bits per weight.

Usage:
    python3 plot_results.py                 # every results/*.json
    python3 plot_results.py results/x.json  # specific runs
Saves results/decode_vs_bits.png (needs matplotlib).
"""
import glob
import json
import sys

import matplotlib.pyplot as plt


def main(paths):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for path in paths:
        run = json.load(open(path))
        rows = sorted(run["rows"], key=lambda r: r["bits"])
        bits = [r["bits"] for r in rows]
        ax1.plot(bits, [r["decode_tps"] for r in rows], marker="o", label=run["chip"])
        ax2.plot(bits, [r["eff_bw"] for r in rows], marker="o", label=run["chip"])
    for ax in (ax1, ax2):
        ax.set_xlabel("bits per weight")
        ax.set_xticks([4, 8, 16])
        ax.legend()
    ax1.set_ylabel("decode tok/s")
    ax2.set_ylabel("~effective GB/s (tok/s x weight GB)")
    ax2.set_ylim(bottom=0)
    fig.suptitle("Qwen2.5-1.5B decode on Apple silicon (MLX)")
    fig.tight_layout()
    fig.savefig("results/decode_vs_bits.png", dpi=150)
    print("Saved results/decode_vs_bits.png")


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(glob.glob("results/*.json")))
