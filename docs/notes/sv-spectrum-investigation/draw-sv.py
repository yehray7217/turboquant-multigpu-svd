"""
Draw singular value decay plots from .npz files produced by calc-sv-from-XXX.py.

Usage:
    python draw-sv.py sv-of-NLP-data-set.npz
    python draw-sv.py sv-of-image-matrices.npz

Output:
    One PNG per matrix:
        plot-sv-of-NLP-data-set-0.png
        plot-sv-of-NLP-data-set-1.png
        ...

Each .npz file contains:
    - names: array of matrix labels
    - sv_0, sv_1, ...: singular values (descending) for each matrix
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def load_sv(filepath):
    """Load singular values from .npz file.
    Returns list of (name, sv_array) tuples.
    """
    data = np.load(filepath, allow_pickle=True)
    names = list(data["names"])
    result = []
    for i, name in enumerate(names):
        sv = data[f"sv_{i}"]
        result.append((str(name), sv))
    return result

def trim_near_zero(sv, threshold=1e-10):
    """Remove trailing singular values below threshold."""
    idx = np.searchsorted(-sv, -threshold)  # sv is descending
    return sv[:max(idx, 1)]


def draw_one(name, sv, outpath):
    """Draw 3 plots for a single matrix: linear-linear, linear-log, log-log."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(name, fontsize=14, fontweight='bold')

    sv = trim_near_zero(sv)
    ranks = np.arange(1, len(sv) + 1)

    configs = [
        ("Linear-Linear", False, False),
        ("Linear-Log",    False, True),
        ("Log-Log",       True,  True),
    ]

    for ax, (title, xlog, ylog) in zip(axes, configs):
        ax.plot(ranks, sv, linewidth=1.2, color='tab:blue')
        ax.set_xlabel("rank i")
        ax.set_ylabel("singular value σ_i")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if xlog:
            ax.set_xscale("log")
        if ylog:
            ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f"  Saved: {outpath}")
    plt.show()
    plt.close(fig)



def main():
    if len(sys.argv) < 2:
        print("Usage: python draw-sv.py <sv-of-XXX.npz>")
        sys.exit(1)

    inpath = sys.argv[1]
    if not os.path.exists(inpath):
        print(f"File not found: {inpath}")
        sys.exit(1)

    sv_list = load_sv(inpath)
    print(f"Loaded {len(sv_list)} matrices from {inpath}")

    # Derive base name: sv-of-XXX.npz -> plot-sv-of-XXX
    basename = os.path.basename(inpath).replace(".npz", "")
    base_prefix = "plot-" + basename
    outdir = os.path.dirname(inpath) or "."

    for i, (name, sv) in enumerate(sv_list):
        print(f"[{i}] {name}: {len(sv)} singular values")
        outpath = os.path.join(outdir, f"{base_prefix}-{i}.png")
        draw_one(name, sv, outpath)


if __name__ == "__main__":
    main()
