"""Plot archived measurements without inference or hand-entered scores."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def plot(filename, output, ablation=False):
    with (ROOT / "results" / filename).open() as f:
        rows = list(csv.DictReader(f))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.5), layout="constrained")
    labels = (["w/o evidence", "w/o geometry loss", "w/o intrinsic loss", "w/o memory", "Full"]
              if ablation else ["3DCodeVerse (30)", "GPT-6 Sol (32)", "VIGA (32)", "weiSceneRig (14)", "Ours (32)"])
    for ax, field, title in zip(axes, ["psnr_db", "ssim", "lpips_alex_v0_1", "dreamsim_ensemble"],
                               ["PSNR ↑", "SSIM ↑", "LPIPS ↓", "DreamSim ↓"]):
        values = [float(r[field if ablation else field + "_mean"]) for r in rows]
        bars = ax.barh(labels, values, color=["#b5beca"] * 4 + ["#466f76"], height=.62)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlim(0, max(values) * 1.28)
        ax.bar_label(bars, fmt="%.2f" if field == "psnr_db" else "%.3f", padding=3, fontsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e6e9ec", linewidth=.6)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    fig.suptitle("Ablation: matched 16-image subset" if ablation else "Source-view reconstruction: available image pairs", x=.01, ha="left", fontsize=12)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(ROOT / "assets" / f"{output}.{ext}", dpi=240, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    plot("manuscript_results_source.csv", "main_results")
    plot("ablation_results_source.csv", "ablation_results", True)
