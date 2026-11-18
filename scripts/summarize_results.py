"""Recompute archived image-metric means and check the manuscript CSV."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("psnr_db", "ssim", "lpips_alex_v0_1", "dreamsim_ensemble")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/recomputed_image_metrics.csv")
    args = parser.parse_args()
    rows = json.loads((ROOT / "results/manuscript_per_image_metrics.json").read_text())
    with (ROOT / "results/manuscript_results_source.csv").open() as f:
        expected = list(csv.DictReader(f))
    result = []
    for e in expected:
        selected = [r for r in rows if r["method"] == e["method"] and r["status"] == "evaluated"]
        if len(selected) != int(e["evaluated"]):
            raise ValueError(f"Sample count mismatch: {e['method']}")
        out = {"method": e["method"], "count": len(selected)}
        for metric in FIELDS:
            values = [r[metric] for r in selected if r.get(metric) is not None]
            value = mean(values)
            if len(values) != int(e[metric + "_count"]) or abs(value - float(e[metric + "_mean"])) > 1e-10:
                raise ValueError(f"Manuscript snapshot mismatch: {e['method']} / {metric}")
            out[metric] = value
        result.append(out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "count", *FIELDS])
        writer.writeheader()
        writer.writerows(result)
    print(f"Verified {sum(r['count'] for r in result)} image pairs against the manuscript snapshot.")
    print("Manuscript BT scores are preserved only as a snapshot; see docs/results.md.")
    print("The current image archive replaces weiSceneRig with SceneRig; its images are not the old table's inputs.")
    print(args.output)


if __name__ == "__main__":
    main()
