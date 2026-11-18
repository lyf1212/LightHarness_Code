"""Refit pooled BT scores from the latest archived pair-win counts."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from evaluate_vlm import fit_bradley_terry

report = json.loads((ROOT / "results/vlm_latest.json").read_text())
out = {}
for criterion, summary in report["summary"].items():
    expected = summary["pooled_mean_centered_bt_scores"]
    names = list(expected)
    counts = {}
    for pair, values in summary["pair_win_counts"].items():
        a, b = pair.split(" vs ")
        counts[names.index(a), names.index(b)] = values
    scores = fit_bradley_terry(names, counts)
    if any(abs(scores[n] - expected[n]) > 1e-9 for n in names):
        raise ValueError(f"Archived pooled votes and scores differ: {criterion}")
    out[criterion] = scores
path = ROOT / "results/recomputed_vlm_latest.json"
path.write_text(json.dumps(out, indent=2) + "\n")
print(path)
print("This refits the later vote archive, not the manuscript BT snapshot.")
