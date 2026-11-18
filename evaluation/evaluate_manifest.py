"""Evaluate image pairs from a portable JSON manifest using the archived protocol."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate import load_rgb, rgb_metrics, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--perceptual", action="store_true", help="Also compute LPIPS and DreamSim (downloads weights on first use)")
    parser.add_argument("--resize-reference-to-render", action="store_true")
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text())
    if args.perceptual:
        import torch
        import lpips
        from evaluate_dreamsim import DreamSimMetric
        lp = lpips.LPIPS(net="alex", version="0.1").to(args.device).eval()
        ds = DreamSimMetric(args.device)
    results = []
    for record in records:
        row = {k: record[k] for k in ("scene_index", "scene_id", "method") if k in record}
        ref_path = (args.manifest.parent / record["reference"]).resolve()
        pred_path = (args.manifest.parent / record["render"]).resolve()
        if not ref_path.is_file() or not pred_path.is_file():
            raise FileNotFoundError(f"Missing input pair: {ref_path}, {pred_path}")
        ref, _ = load_rgb(ref_path, (0, 0, 0))
        pred, _ = load_rgb(pred_path, (0, 0, 0))
        resized = ref.shape != pred.shape
        if resized:
            if not args.resize_reference_to_render:
                raise ValueError("Image sizes differ; use --resize-reference-to-render for the paper protocol")
            ref = np.asarray(Image.fromarray(ref).resize((pred.shape[1], pred.shape[0]), Image.Resampling.LANCZOS))
        row.update(rgb_metrics(ref, pred, np.ones(ref.shape[:2], dtype=bool)))
        row.update(reference_sha256=sha256(ref_path), render_sha256=sha256(pred_path), resize_applied=resized)
        if args.perceptual:
            def tensor(x):
                return torch.from_numpy(np.array(x, dtype=np.float32) / 255).permute(2, 0, 1)[None].to(args.device)
            with torch.inference_mode():
                row["lpips_alex_v0_1"] = float(lp(tensor(ref), tensor(pred), normalize=True).item())
            row["dreamsim_ensemble"] = ds.score(ref, pred)
        results.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
