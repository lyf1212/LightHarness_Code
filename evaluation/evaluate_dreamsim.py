#!/usr/bin/env python3
"""Compute official DreamSim ensemble distance for a source-view image pair."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate import load_rgb, sha256


DEFAULT_DREAMSIM_CACHE = Path(__file__).resolve().parent / "models" / "dreamsim"


class DreamSimMetric:
    def __init__(self, device: str = "cpu", cache_dir: Path = DEFAULT_DREAMSIM_CACHE):
        import torch
        from dreamsim import dreamsim

        self.torch = torch
        self.device = torch.device(device)
        self.cache_dir = cache_dir.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        hub_dir = torch.hub.get_dir()
        try:
            self.model, self.preprocess = dreamsim(
                pretrained=True,
                device=str(self.device),
                cache_dir=str(self.cache_dir),
                dreamsim_type="ensemble",
                normalize_embeds=True,
                use_patch_model=False,
            )
        finally:
            # DreamSim changes Torch Hub's global directory while loading DINO.
            torch.hub.set_dir(hub_dir)
        self.model.eval().requires_grad_(False)

    def score(self, reference: np.ndarray, rendering: np.ndarray) -> float:
        # Official preprocessing already returns a 1x3x224x224 batch in [0,1].
        ref = self.preprocess(Image.fromarray(reference)).to(self.device)
        pred = self.preprocess(Image.fromarray(rendering)).to(self.device)
        with self.torch.inference_mode():
            value = float(self.model(ref, pred).item())
        if not np.isfinite(value):
            raise ValueError(f"DreamSim returned a nonfinite distance: {value}")
        return value

    def metadata(self) -> dict:
        packages = ("dreamsim", "torch", "torchvision", "timm", "open-clip-torch", "peft", "transformers")
        return {
            "implementation": "https://github.com/ssundaram21/dreamsim",
            "model": "ensemble",
            "backbones": ["dino_vitb16", "clip_vitb16", "open_clip_vitb16"],
            "pretrained": True,
            "normalize_embeds": True,
            "use_patch_model": False,
            "distance": "1 - cosine_similarity; lower is better",
            "preprocessing": "official PIL RGB bicubic resize to 224x224, ToTensor [0,1]; normalization inside model",
            "cache_dir": str(self.cache_dir),
            "device": str(self.device),
            "package_versions": {name: importlib.metadata.version(name) for name in packages},
        }


def evaluate_dreamsim(args: argparse.Namespace) -> dict:
    reference, _ = load_rgb(args.reference, tuple(args.background))
    rendering, _ = load_rgb(args.render, tuple(args.background))
    if reference.shape != rendering.shape:
        raise ValueError(f"Source-view images must have the same shape: {reference.shape} vs {rendering.shape}")
    metric = DreamSimMetric(args.device, args.dreamsim_cache)
    report = {
        "protocol": {
            **metric.metadata(),
            "reference": str(args.reference.resolve()),
            "render": str(args.render.resolve()),
            "reference_sha256": sha256(args.reference),
            "render_sha256": sha256(args.render),
            "original_size_wh": [reference.shape[1], reference.shape[0]],
            "background_rgb_for_alpha": list(args.background),
            "reconstruction_method": args.method_name,
            "source_texture_derived": bool(args.source_texture_derived),
            "image_registration": False,
        },
        "source_view": {"full_frame_dreamsim_ensemble": metric.score(reference, rendering)},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dreamsim_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu, cuda or cuda:4")
    parser.add_argument("--dreamsim-cache", type=Path, default=DEFAULT_DREAMSIM_CACHE)
    parser.add_argument("--background", type=int, nargs=3, default=(0, 0, 0), metavar=("R", "G", "B"))
    parser.add_argument("--method-name", default="unspecified")
    parser.add_argument("--source-texture-derived", action="store_true")
    args = parser.parse_args()
    if not all(0 <= x <= 255 for x in args.background):
        parser.error("Background RGB values must be in [0,255]")
    report = evaluate_dreamsim(args)
    print(json.dumps({"output": str(args.output_dir / "dreamsim_metrics.json"),
                      "source_view": report["source_view"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
