#!/usr/bin/env python3
"""Evaluate a Blender rendering against its pixel-aligned reference image."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_rgb(path: Path, background: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray | None]:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        has_alpha = "A" in image.getbands() or "transparency" in image.info
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32)
    bg = np.asarray(background, dtype=np.float32)
    composited = np.rint(rgb * alpha[:, :, None] + bg * (1 - alpha[:, :, None])).astype(np.uint8)
    return composited, (rgba[:, :, 3] > 127) if has_alpha else None


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as opened:
        mask = np.asarray(ImageOps.exif_transpose(opened).convert("L"), dtype=np.uint8) > 127
    if mask.shape != size:
        raise ValueError(f"Mask {path} has size {mask.shape[::-1]}, expected {size[::-1]}")
    return mask


def ssim_map(reference: np.ndarray, rendering: np.ndarray) -> np.ndarray:
    """Channel-averaged SSIM, 11x11 Gaussian window, sigma=1.5, 8-bit range."""
    x = reference.astype(np.float32)
    y = rendering.astype(np.float32)
    blur = lambda a: cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x * mu_x
    var_y = blur(y * y) - mu_y * mu_y
    cov_xy = blur(x * y) - mu_x * mu_y
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    return np.mean(numerator / denominator, axis=2)


def rgb_metrics(reference: np.ndarray, rendering: np.ndarray, region: np.ndarray) -> dict:
    count = int(region.sum())
    if not count:
        return {"pixel_count": 0, "mae_8bit": None, "rmse_8bit": None,
                "psnr_db": None, "ssim": None}
    residual = reference.astype(np.float32) - rendering.astype(np.float32)
    selected = residual[region]
    mae = float(np.mean(np.abs(selected)))
    rmse = float(np.sqrt(np.mean(selected * selected)))
    structural = ssim_map(reference, rendering)
    # SSIM uses an 11x11 neighborhood; exclude pixels whose neighborhood leaves the ROI.
    interior = cv2.erode(region.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    return {
        "pixel_count": count,
        "mae_8bit": mae,
        "rmse_8bit": rmse,
        "psnr_db": float(20 * np.log10(255.0 / rmse)) if rmse else None,
        "perfect_rgb_match": rmse == 0,
        "ssim": float(np.mean(structural[interior])) if interior.any() else None,
        "ssim_pixel_count": int(interior.sum()),
    }


def contour_metrics(reference: np.ndarray, rendering: np.ndarray, tolerance: float) -> dict:
    n_ref, n_render = int(reference.sum()), int(rendering.sum())
    result = {"reference_pixels": n_ref, "render_pixels": n_render,
              "tolerance_px": tolerance}
    if not n_ref or not n_render:
        result.update({"precision": 0.0 if n_render else None,
                       "recall": 0.0 if n_ref else None,
                       "f1": 0.0 if n_ref != n_render else None,
                       "symmetric_chamfer_px": None})
        return result
    dist_to_ref = cv2.distanceTransform((~reference).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist_to_render = cv2.distanceTransform((~rendering).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    precision = float(np.mean(dist_to_ref[rendering] <= tolerance))
    recall = float(np.mean(dist_to_render[reference] <= tolerance))
    result.update({
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "symmetric_chamfer_px": float((np.mean(dist_to_ref[rendering]) + np.mean(dist_to_render[reference])) / 2),
    })
    return result


def mask_metrics(reference: np.ndarray, rendering: np.ndarray, tolerance: float) -> dict:
    intersection = int(np.logical_and(reference, rendering).sum())
    union = int(np.logical_or(reference, rendering).sum())
    kernel = np.ones((3, 3), np.uint8)
    ref_boundary = cv2.morphologyEx(reference.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    render_boundary = cv2.morphologyEx(rendering.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    return {
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou": intersection / union if union else None,
        "boundary": contour_metrics(ref_boundary, render_boundary, tolerance),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(args: argparse.Namespace) -> dict:
    reference, reference_alpha = load_rgb(args.reference, tuple(args.background))
    rendering, render_alpha = load_rgb(args.render, tuple(args.background))
    if reference.shape != rendering.shape:
        raise ValueError(f"Image sizes differ: reference {reference.shape[:2][::-1]}, render {rendering.shape[:2][::-1]}; images are never resized")
    height, width = reference.shape[:2]
    ref_mask = load_mask(args.reference_mask, (height, width)) if args.reference_mask else reference_alpha
    render_mask = load_mask(args.render_mask, (height, width)) if args.render_mask else render_alpha
    full = np.ones((height, width), dtype=bool)
    reference_edges = cv2.Canny(cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY), args.canny_low, args.canny_high) > 0
    render_edges = cv2.Canny(cv2.cvtColor(rendering, cv2.COLOR_RGB2GRAY), args.canny_low, args.canny_high) > 0
    results = {
        "inputs": {
            "reference": str(args.reference.resolve()), "reference_sha256": sha256(args.reference),
            "render": str(args.render.resolve()), "render_sha256": sha256(args.render),
            "reference_mask": str(args.reference_mask.resolve()) if args.reference_mask else ("reference_alpha" if reference_alpha is not None else None),
            "reference_mask_sha256": sha256(args.reference_mask) if args.reference_mask else None,
            "render_mask": str(args.render_mask.resolve()) if args.render_mask else ("render_alpha" if render_alpha is not None else None),
            "render_mask_sha256": sha256(args.render_mask) if args.render_mask else None,
        },
        "protocol": {
            "width": width, "height": height,
            "registration": "same pixel coordinates; no resizing or geometric warping",
            "rgb_space": "8-bit encoded RGB (display space)",
            "alpha_background_rgb": list(args.background),
            "mask_threshold_8bit": 127,
            "canny_thresholds_8bit": [args.canny_low, args.canny_high],
            "psnr_null_means_perfect_match": True,
        },
        "full_frame_rgb": rgb_metrics(reference, rendering, full),
        "full_frame_edges": contour_metrics(reference_edges, render_edges, args.edge_tolerance),
    }
    if render_mask is not None:
        results["render_coverage_fraction"] = float(render_mask.mean())
        results["render_visible_rgb"] = rgb_metrics(reference, rendering, render_mask)
    if ref_mask is not None:
        results["reference_region_rgb"] = rgb_metrics(reference, rendering, ref_mask)
    if ref_mask is not None and render_mask is not None:
        results["shared_region_rgb"] = rgb_metrics(reference, rendering, ref_mask & render_mask)
        results["foreground_mask"] = mask_metrics(ref_mask, render_mask, args.edge_tolerance)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    error = np.abs(reference.astype(np.int16) - rendering.astype(np.int16)).astype(np.uint8)
    Image.fromarray(error).save(args.output_dir / "absolute_error.png")
    overlay = np.zeros_like(reference)
    overlay[reference_edges, 0] = 255
    overlay[render_edges, 1:] = 255
    Image.fromarray(overlay).save(args.output_dir / "edge_overlay.png")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Input image")
    parser.add_argument("--render", type=Path, required=True, help="Blender RGB/RGBA rendering")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-mask", type=Path, help="Optional binary foreground mask")
    parser.add_argument("--render-mask", type=Path, help="Optional binary mask; overrides render alpha")
    parser.add_argument("--background", type=int, nargs=3, default=(0, 0, 0), metavar=("R", "G", "B"), help="RGB background for RGBA render, default black")
    parser.add_argument("--edge-tolerance", type=float, default=3.0, help="Edge matching tolerance in pixels")
    parser.add_argument("--canny-low", type=int, default=100)
    parser.add_argument("--canny-high", type=int, default=200)
    args = parser.parse_args()
    if not all(0 <= value <= 255 for value in (*args.background, args.canny_low, args.canny_high)):
        parser.error("RGB background and Canny thresholds must lie in [0, 255]")
    if args.canny_low >= args.canny_high or not math.isfinite(args.edge_tolerance) or args.edge_tolerance < 0:
        parser.error("Require canny-low < canny-high and edge-tolerance >= 0")
    print(json.dumps(evaluate(args), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
