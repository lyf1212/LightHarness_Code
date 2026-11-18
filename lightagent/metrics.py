from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image

from .coordinates import range_to_z
from .io import read, write, preview, read_exr
from .evidence import contact_sheet
from .surface_geometry import CONTRACT, compare_surface_geometry


def positive_scale(reference, prediction, valid):
    a, b = reference[valid].astype(np.float64).reshape(-1), prediction[valid].astype(np.float64).reshape(-1)
    if not a.size or np.dot(b, b) < 1e-12:
        return None
    return max(float(np.dot(a, b) / np.dot(b, b)), 1e-8)


def available(value, **extra):
    return {"available": True, "value": float(value), **extra}


def missing(reason):
    return {"available": False, "value": None, "reason": reason}


def postprocess(folder):
    """Convert float EXR passes; metric arrays always preserve unclipped linear values."""
    folder = Path(folder)
    meta = read(folder / "metadata.json")
    arrays = {}
    for name, file in meta["passes"].items():
        x = read_exr(folder / file)
        if name in {"depth_native", "object_index"} and x.ndim == 3:
            x = x[..., 0]
        arrays[name] = x.astype(np.float32)
        np.save(folder / f"{name}.npy", arrays[name])
    camera = meta["camera"]
    if meta["depth_native"] == "camera_z":
        depth = arrays["depth_native"].copy()
    elif meta["depth_native"] == "ray_distance":
        depth = range_to_z(arrays["depth_native"], np.array(camera["K"]))
    else:
        raise ValueError("Unknown Blender depth convention; calibrate with a fronto-parallel plane")
    valid = np.isfinite(depth) & (depth > 0) & (depth < 1e8)
    depth[~valid] = 0
    np.save(folder / "depth.npy", depth)
    np.save(folder / "depth_valid.npy", valid)
    R = np.array(camera["cv_to_blender_world_rotation"])
    normals = arrays["normals_world"] @ R
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals /= np.maximum(norm, 1e-8)
    np.save(folder / "normals_cv.npy", normals)
    shading = arrays["diffuse_direct"] + arrays["diffuse_indirect"]
    np.save(folder / "shading.npy", shading)
    preview(folder / "depth.png", depth, "depth")
    preview(folder / "normals.png", normals, "normal")
    preview(folder / "albedo.png", arrays["albedo"])
    preview(folder / "shading.png", shading, "hdr")
    for name, x in arrays.items():
        if name.startswith("light_"):
            preview(folder / f"{name}.png", x, "hdr")
    (folder / "masks").mkdir(exist_ok=True)
    indices = np.rint(arrays["object_index"]).astype(np.int32)
    for oid, index in meta["object_indices"].items():
        mask = indices == index
        np.save(folder / "masks" / f"{oid}.npy", mask)
        preview(folder / "masks" / f"{oid}.png", mask, "srgb")


def evaluate(evidence, rendered, out, config):
    evidence, rendered, out = Path(evidence), Path(rendered), Path(out)
    out.mkdir(exist_ok=True)
    render_meta = read(rendered / "metadata.json")
    ref = np.asarray(Image.open(evidence / "reference.png").convert("RGB"), np.float32)/255
    rgb = np.asarray(Image.open(rendered / "rgb.png").convert("RGB"), np.float32)/255
    if rgb.shape != ref.shape:
        raise ValueError("Reference/render pixel grid mismatch")
    metrics, errors, masks = {}, {}, {}
    per_object = {}
    for oid, item in read(evidence / "masks/index.json").items():
        if not item["available"]:
            per_object[oid] = missing("reference SAM3 mask unavailable")
            continue
        a = np.load(evidence / item["path"]).astype(bool)
        p = rendered / "masks" / f"{oid}.npy"
        b = np.load(p).astype(bool) if p.exists() else np.zeros_like(a)
        union = (a | b).sum()
        per_object[oid] = available((a & b).sum()/max(union, 1), reference_pixels=int(a.sum()), rendered_pixels=int(b.sum()))
        masks[oid] = (a, b)
    ious = [m["value"] for m in per_object.values() if m["available"]]
    metrics["silhouette_iou"] = available(np.mean(ious), per_object=per_object) if ious else missing("no masks")
    if ious:
        errors["silhouette"] = 1-float(np.mean(ious))

    zref = np.load(evidence / "geometry/depth.npy")
    zren = np.load(rendered / "depth.npy")
    trusted = np.load(evidence / "geometry/valid.npy").astype(bool)
    ren_valid = np.load(rendered / "depth_valid.npy").astype(bool)
    valid = trusted & ren_valid & np.isfinite(zref) & (zref > 0)
    scale = float(np.median(zref[valid]/zren[valid])) if valid.any() else None
    if scale is not None:
        relative = np.abs(zref[valid]-zren[valid]*scale)/zref[valid]
        coverage = valid.sum()/max(trusted.sum(), 1)
        metrics["relative_depth"] = available(np.mean(relative), alignment="positive median scale, no shift", scale=scale,
                                                valid_pixels=int(valid.sum()), coverage=float(coverage))
        errors["depth"] = min(float(np.mean(relative)), 1.) + (1-coverage)
    else:
        metrics["relative_depth"] = missing("no overlapping valid depth pixels")
        errors["depth"] = 2.0  # absent reconstructed geometry must not improve selection
    pairs = []
    medians = {}
    for oid, (a, b) in masks.items():
        av, bv = a & trusted, b & ren_valid
        if av.sum() >= 10 and bv.sum() >= 10:
            medians[oid] = (float(np.median(zref[av])), float(np.median(zren[bv])))
    names = sorted(medians)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            ra, pa = medians[a]; rb, pb = medians[b]
            if abs(ra-rb) <= .05*max(ra, rb):
                continue
            pairs.append({"a": a, "b": b, "reference_delta": ra-rb, "render_delta": pa-pb,
                          "correct": bool((ra-rb)*(pa-pb) > 0)})
    metrics["depth_order"] = available(np.mean([p["correct"] for p in pairs]), pairs=pairs) if pairs else missing("no confident distinct object-depth pairs")
    surfaces = read(evidence / "surface_geometry.json")
    if surfaces.get("contract") != CONTRACT:
        raise ValueError("Expected surface-plane normal evidence")
    surface_score = compare_surface_geometry(surfaces, zren, ren_valid,
        read(rendered / "metadata.json")["camera"], {oid: b for oid, (_, b) in masks.items()})
    metrics["surface_normal_degrees"] = surface_score
    if surface_score["expected_surfaces"]:
        errors["normal"] = surface_score["selection_component"]
    diffuse_valid = np.load(evidence / "intrinsics/diffuse_valid.npy").astype(bool) & valid
    albedo_pred = np.load(rendered / "albedo.npy")
    diffuse_valid &= albedo_pred.min(-1) > .02
    ar = np.load(evidence / "intrinsics/albedo.npy")
    if diffuse_valid.any():
        value = float(np.mean(np.abs(ar[diffuse_valid]-albedo_pred[diffuse_valid])))
        metrics["albedo_linear_mae"] = available(value, pixels=int(diffuse_valid.sum()), constraint="soft")
        errors["albedo"] = value
    else:
        metrics["albedo_linear_mae"] = missing("no trusted diffuse overlap")
    sr = np.load(evidence / "intrinsics/shading.npy")
    sp = np.load(rendered / "shading.npy")
    shading_scale = positive_scale(sr, sp, diffuse_valid)
    if shading_scale is not None:
        error = np.mean(np.abs(sr[diffuse_valid]-sp[diffuse_valid]*shading_scale))
        denom = max(float(np.mean(np.abs(sr[diffuse_valid]))), 1e-6)
        metrics["shading_linear_nmae"] = available(error/denom, scale=shading_scale,
                                                   alignment="one positive scalar shared by RGB; no per-channel recoloring")
        errors["shading"] = min(float(error/denom), 2.)
        preview(out / "shading_error.png", np.mean(np.abs(sr-sp*shading_scale), -1), "scalar")
    else:
        metrics["shading_linear_nmae"] = missing("no trusted diffuse overlap or zero diffuse lighting")
    error_rgb = np.abs(ref-rgb)
    mae = float(np.mean(error_rgb)); mse = float(np.mean((ref-rgb)**2))
    metrics["rgb_srgb_mae"] = available(mae)
    metrics["rgb_psnr"] = available(-10*np.log10(max(mse, 1e-12)), domain="display sRGB")
    errors["rgb"] = mae
    preview(out / "rgb_error.png", error_rgb, "srgb")
    groups = {}
    total = np.zeros_like(ref)
    for p in sorted(rendered.glob("light_*.npy")):
        x = np.load(p)
        name = p.stem[len("light_"):]
        total += x
        luma = x @ np.array([.2126, .7152, .0722])
        groups[name] = {"mean_linear_luminance": float(np.mean(luma)),
                        "by_object": {oid: float(np.mean(luma[a])) for oid, (a, _) in masks.items() if a.any()}}
    combined = np.load(rendered / "rgb_linear.npy")
    closure = float(np.mean(np.abs(total-combined))/max(float(np.mean(np.abs(combined))), 1e-6))
    metrics["light_contributions"] = {"available": bool(groups), "groups": groups,
                                      "additive_closure_nmae": closure, "reference_per_light_ground_truth": False}
    daylight = {}
    for gid, group in read(rendered / "metadata.json").get("illumination_groups", {}).items():
        members = [groups[lid] for lid in group["members"] if lid in groups]
        complete = len(members) == len(group["members"]) and bool(members)
        daylight[gid] = {**group, "available": complete,
                        "mean_linear_luminance": sum(m["mean_linear_luminance"] for m in members) if complete else None,
                        "by_object": {oid: sum(m["by_object"].get(oid, 0.) for m in members)
                                      for oid, (a, _) in masks.items() if a.any()} if complete else {},
                        "reference_per_group_ground_truth": False}
    # Aggregates of existing passes, never added to the closure sum a second time.
    metrics["illumination_group_contributions"] = daylight
    # Fixed weights and explicit penalties prevent missing evidence from silently scoring as perfect.
    weights = config.get("metric_weights", {"silhouette": 2., "depth": 1., "normal": .5, "albedo": .5, "shading": 1., "rgb": 1.})
    missing_penalties = {"silhouette": 1., "depth": 2., "normal": 1., "albedo": 1., "shading": 2., "rgb": 1.}
    weighted = sum(w*errors.get(k, missing_penalties[k]) for k, w in weights.items())/sum(weights.values())
    result = {"normal_contract": CONTRACT,
              "render_settings": {key: render_meta[key] for key in
                                  ("samples", "denoising", "denoiser", "use_adaptive_sampling",
                                   "adaptive_threshold", "adaptive_min_samples", "rgb_output", "rgb_linear_source")
                                  if key in render_meta},
              "metrics": metrics, "selection_loss": float(weighted), "lower_is_better": True,
              "weights": weights, "component_losses": errors,
              "missing_component_penalties": {k: missing_penalties[k] for k in weights if k not in errors},
              "warnings": ["Single-image geometry/intrinsics/HDRI are estimates; these scores are not physical ground truth."]}
    write(out / "scores.json", result)
    items = [("reference", evidence / "reference.png"), ("render", rendered / "rgb.png"), ("RGB error", out / "rgb_error.png")]
    for k in ("depth", "albedo", "shading"):
        items.append(("render " + k, rendered / f"{k}.png"))
    items += [(p.stem, p) for p in sorted(rendered.glob("light_*.png"))]
    for oid in masks:
        items.append((oid + " rendered", rendered / "masks" / f"{oid}.png"))
    contact_sheet(items, out / "comparison.png")
    return result
