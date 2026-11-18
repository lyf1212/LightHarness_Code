from __future__ import annotations
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageOps, ImageDraw

from .coordinates import CV_TO_WORLD, camera_contract, canonicalize_environment, normals_from_z, rays, resize_affine
from .io import ROOT, read, write, sha, preview, load_hdr, save_hdr
from .llm import Client
from .prompts import INVENTORY
from .state import initialize
from .resident import run_model
from .surface_geometry import CONTRACT, save_surface_geometry


def worker(kind, config, image, output, job_extra=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    job = {"image": str(image), "output": str(output), "options": config[kind], "seed": config["seed"], **(job_extra or {})}
    write(output / "job.json", job)
    run_model(kind, job, output / "worker.log", config)


def contact_sheet(items, dest, width=320, height=240, cols=3):
    rows = (len(items)+cols-1)//cols
    sheet = Image.new("RGB", (cols*width, max(1, rows)*(height+28)), "#242424")
    draw = ImageDraw.Draw(sheet)
    for i, (label, file) in enumerate(items):
        im = Image.open(file).convert("RGB")
        im.thumbnail((width, height))
        x, y = i % cols*width, i//cols*(height+28)
        sheet.paste(im, (x+(width-im.width)//2, y+28+(height-im.height)//2))
        draw.text((x+5, y+7), label[:45], fill="white")
    sheet.save(dest)


def finish_geometry(evidence, config):
    g = evidence / "geometry"
    pred = read(g / "vggt.json")
    K = np.array(pred["K"])
    depth = np.load(g / "depth_raw.npy")
    confidence = np.load(g / "depth_confidence.npy")
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
    if not valid.any():
        raise ValueError("VGGT returned no valid geometry")
    threshold = float(np.quantile(confidence[valid], config.get("depth_confidence_quantile", .5)))
    valid &= confidence >= threshold
    scale = config.get("scene_median_depth", 4.) / float(np.median(depth[valid]))
    z = depth*scale
    np.save(g / "depth.npy", z)
    np.save(g / "valid.npy", valid)
    normal, normal_valid = normals_from_z(z, K, valid)
    np.save(g / "normals_cv.npy", normal)
    np.save(g / "normals_valid.npy", normal_valid)
    camera = camera_contract(K, pred["extrinsic"], z.shape, scale)
    camera["confidence_threshold"] = threshold
    write(evidence / "camera.json", camera)
    preview(g / "depth.png", z, "depth")
    points = (rays(K, z.shape)*z[..., None]) @ CV_TO_WORLD.T
    priors = {}
    for oid, info in read(evidence / "masks/index.json").items():
        if not info["available"]:
            priors[oid] = {"available": False, "reason": "SAM3 did not identify a unique matching instance"}
            continue
        selected = np.load(evidence / info["path"]).astype(bool) & valid
        p = points[selected]
        if len(p) < 10:
            priors[oid] = {"available": False, "reason": "too few high-confidence depth pixels"}
            continue
        priors[oid] = {"available": True, "visible_surface_center_world": np.median(p, 0).tolist(),
                       "visible_surface_p05_world": np.percentile(p, 5, axis=0).tolist(),
                       "visible_surface_p95_world": np.percentile(p, 95, axis=0).tolist(),
                       "median_z": float(np.median(z[selected])), "pixels": int(selected.sum()),
                       "note": "visible surface bounds, not a complete 3D object box"}
    write(evidence / "object_priors.json", priors)


def environment(evidence, config):
    opt = config["lumen"]
    out = evidence / "environment"
    out.mkdir()
    mode = opt["mode"]
    if mode == "disabled":
        info = {"available": False, "reason": "explicitly disabled by configuration; no LUMEN evidence"}
        write(out / "metadata.json", info)
        return info
    if mode == "import":
        source = Path(opt["hdr_path"])
        if not opt.get("forward_world") or not opt.get("up_world") or opt.get("longitude_sign") not in {-1, 1}:
            raise ValueError("Imported HDRI requires explicit forward_world, up_world and longitude_sign (+1/-1)")
        forward, up = opt["forward_world"], opt["up_world"]
    elif mode == "official":
        for name in ("python", "repo", "flux_checkpoint", "lora_checkpoint", "redux_checkpoint"):
            if not opt.get(name) or not Path(opt[name]).exists():
                raise FileNotFoundError(f"LUMEN is not installed/configured: lumen.{name}. See README; import HDRI is also supported.")
        inp, raw = out / "input", out / "raw"
        inp.mkdir(); raw.mkdir()
        shutil.copyfile(evidence / "reference.png", inp / "reference.png")
        job = {"image": str(inp / "reference.png"), "output": str(raw), "camera": str(evidence / "camera.json"),
               "options": opt, "seed": config["seed"], "vggt_repo": config["vggt"].get("repo")}
        write(out / "job.json", job)
        run_model("lumen", job, out / "lumen.log", config)
        source = raw / "reference.exr"
        if not source.exists():
            source = raw / "reference.npy"
        forward, up = [0, 1, 0], [0, 0, 1]
    else:
        raise ValueError("lumen.mode must be official/import/disabled")
    hdr = load_hdr(source)
    if hdr.shape[1] != 2*hdr.shape[0]:
        raise ValueError("Expected a 2:1 equirectangular HDRI")
    # Preserve raw source and basis; never change evidence when the agent rotates World.
    shutil.copyfile(source, out / ("source" + source.suffix))
    longitude_sign = opt["longitude_sign"] if mode == "import" else -1
    canonical = canonicalize_environment(hdr, forward, up, longitude_sign)
    save_hdr(out / "environment.exr", canonical)
    np.save(out / "environment.npy", canonical)
    preview(out / "environment.png", canonical, "hdr")
    info = {"available": True, "provider": "LUMEN" if mode == "official" else opt.get("provider", "imported_HDRI"),
            "source_sha256": sha(source), "source_forward_world": forward, "source_up_world": up,
            "source_longitude_sign": longitude_sign,
            "canonical_forward_world": [1, 0, 0], "canonical_up_world": [0, 0, 1],
            "canonical_increasing_u_world": [0, -1, 0],
            "projection": "equirectangular", "color_space": "linear", "radiometric_scale_known": False,
            "probe_position_world": opt.get("probe_position_world", [0, 0, 0]),
            "hdr": "environment/environment.exr", "preview": "environment/environment.png",
            "note": "camera-local illumination estimate; distant-World use is an approximation"}
    luminance = canonical @ np.array([.2126, .7152, .0722])
    h, w = luminance.shape
    flat = np.argsort(luminance.reshape(-1))[-min(16, luminance.size):][::-1]
    peaks = []
    for index in flat:
        y, x = np.unravel_index(index, (h, w))
        lon, lat = ((x+.5)/w-.5)*2*np.pi, (.5-(y+.5)/h)*np.pi
        peaks.append({"pixel": [int(x), int(y)], "direction_world": [float(np.cos(lat)*np.cos(lon)),
                       float(-np.cos(lat)*np.sin(lon)), float(np.sin(lat))],
                      "linear_rgb": canonical[y, x].tolist(), "luminance": float(luminance[y, x])})
    info["radiance_statistics"] = {"luminance_p50_p90_p99_max": [*np.percentile(luminance, [50, 90, 99]).tolist(), float(luminance.max())],
                                    "brightest_pixels": peaks, "note": "adjacent peaks may belong to the same source"}
    write(out / "metadata.json", info)
    return info


def finalize_evidence(evidence, config, provenance="real"):
    im = np.asarray(Image.open(evidence / "reference.png").convert("RGB"), np.float32)/255
    h, w = im.shape[:2]
    intr = evidence / "intrinsics"
    arrays = {k: np.load(intr / f"{k}.npy") for k in ("albedo", "shading", "residual")}
    for k, x in arrays.items():
        if x.shape != im.shape or not np.isfinite(x).all():
            raise ValueError(f"{k} is not aligned/finite: {x.shape} vs {im.shape}")
        preview(intr / f"{k}.png", x, "hdr" if k != "albedo" else "linear")
    # Conservative diffuse-region heuristic, explicitly not a learned uncertainty map.
    residual = np.mean(np.abs(arrays["residual"]), -1)
    valid = (im.max(-1) < .98) & (im.min(-1) > .02) & (arrays["albedo"].min(-1) > .03)
    valid &= residual < config.get("residual_threshold", .12)
    valid &= np.load(evidence / "geometry/valid.npy")
    np.save(intr / "diffuse_valid.npy", valid)
    surfaces = save_surface_geometry(evidence, seed=config.get("seed", 0),
                                     source="VGGT depth" if provenance == "real" else provenance)
    items = [("reference sRGB", evidence / "reference.png"),
             ("Z-depth (relative gauge)", evidence / "geometry/depth.png")]
    items += [(k + " (display preview)", intr / f"{k}.png") for k in arrays]
    env = read(evidence / "environment/metadata.json")
    if env["available"]:
        items.append(("HDRI (Reinhard preview)", evidence / env["preview"]))
    for oid, item in read(evidence / "masks/index.json").items():
        if item["available"]:
            items.append((oid, evidence / "masks" / f"{oid}.png"))
    contact_sheet(items, evidence / "overview.png")
    manifest = {"schema_version": 2, "normal_contract": CONTRACT,
                "provenance": provenance, "width": w, "height": h,
                "pixel_grid": "reference.png after EXIF and one resize; all dense evidence restored to this grid",
                "camera": read(evidence / "camera.json"), "environment": env,
                "diffuse_mask": "heuristic: non-saturated, non-dark, low residual, valid geometry",
                "normal_source": surfaces["source"],
                "normal_reference": "surface_geometry.json: reliable surface points/normals in Blender world; no normal images sent",
                "dense_normals": "internal depth-derived arrays for support-direction fallback; not model image inputs",
                "rgb_assumption": "input treated as sRGB display-referred; inverse transfer is not inverse camera tone mapping",
                "files": {}}
    for f in sorted(evidence.rglob("*")):
        if f.is_file() and f.name != "manifest.json":
            manifest["files"][str(f.relative_to(evidence))] = {"sha256": sha(f), "bytes": f.stat().st_size}
    write(evidence / "manifest.json", manifest)
    return manifest


def prepare(image, run, config, inventory_file=None):
    run = Path(run)
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("Prepare requires a new empty run directory")
    evidence = run / "evidence"
    evidence.mkdir(parents=True)
    write(run / "config.json", config)
    image = Path(image)
    oriented = ImageOps.exif_transpose(Image.open(image)).convert("RGB")
    original_size = oriented.size
    oriented.thumbnail((config["max_side"], config["max_side"]), Image.Resampling.LANCZOS)
    oriented.save(evidence / "reference.png")
    write(evidence / "input.json", {"source": str(image.resolve()), "source_sha256": sha(image),
                                   "exif_applied": True, "oriented_size": list(original_size),
                                   "canonical_size": list(oriented.size),
                                   "oriented_to_canonical": resize_affine(*original_size, *oriented.size).tolist()})
    client = Client(config["llm"])
    inventory = client.ask(run / "prepare/inventory", INVENTORY, {"image_size": list(oriented.size),
                           "configured_hdri_mode": config["lumen"]["mode"],
                           "initial_world_policy": "use evidence_only; enable World later only with calibrated HDRI and evidence"},
                           [("reference", evidence / "reference.png")], replay=inventory_file)
    state = initialize(inventory)
    write(evidence / "inventory.json", inventory)
    write(run / "initial_state.json", state)
    write(run / "state.json", state)
    # Each image follows its dependencies; other images share the resident model pool.
    worker("sam3", config, evidence / "reference.png", evidence / "masks", {"inventory": str(evidence / "inventory.json")})
    worker("vggt", config, evidence / "reference.png", evidence / "geometry")
    finish_geometry(evidence, config)
    worker("marigold", config, evidence / "reference.png", evidence / "intrinsics")
    environment(evidence, config)
    finalize_evidence(evidence, config)
    write(run / "progress.json", {"status": "prepared", "completed_rounds": 0})
