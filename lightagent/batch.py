"""Concurrent independent scenes sharing persistent, per-model GPU services."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import fcntl
from pathlib import Path
import re
import time

from .io import ROOT, inside, read, sha, write
from .resident import KINDS, gpu_targets


def check_models(config):
    for kind in KINDS:
        gpu_targets(config, kind)
    if config["lumen"]["mode"] == "official":
        for key in ("python", "repo", "flux_checkpoint", "lora_checkpoint", "redux_checkpoint"):
            value = config["lumen"].get(key)
            if not value or not Path(value).exists():
                raise FileNotFoundError(f"LUMEN missing {key}: configure it, import HDRI, or use lumen.mode=disabled")


def load_batch(path, config, parallelism=None, rounds=None):
    path = Path(path).resolve()
    spec = read(path)
    if isinstance(spec, list):
        spec = {"images": spec}
    if not isinstance(spec, dict) or not isinstance(spec.get("images"), list) or not spec["images"]:
        raise ValueError("Batch JSON requires a nonempty images list")
    config = deepcopy(config)
    for kind, targets in spec.get("model_gpus", {}).items():
        if kind not in KINDS:
            raise ValueError(f"Unknown model in model_gpus: {kind}")
        config[kind]["gpu_pool"] = targets if isinstance(targets, list) else [targets]
    count = parallelism if parallelism is not None else spec.get("parallelism", 2)
    rounds = rounds if rounds is not None else spec.get("rounds", config["rounds"])
    for name, value in (("parallelism", count), ("rounds", rounds)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    items, ids = [], set()
    for index, entry in enumerate(spec["images"]):
        entry = {"image": entry} if isinstance(entry, str) else entry
        if not isinstance(entry, dict) or not isinstance(entry.get("image"), str):
            raise ValueError(f"images[{index}] requires an image path")
        image = Path(entry["image"])
        image = (path.parent / image).resolve() if not image.is_absolute() else image.resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        name = entry.get("id", f"image_{index:03d}")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in ids:
            raise ValueError(f"Invalid or duplicate image id: {name}")
        ids.add(name)
        item = {"id": name, "image": str(image), "sha256": sha(image)}
        if entry.get("inventory"):
            inventory = (path.parent / entry["inventory"]).resolve()
            if not inventory.is_file():
                raise FileNotFoundError(inventory)
            item.update(inventory=str(inventory), inventory_sha256=sha(inventory))
        items.append(item)
    check_models(config)
    return config, items, count, rounds


def run_scene(item, root, config, rounds, resume):
    # Each attempt is immutable once abandoned. A prepared attempt resumes the
    # existing round-level checkpoints; incomplete preparation gets a fresh folder.
    from .evidence import prepare
    from .pipeline import iterate
    scene = inside(root, root / item["id"])
    scene.mkdir(exist_ok=True)
    attempts = sorted(scene.glob("attempt_[0-9]*"))
    run = attempts[-1] if resume and attempts else None
    prepared = run is not None and (run / "evidence/manifest.json").is_file() and (run / "initial_state.json").is_file()
    if not prepared:
        index = max((int(p.name.split("_")[1]) for p in attempts), default=-1) + 1
        run = scene / f"attempt_{index:03d}"
    else:
        saved = read(run / "config.json")
        if saved != config:
            raise ValueError(f"Cannot change config for prepared scene {item['id']}; use a new batch directory")
    status = {"id": item["id"], "image": item["image"], "run": str(run), "status": "running"}
    write(scene / "status.json", status)
    started = time.monotonic()
    try:
        print(f"[{item['id']}] {'resume' if prepared else 'prepare'} {run}", flush=True)
        if not prepared:
            prepare(item["image"], run, deepcopy(config), item.get("inventory"))
        iterate(run, rounds)
        status.update(status="completed", final=str(run / "final"))
    except Exception as exc:
        # One failed image does not cancel independent images.
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    status["seconds"] = time.monotonic() - started
    write(scene / "status.json", status)
    print(f"[{item['id']}] {status['status']}", flush=True)
    return status


def run_batch(path, run, config, *, parallelism=None, rounds=None, resume=False):
    config, items, count, rounds = load_batch(path, config, parallelism, rounds)
    root = inside(ROOT, run)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".batch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = root / "batch.json"
        if manifest.exists():
            if not resume:
                raise FileExistsError("Batch already exists; use --resume or a new --run directory")
            previous = read(manifest)
            if previous["images"] != items or previous["config"] != config:
                raise ValueError("Resume requires unchanged images and model config; use a new batch directory")
        elif any(p.name != ".batch.lock" for p in root.iterdir()):
            raise FileExistsError("Batch requires an empty output directory")
        write(manifest, {"images": items, "config": config, "parallelism": count, "rounds": rounds})
        results = {item["id"]: {"id": item["id"], "status": "pending"} for item in items}
        def save_summary(status):
            write(root / "summary.json", {"status": status, "parallelism": count, "rounds": rounds,
                                          "images": list(results.values())})
        save_summary("running")
        try:
            with ThreadPoolExecutor(max_workers=count, thread_name_prefix="scene") as executor:
                futures = {executor.submit(run_scene, item, root, config, rounds, resume): item for item in items}
                for future in as_completed(futures):
                    item = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"id": item["id"], "status": "failed", "error": str(exc)}
                    results[item["id"]] = result
                    save_summary("running")
        except BaseException:
            save_summary("interrupted")
            raise
        ok = all(result["status"] == "completed" for result in results.values())
        save_summary("completed" if ok else "failed")
        return ok
