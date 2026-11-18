from __future__ import annotations
import fcntl
import shutil
import traceback
from pathlib import Path

from .codecheck import validate_code
from .evidence import contact_sheet
from .io import ROOT, read, write, sha, run_process
from .llm import Client, ModelOutputError
from .metrics import postprocess, evaluate
from .object_review import ReviewValidationError, object_memory, validate_review
from .prompts import BUILD, REVIEW
from .state import apply_patch, validate
from .surface_geometry import CONTRACT


def render_settings(options):
    """Resolve settings; absent quality flags preserve legacy run configurations."""
    return {"samples": options.get("samples", 512),
            "device": options.get("device", "CPU"),
            "exposure": options.get("exposure", 0.),
            "depth_type": options.get("depth_type", "camera_z"),
            "use_denoising": options.get("use_denoising", False),
            "denoiser": options.get("denoiser", "OPENIMAGEDENOISE"),
            "use_adaptive_sampling": options.get("use_adaptive_sampling", False),
            "adaptive_threshold": options.get("adaptive_threshold", .01),
            "adaptive_min_samples": options.get("adaptive_min_samples", 0)}


def render(code, state, evidence, output, config):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    hdr = evidence / "environment/environment.exr"
    job = {"code": str(code), "state": str(state), "camera": str(evidence / "camera.json"),
           "priors": str(evidence / "object_priors.json"), "output": str(output),
           "hdr": str(hdr) if hdr.exists() else None, "seed": config["seed"],
           **render_settings(config["render"])}
    write(output / "job.json", job)
    run_process([config["blender"], "-b", "-t", str(config["render"].get("threads", 8)),
                 "--python-exit-code", "1", "--python", ROOT / "lightagent/blender_host.py", "--", output / "job.json"],
                output / "blender.log", timeout=config["render"].get("timeout", 900),
                env={"CUDA_VISIBLE_DEVICES": str(config["render"].get("gpu", config["gpu"]))})
    postprocess(output)


def verify_evidence(evidence):
    manifest = read(evidence / "manifest.json")
    for file, info in manifest["files"].items():
        if sha(evidence / file) != info["sha256"]:
            raise ValueError(f"Immutable evidence changed: {file}")
    return manifest


def next_folder(parent, prefix):
    i = 0
    while (parent / f"{prefix}_{i:03d}").exists():
        i += 1
    return parent / f"{prefix}_{i:03d}"


def evidence_images(evidence):
    images = [("reference", evidence / "reference.png"), ("mask and geometry overview", evidence / "overview.png"),
              ("albedo linear-to-sRGB preview", evidence / "intrinsics/albedo.png"),
              ("diffuse shading Reinhard preview; compare original linear arrays in scores", evidence / "intrinsics/shading.png"),
              ("relative camera Z-depth; surface plane parameters are in surface_geometry JSON", evidence / "geometry/depth.png")]
    hdri = evidence / "environment/environment.png"
    if hdri.exists():
        images.append(("HDR environment Reinhard preview; source basis and radiance statistics are in manifest", hdri))
    return images


def export_round_comparison(run, completed, best):
    """Compare the accepted render of every completed round in chronological order."""
    items = [("Reference | loss: lower is better", run / "evidence/reference.png")]
    for entry in sorted(completed, key=lambda item: item["round"]):
        index = entry["round"]
        folder = run / f"round_{index:03d}"
        attempt = folder / read(folder / "accepted.json")["attempt"]
        label = f"Round {index:03d} | loss={entry['loss']:.6f}"
        if index == best:
            label += " | BEST"
        items.append((label, attempt / "render/rgb.png"))
    dest = run / "final"
    dest.mkdir(exist_ok=True)
    contact_sheet(items, dest / "rounds_comparison.png", width=512, height=384,
                  cols=min(3, len(items)))


def export_final(run, completed, best):
    source = run / f"round_{best:03d}"
    accepted = read(source / "accepted.json")
    attempt = source / accepted["attempt"]
    dest = run / "final"
    dest.mkdir(exist_ok=True)
    for src, name in [(source / "scene.py", "geometry.py"), (source / "state_after.json", "state.json"),
                      (run / "evidence/camera.json", "camera.json"), (run / "evidence/object_priors.json", "object_priors.json"),
                      (run / "evidence/surface_geometry.json", "surface_geometry.json"),
                      (attempt / "render/scene.blend", "scene.blend"), (attempt / "render/rgb.png", "render.png"),
                      (attempt / "scores/scores.json", "scores.json"), (ROOT / "lightagent/blender_host.py", "runtime.py"),
                      (ROOT / "lightagent/light_rules.py", "light_rules.py")]:
        shutil.copyfile(src, dest / name)
    hdr = run / "evidence/environment/environment.exr"
    if hdr.exists():
        shutil.copyfile(hdr, dest / "environment.exr")
    config = read(run / "config.json")
    # Export the selected render's settings, even if the run config changed later.
    job_path = attempt / "render/job.json"
    selected_job = read(job_path) if job_path.exists() else {**config["render"], "seed": config["seed"]}
    write(dest / "settings.json", {**render_settings(selected_job), "seed": selected_job["seed"]})
    (dest / "scene.py").write_text('''# Standalone entry: blender -b --python scene.py
from pathlib import Path
import json
import runpy
root = Path(__file__).resolve().parent
job = json.loads((root / "settings.json").read_text())
job.update(code=str(root / "geometry.py"), state=str(root / "state.json"),
           camera=str(root / "camera.json"), priors=str(root / "object_priors.json"),
           hdr=str(root / "environment.exr") if (root / "environment.exr").exists() else None,
           output=str(root / "rerender"))
runtime = runpy.run_path(str(root / "runtime.py"))
runtime["main"](job)
''')
    write(dest / "selection.json", {"best_round": best, "last_round": completed[-1]["round"],
                                    "selection": "minimum fixed-weight selection_loss among completed rounds",
                                    "round_losses": completed, "state_matches_selected_scene": True,
                                    "note": "run/state.json is the latest state; final/state.json belongs to best_round"})
    export_round_comparison(run, completed, best)


def iterate(run, rounds=None, replay_dir=None):
    run = Path(run).resolve()
    config = read(run / "config.json")
    rounds = rounds if rounds is not None else config["rounds"]
    if rounds < 1:
        raise ValueError("rounds must be positive")
    with (run / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _iterate_locked(run, config, rounds, Path(replay_dir) if replay_dir else None)


def _iterate_locked(run, config, rounds, replay):
    evidence = run / "evidence"
    manifest = verify_evidence(evidence)
    if manifest.get("normal_contract") != CONTRACT:
        raise ValueError("Legacy normal evidence cannot be mixed with surface-plane scoring. "
                         "Use scripts/fork_surface_run.py SOURCE_RUN NEW_RUN to reuse evidence without model inference.")
    surface_geometry = read(evidence / "surface_geometry.json")
    if surface_geometry.get("contract") != CONTRACT:
        raise ValueError("Surface geometry contract mismatch")
    state = validate(read(run / "initial_state.json"))
    history, completed = [], []
    previous_code = None
    client = Client(config["llm"])
    for index in range(rounds):
        folder = run / f"round_{index:03d}"
        folder.mkdir(exist_ok=True)
        if (folder / "done.json").exists():
            done = read(folder / "done.json")
            state = validate(read(folder / "state_after.json"))
            if sha(folder / "state_after.json") != done["state_sha256"] or sha(folder / "scene.py") != done["code_sha256"]:
                raise ValueError("A completed round was modified")
            previous_code = (folder / "scene.py").read_text()
            history.append(read(folder / "history_entry.json"))
            completed.append({"round": index, "loss": done["selection_loss"]})
            continue
        print(f"round {index:03d}: building/reviewing", flush=True)
        write(folder / "state_before.json", state)
        write(folder / "history_before.json", history)
        context = {"round": index, "evidence_manifest": manifest, "state": state,
                   "surface_geometry": surface_geometry,
                   "inventory": read(evidence / "inventory.json"), "object_priors": read(evidence / "object_priors.json"),
                   "mask_status": read(evidence / "masks/index.json"), "history": history, "previous_code": previous_code,
                   "object_memory": object_memory(state["objects"], history)}
        images = evidence_images(evidence)
        if index:
            previous = run / f"round_{index-1:03d}"
            prev_attempt = previous / read(previous / "accepted.json")["attempt"]
            images += [("previous render", prev_attempt / "render/rgb.png"),
                       ("previous comparison and light contributions", prev_attempt / "scores/comparison.png")]
        # An accepted render survives interrupted reviews and is reused on resume.
        if (folder / "accepted.json").exists():
            attempt = folder / read(folder / "accepted.json")["attempt"]
            candidate = validate(read(attempt / "candidate_state.json"))
            if candidate["revision"] != state["revision"]+1:
                raise ValueError("Resumed candidate has wrong revision")
            scores = read(attempt / "scores/scores.json")
        else:
            errors = []
            for repair in range(config.get("max_repairs", 2)+1):
                attempt = next_folder(folder, "attempt")
                attempt.mkdir()
                try:
                    replay_file = replay / f"build_{index:03d}.json" if replay else None
                    answer = client.ask(attempt / "build", BUILD, {**context, "failed_attempts": errors}, images, replay=replay_file)
                    code = validate_code(answer["scene_code"])
                    candidate = apply_patch(state, answer["state_patch"], index)
                    write(attempt / "candidate_state.json", candidate)
                    (attempt / "scene.py").write_text(code)
                    render(attempt / "scene.py", attempt / "candidate_state.json", evidence, attempt / "render", config)
                    scores = evaluate(evidence, attempt / "render", attempt / "scores", config)
                    write(folder / "accepted.json", {"attempt": attempt.name})
                    break
                except Exception as exc:
                    error = {"attempt": attempt.name, "error": str(exc), "traceback": traceback.format_exc()}
                    log = attempt / "render/blender.log"
                    if log.exists():
                        error["blender_log_tail"] = log.read_text(errors="replace")[-14000:]
                    write(attempt / "failure.json", error)
                    errors.append(error)
                    if repair == config.get("max_repairs", 2):
                        write(run / "progress.json", {"status": "failed", "round": index, "error": str(exc), "completed_rounds": len(completed)})
                        raise
        review_context = {"round": index, "scores": scores, "state": candidate, "previous_history": history,
                          "object_memory": context["object_memory"], "previous_code": previous_code,
                          "mask_status": context["mask_status"],
                          "surface_geometry": surface_geometry,
                          "evidence_manifest": manifest, "object_priors": context["object_priors"],
                          "render_metadata": read(attempt / "render/metadata.json"),
                          "scene_code": (attempt / "scene.py").read_text()}
        review_errors = []
        for repair in range(config.get("max_repairs", 2)+1):
            review_folder = next_folder(folder, "review")
            try:
                review = client.ask(review_folder, REVIEW, {**review_context, "failed_reviews": review_errors},
                                    images + [("current render", attempt / "render/rgb.png"),
                                              ("diagnostic comparisons", attempt / "scores/comparison.png")],
                                    replay=replay / f"review_{index:03d}.json" if replay else None)
                validate_review(review, candidate["objects"], history)
                break
            except Exception as exc:
                error = {"review": review_folder.name, "error": str(exc), "traceback": traceback.format_exc()}
                write(review_folder / "failure.json", error)
                review_errors.append(error)
                # Parse/contract errors may be repaired. Do not implicitly retry transport failures.
                if not isinstance(exc, (ModelOutputError, ReviewValidationError)) or repair == config.get("max_repairs", 2):
                    write(run / "progress.json", {"status": "failed", "stage": "review", "round": index,
                                                  "error": str(exc), "completed_rounds": len(completed)})
                    raise
        entry = {"round": index, "state_revision": candidate["revision"], "scores": scores, "review": review,
                 "archive": str(folder.relative_to(run)), "review_archive": review_folder.name}
        write(folder / "history_entry.json", entry)
        write(folder / "state_after.json", candidate)
        shutil.copyfile(attempt / "scene.py", folder / "scene.py")
        write(folder / "done.json", {"selection_loss": scores["selection_loss"],
                                     "state_sha256": sha(folder / "state_after.json"), "code_sha256": sha(folder / "scene.py")})
        state = candidate
        previous_code = (folder / "scene.py").read_text()
        history.append(entry)
        completed.append({"round": index, "loss": scores["selection_loss"]})
        write(run / "state.json", state)
        write(run / "history.json", history)
        write(run / "progress.json", {"status": "running", "completed_rounds": index+1})
        best = min(completed, key=lambda item: item["loss"])["round"]
        export_final(run, completed, best)
    # Reconstruct root state after a crash between done.json and state.json.
    write(run / "state.json", state)
    write(run / "history.json", history)
    export_final(run, completed, min(completed, key=lambda item: item["loss"])["round"])
    write(run / "progress.json", {"status": "complete", "completed_rounds": len(completed)})
