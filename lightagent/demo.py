"""Deterministic integration fixture: real Blender, synthetic evidence, replayed agents."""
from pathlib import Path
import copy
import shutil

import numpy as np

from .coordinates import camera_contract
from .evidence import finalize_evidence
from .io import read, write, save_hdr, preview
from .pipeline import render, iterate
from .state import initialize


CODE = '''import bpy
def build_scene(ctx):
    def box(name, oid, location, scale, color):
        bpy.ops.mesh.primitive_cube_add(size=1, location=location)
        obj = bpy.context.object
        obj.name = name
        obj.scale = scale
        obj.data.materials.append(ctx.material(name + "_mat", color))
        ctx.register(obj, oid)
        return obj
    box("cube", "obj_cube", (0.0, 4, 0), (1.5, 1.5, 1.5), (0.45, 0.12, 0.06))
    box("floor", "obj_floor", (0, 5, -1), (12, 14, 0.2), (0.55, 0.55, 0.55))
    box("wall", "obj_wall", (0, 8, 2), (12, 0.2, 8), (0.65, 0.65, 0.65))
'''


def inventory():
    def light(lid, position, direction, energy, color):
        return {"id": lid, "label": lid, "source_type": "artificial", "visibility": "off_frame", "status": "active", "confidence": 1.,
                "evidence": [{"source": "synthetic_fixture", "observation": "known light"}], "occluded_by": [],
                "environment_relation": "independent", "elements": [
                    {"id": "area", "kind": "AREA", "position": position, "direction": direction,
                     "energy": energy, "color": color, "size": [1.5, 1.5], "enabled": True}]}
    return {"objects": [
        {"id": "obj_cube", "label": "cube", "prompt": "cube", "role": "object", "bbox": [.25, .2, .7, .8], "confidence": 1.},
        {"id": "obj_floor", "label": "floor", "prompt": "floor", "role": "floor", "bbox": [0., .6, 1., 1.], "confidence": 1.},
        {"id": "obj_wall", "label": "wall", "prompt": "wall", "role": "wall", "bbox": [0., 0., 1., .7], "confidence": 1.}],
        "lights": [light("light_key", [-2, 1, 3], [2, 3, -3], 250., [1., .85, .7]),
                   light("light_fill", [3, 2, 2], [-3, 2, -2], 80., [.7, .85, 1.])],
        "world": {"mode": "world", "strength": .2, "rotation_z": .3}, "analysis": "synthetic integration fixture"}


def demo_review(index):
    """Known corrections for the synthetic fixture; not a model's visual judgment."""
    objects, feedback = {}, []
    for obj in inventory()["objects"]:
        oid = obj["id"]
        displaced = oid == "obj_cube" and index == 0
        objects[oid] = {"difference_level": "significant" if index == 0 else "minor",
                        "observation": ("Cube displaced and lighting too dim" if displaced else "Lighting too dim")
                                       if index == 0 else "Matches the known fixture; no further correction needed"}
        if displaced:
            objects[oid]["region"] = {"difference": "Cube is displaced right",
                                      "suggestion": "Restore cube x from 0.3 to 0.0",
                                      "evidence": "Synthetic known scene coordinates; inspect silhouette IoU", "confidence": 1.}
        if index == 0:
            objects[oid]["color"] = {"difference": "Lighting is too dim",
                                     "suggestion": "Restore light_key energy to 250; preserve base colors",
                                     "evidence": "Known key-light energy and unchanged linear base colors; inspect RGB and shading",
                                     "confidence": 1.}
        if index:
            for aspect in (("region", "color") if oid == "obj_cube" else ("color",)):
                feedback.append({"object_id": oid, "source_round": index-1, "aspect": aspect,
                                 "implementation": "implemented", "outcome": "improved",
                                 "evidence": "Known fixture correction restores cube position and key energy; round 1 matches truth render"})
    return {"summary": "replayed synthetic diagnostic fixture, not a model prediction",
            "next_priorities": ["restore cube position and key-light power"] if index == 0 else [],
            "lighting_changes": [], "geometry_changes": [], "material_changes": [], "uncertainties": [],
            "object_reviews": objects, "history_feedback": feedback}


def make_demo(run, config):
    run = Path(run)
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("Demo requires a new directory")
    run.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(config)
    config.update(max_side=160, rounds=2, max_repairs=0)
    config["render"].update(samples=16, threads=4, device="CPU")
    write(run / "config.json", config)
    e = run / "evidence"
    for folder in (e, e/"geometry", e/"intrinsics", e/"masks", e/"environment"):
        folder.mkdir(exist_ok=True)
    inv = inventory()
    state = initialize(inv)
    write(e / "inventory.json", inv)
    write(run / "initial_state.json", state)
    write(run / "state.json", state)
    camera = camera_contract(np.array([[110., 0, 76.3], [0, 115., 57.7], [0, 0, 1.]]), np.eye(4), (120, 160), 1.)
    write(e / "camera.json", camera)
    write(e / "object_priors.json", {})
    hdr = np.ones((32, 64, 3), np.float32)*.12
    hdr[:, :32] *= np.array([.8, 1, 1.3])
    save_hdr(e / "environment/environment.exr", hdr)
    preview(e / "environment/environment.png", hdr, "hdr")
    write(e / "environment/metadata.json", {"available": True, "provider": "synthetic_fixture_not_LUMEN",
                                            "preview": "environment/environment.png", "hdr": "environment/environment.exr"})
    (run / "truth_geometry.py").write_text(CODE)
    render(run / "truth_geometry.py", run / "state.json", e, run / "truth_render", config)
    tr = run / "truth_render"
    shutil.copyfile(tr / "rgb.png", e / "reference.png")
    for name in ("depth", "normals_cv"):
        shutil.copyfile(tr / f"{name}.npy", e / "geometry" / f"{name}.npy")
    valid = np.load(tr / "depth_valid.npy")
    np.save(e / "geometry/valid.npy", valid)
    np.save(e / "geometry/normals_valid.npy", valid)
    for name in ("depth", "normals"):
        shutil.copyfile(tr / f"{name}.png", e / "geometry" / f"{name}.png")
    for name in ("albedo", "shading"):
        shutil.copyfile(tr / f"{name}.npy", e / "intrinsics" / f"{name}.npy")
    np.save(e / "intrinsics/residual.npy", np.zeros((120, 160, 3), np.float32))
    masks = {}
    for obj in inv["objects"]:
        oid = obj["id"]
        for ext in ("npy", "png"):
            shutil.copyfile(tr / "masks" / f"{oid}.{ext}", e / "masks" / f"{oid}.{ext}")
        masks[oid] = {"id": oid, "available": True, "path": f"masks/{oid}.npy", "source": "synthetic_ground_truth"}
    write(e / "masks/index.json", masks)
    finalize_evidence(e, config, provenance="synthetic_fixture; no SAM3/VGGT/Marigold/LUMEN inference")
    replays = run / "replay"
    replays.mkdir()
    for i in range(2):
        light = copy.deepcopy(inv["lights"][0])
        light["elements"][0]["energy"] = 120. if i == 0 else 250.
        write(replays / f"build_{i:03d}.json", {"rationale": "deterministic test, not a model prediction",
               "scene_code": CODE.replace('(0.0, 4, 0)', '(0.3, 4, 0)') if i == 0 else CODE,
               "state_patch": {"base_revision": i, "reason": "synthetic known correction", "upsert_lights": [light], "retire_lights": []}})
        write(replays / f"review_{i:03d}.json", demo_review(i))
    iterate(run, 2, replays)
