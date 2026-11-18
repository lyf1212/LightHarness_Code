"""Analytical Blender World-basis and camera tests, independent of reconstruction scores."""
from pathlib import Path
import sys
import copy
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from lightagent.config import load_config, default_config
from lightagent.io import ROOT, read, write, save_hdr
from lightagent.coordinates import camera_contract, rays, CV_TO_WORLD
from lightagent.state import initialize
from lightagent.pipeline import render


def main():
    out = ROOT / "validation/calibration"
    out.mkdir(parents=True, exist_ok=True)
    evidence = out / "evidence"
    (evidence / "environment").mkdir(parents=True, exist_ok=True)
    K = np.array([[85., 0, 46.2], [0, 90., 30.1], [0, 0, 1.]])
    camera = camera_contract(K, np.eye(4), (64, 96), 1.)
    write(evidence / "camera.json", camera)
    write(evidence / "object_priors.json", {})
    h, w = 256, 512
    v, u = np.mgrid[:h, :w]
    lon, lat = ((u+.5)/w-.5)*2*np.pi, (.5-(v+.5)/h)*np.pi
    d = np.stack([np.cos(lat)*np.cos(lon), -np.cos(lat)*np.sin(lon), np.sin(lat)], -1)
    save_hdr(evidence / "environment/environment.exr", (d+1)/2)
    code = out / "empty.py"
    code.write_text('def build_scene(ctx):\n    pass\n')
    config = copy.deepcopy(load_config(default_config()))
    config["render"].update(samples=1, device="CPU", threads=4)
    # Sharp pixel sampling makes the analytical check independent of reconstruction AA.
    results = []
    for i, theta in enumerate((0., .4)):
        state = initialize({"objects": [], "lights": [], "world": {"mode": "world", "strength": 1., "rotation_z": theta}})
        write(out / f"state_{i}.json", state)
        r = out / f"render_{i}"
        render(code, out / f"state_{i}.json", evidence, r, config)
        actual = np.load(r / "rgb_linear.npy")
        world = rays(K, (64, 96)) @ CV_TO_WORLD.T
        world /= np.linalg.norm(world, axis=-1, keepdims=True)
        rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1.]])
        expected = (world @ rot + 1)/2  # inverse rotation applied to column sampling directions
        mae = float(np.mean(np.abs(actual-expected)))
        results.append({"rotation_z": theta, "mae": mae,
                        "camera_error_pixels": read(r / "metadata.json")["camera_projection_error_pixels"]})
        if mae > .006:
            write(out / "result.json", {"passed": False, "results": results})
            raise AssertionError(f"World basis/rotation incorrect: MAE {mae}")
    write(out / "result.json", {"passed": True, "results": results})
    print(results)


if __name__ == "__main__":
    main()
