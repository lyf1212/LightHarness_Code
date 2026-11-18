"""Real Cycles regression: denoised display, raw closure, and standalone replay."""
from pathlib import Path
import os

import numpy as np
from PIL import Image
import pytest

from lightagent.demo import make_demo
from lightagent.io import ROOT, linear_to_srgb, read, run_process, sha, write
from lightagent.pipeline import export_final, render
from lightagent.config import default_config, load_config


def closure(folder):
    raw = np.load(folder / "rgb_linear.npy")
    total = sum(np.load(p) for p in folder.glob("light_*.npy"))
    return float(np.mean(abs(total - raw)) / max(float(np.mean(abs(raw))), 1e-6))


def test_cycles_quality_and_standalone_replay(tmp_path):
    config = load_config(os.environ.get("LIGHTPROXY_TEST_CONFIG", default_config()))
    if not Path(config["blender"]).is_file():
        pytest.skip("Configured Blender is unavailable")
    run = tmp_path / "quality_demo"
    make_demo(run, config)  # Explicit 16-sample test override must survive.
    truth = run / "truth_render"
    meta = read(truth / "metadata.json")
    assert meta["samples"] == 16
    assert meta["denoising"] and meta["view_layer_denoising"]
    assert meta["denoiser"] == "OPENIMAGEDENOISE"
    assert meta["use_adaptive_sampling"]
    assert meta["adaptive_threshold"] == pytest.approx(.01)
    assert meta["adaptive_min_samples"] == 0
    assert meta["rgb_linear_source"] == "Noisy Image"
    assert closure(truth) < .001
    raw = np.clip(linear_to_srgb(np.load(truth / "rgb_linear.npy")), 0, 1)
    shown = np.asarray(Image.open(truth / "rgb.png"), dtype=np.float32) / 255
    assert np.mean(abs(shown - raw)) > .001  # PNG is actually denoised, not raw.

    final = run / "final"
    selection = read(final / "selection.json")
    assert selection["best_round"] == 1
    source = run / "round_001"
    attempt = source / read(source / "accepted.json")["attempt"]
    assert sha(final / "render.png") == sha(attempt / "render/rgb.png")
    assert sha(final / "scores.json") == sha(attempt / "scores/scores.json")
    assert read(final / "scores.json")["render_settings"]["rgb_output"] == "denoised"

    # Later config edits cannot change settings exported for the already selected image.
    changed = read(run / "config.json")
    changed["render"].update(samples=7, use_denoising=False, use_adaptive_sampling=False)
    write(run / "config.json", changed)
    export_final(run, selection["round_losses"], selection["best_round"])
    settings = read(final / "settings.json")
    assert settings["samples"] == 16 and settings["use_denoising"]
    assert settings["use_adaptive_sampling"] and settings["adaptive_threshold"] == .01
    run_process([config["blender"], "-b", "-t", "4", "--python-exit-code", "1",
                 "--python", final / "scene.py"], final / "standalone.log", timeout=1200)
    replay_meta = read(final / "rerender/metadata.json")
    for key in ("samples", "denoising", "denoiser", "use_adaptive_sampling",
                "adaptive_threshold", "adaptive_min_samples", "rgb_linear_source"):
        assert replay_meta[key] == read(attempt / "render/metadata.json")[key]
    replay = np.asarray(Image.open(final / "rerender/rgb.png"), dtype=float)
    original = np.asarray(Image.open(final / "render.png"), dtype=float)
    assert np.max(abs(replay - original)) <= 1

    # Old saved configs lacking the new keys retain their unfiltered fixed sampling.
    legacy = read(run / "config.json")
    legacy["render"] = {"samples": 16, "device": "CPU", "threads": 4}
    render(run / "truth_geometry.py", run / "initial_state.json", run / "evidence",
           run / "legacy", legacy)
    legacy_meta = read(run / "legacy/metadata.json")
    assert not legacy_meta["denoising"] and not legacy_meta["use_adaptive_sampling"]
    assert legacy_meta["samples"] == 16 and legacy_meta["rgb_linear_source"] == "Image"
    assert closure(run / "legacy") < .001
