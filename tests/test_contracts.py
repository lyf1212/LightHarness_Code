import copy
import json
from pathlib import Path

import numpy as np
import pytest

from lightagent.codecheck import validate_code
from lightagent.coordinates import resize_affine, range_to_z, z_to_range, normals_from_z, canonicalize_environment
from lightagent.demo import inventory, CODE
from lightagent.io import read_exr, save_hdr
from lightagent.llm import Client, load_env
from lightagent.lumen_worker import project_condition
from lightagent.metrics import positive_scale
from lightagent.state import initialize, apply_patch


def test_pixel_centers_and_intrinsics_roundtrip():
    A = resize_affine(811, 533, 518, 336, 0, 91)
    K = np.array([[620., 0, 398.3], [0, 618., 263.9], [0, 0, 1.]])
    ray = np.array([.2, -.13, 1.])
    assert np.allclose(np.linalg.inv(A) @ (A @ K @ ray), K @ ray)
    assert A[0, 2] == pytest.approx((518/811-1)/2)


def test_depth_range_is_not_z():
    K = np.array([[80., 0, 40.], [0, 90., 25.], [0, 0, 1.]])
    z = np.full((60, 100), 3.)
    distance = z_to_range(z, K)
    assert distance[0, 0] > 3
    assert np.allclose(range_to_z(distance, K), z)
    normals, valid = normals_from_z(z, K)
    assert np.allclose(normals[valid], [0, 0, -1])


def test_environment_basis_rotation():
    h, w = 80, 160
    v, u = np.mgrid[:h, :w]
    lon = ((u+.5)/w-.5)*2*np.pi
    lat = (.5-(v+.5)/h)*np.pi
    # Encode source direction as a color. Source center is world +Y, increasing u = -X.
    d = np.stack([-np.cos(lat)*np.sin(lon), np.cos(lat)*np.cos(lon), np.sin(lat)], -1)
    hdr = ((d+1)/2).astype(np.float32)
    result = canonicalize_environment(hdr, [0, 1, 0], [0, 0, 1])
    expected = (np.stack([np.cos(lat)*np.cos(lon), -np.cos(lat)*np.sin(lon), np.sin(lat)], -1)+1)/2
    assert np.max(np.abs(result-expected)) < .025
    assert np.allclose(canonicalize_environment(result, [1, 0, 0], [0, 0, 1], -1), result, atol=.001)
    with pytest.raises(ValueError):
        canonicalize_environment(hdr, [0, 0, 0], [0, 0, 1])


def test_calibrated_lumen_projection_uses_principal_point():
    from PIL import Image
    image = np.zeros((60, 100, 3), np.uint8)
    image[20:40, 60:80] = 255
    K = np.array([[90., 0, 70.], [0, 90., 30.], [0, 0, 1.]])
    panorama = np.asarray(project_condition(Image.fromarray(image), K, 128, 256))
    assert panorama[64, 128].mean() > 240
    assert panorama[64, 0].sum() == 0
    image[:, :40] = 50
    image[:, 60:] = 200
    K[0, 2] = 49.5
    panorama = np.asarray(project_condition(Image.fromarray(image), K, 128, 256))
    assert panorama[64, 115].mean() < panorama[64, 140].mean()


def test_transaction_state_history_and_retirement():
    state = initialize(inventory())
    original = copy.deepcopy(state)
    with pytest.raises(ValueError):
        apply_patch(state, {"base_revision": 4, "reason": "stale"}, 0)
    assert state == original
    updated = apply_patch(state, {"base_revision": 0, "reason": "source appears off", "retire_lights": ["light_key"]}, 0)
    assert updated["lights"]["light_key"]["status"] == "retired"
    assert "light_key" in updated["lights"] and state == original
    assert updated["history"][-1]["patch"]["reason"] == "source appears off"


def test_rejects_invalid_lighting_without_partial_commit():
    state = initialize(inventory())
    light = copy.deepcopy(state["lights"]["light_key"])
    light["elements"][0]["energy"] = -1
    with pytest.raises(ValueError):
        apply_patch(state, {"base_revision": 0, "reason": "bad", "upsert_lights": [light]}, 0)
    assert state["lights"]["light_key"]["elements"][0]["energy"] == 250


def test_exr_preserves_hdr_negative_and_channel_order(tmp_path):
    x = np.array([[[100., .2, -.6], [1., 2., 3.]]], np.float32)
    save_hdr(tmp_path / "test.exr", x)
    assert np.array_equal(read_exr(tmp_path / "test.exr"), x)


def test_shading_scale_shared_across_channels():
    pred = np.ones((4, 5, 3), np.float32)
    ref = pred * 3
    valid = np.ones((4, 5), bool)
    assert positive_scale(ref, pred, valid) == pytest.approx(3)
    assert positive_scale(ref, pred*0, valid) is None
    colored = pred*np.array([1, 2, 3])
    scale = positive_scale(colored, pred, valid)
    assert np.mean(abs(colored-pred*scale)) > .5


def test_code_contract():
    validate_code(CODE)
    with pytest.raises(ValueError):
        validate_code("import os\ndef build_scene(ctx):\n    os.system('pwd')")


def test_builder_context_cannot_mutate_authoritative_state(monkeypatch):
    import importlib.util
    import sys
    import types
    from lightagent.io import ROOT
    monkeypatch.setitem(sys.modules, 'bpy', types.ModuleType('bpy'))
    fake_math = types.ModuleType('mathutils')
    fake_math.Matrix = fake_math.Vector = None
    monkeypatch.setitem(sys.modules, 'mathutils', fake_math)
    spec = importlib.util.spec_from_file_location('host_contract_test', ROOT/'lightagent/blender_host.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state, camera = initialize(inventory()), {'K': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    ctx = module.Context(state, camera, {})
    ctx.state['lights']['light_key']['elements'][0]['energy'] = 999
    ctx.camera['K'][0][0] = 10
    assert state['lights']['light_key']['elements'][0]['energy'] == 250
    assert camera['K'][0][0] == 1


def test_complete_request_archive_and_no_credential(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-not-for-archive")
    im = tmp_path / "ref.png"
    Image.new("RGB", (3, 2)).save(im)
    dest = tmp_path / "call"
    response = Client({"api": "responses", "model": "test"}).ask(dest, "test", {"foo": 3}, [("ref", im)], replay={"ok": True})
    assert response == {"ok": True}
    payload = json.loads((dest / "request.json").read_text())
    assert payload["input"][0]["content"][-1]["image_url"].startswith("data:image/png;base64,")
    assert "test-secret" not in "".join(p.read_text() for p in dest.glob("*.json"))


def test_dotenv_does_not_execute_shell(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text('LIGHTAGENT_TEST_VALUE="$(touch /tmp/should-not-exist-lightagent)"\n')
    monkeypatch.delenv("LIGHTAGENT_TEST_VALUE", raising=False)
    load_env(p)
    import os
    assert os.environ["LIGHTAGENT_TEST_VALUE"].startswith("$(touch")


def test_blender_demo_when_present():
    import os
    from lightagent.io import ROOT, read
    run = ROOT / os.environ.get("LIGHTPROXY_DEMO_RUN", "runs/demo")
    if not (run / "final/selection.json").exists():
        pytest.skip("Run demo first")
    selection = read(run / "final/selection.json")
    assert selection["best_round"] == 1
    assert selection["round_losses"][1]["loss"] < selection["round_losses"][0]["loss"]
    score = read(run / "final/scores.json")["metrics"]
    assert score["rgb_srgb_mae"]["value"] < .001
    assert score["light_contributions"]["additive_closure_nmae"] < .001
    z = np.load(run / "truth_render/depth.npy")
    # Center ray intersects the cube's y=3.25 front face.
    assert z[58, 76] == pytest.approx(3.25, abs=.001)
    wall = np.load(run / "truth_render/masks/obj_wall.npy")
    assert np.max(np.abs(z[wall]-7.9)) < 1e-4
