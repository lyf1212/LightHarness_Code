import json
import numpy as np
import pytest

from lightagent.coordinates import camera_contract, normals_from_z
from lightagent.io import write, read
from lightagent.surface_geometry import (
    CONTRACT, build_surface_geometry, compare_surface_geometry, fit_plane,
    plumb_normal, points_world, robust_normal,
)


def test_ransac_rejects_outliers_and_degenerate_lines():
    rng = np.random.default_rng(3)
    xy = rng.uniform(-1, 1, (1000, 2))
    z = .2*xy[:, 0]-.3*xy[:, 1]+2
    cloud = np.column_stack((xy, z))
    cloud = np.concatenate((cloud, rng.uniform(-3, 3, (200, 3))))
    fit = fit_plane(cloud)
    expected = np.array([-.2, .3, 1.]); expected /= np.linalg.norm(expected)
    assert abs(np.dot(fit['normal'], expected)) > .999
    assert .8 < fit['inlier_frac'] < .9
    assert fit_plane(np.column_stack((np.arange(50), np.zeros((50, 2))))) is None
    assert fit_plane(np.full((10, 3), np.nan)) is None


def test_robust_average_and_plumb_use_estimated_up_not_world_z():
    n = robust_normal(np.concatenate((np.tile([0, 0, 1.], (80, 1)), np.tile([1, 0, 0.], (20, 1)))))
    assert np.allclose(n, [0, 0, 1])
    up = np.array([0, .6, .8])
    support, kind = plumb_normal([.1, .6, .8], up)
    assert kind == 'support' and np.allclose(support, up)
    wall, kind = plumb_normal([1, .01, .02], up)
    assert kind == 'wall' and abs(wall @ up) < 1e-10


@pytest.fixture
def plane_evidence(tmp_path):
    e = tmp_path/'evidence'
    (e/'geometry').mkdir(parents=True)
    (e/'masks').mkdir()
    h, w = 60, 80
    depth = np.full((h, w), 3., np.float32)
    K = np.array([[70., 0, 39.5], [0, 70., 29.5], [0, 0, 1.]])
    camera = camera_contract(K, np.eye(4), depth.shape, 1.)
    # Nonzero translation catches an easy camera/world mismatch in plane points.
    camera['matrix_world'][0][3] = 2.
    write(e/'camera.json', camera)
    np.save(e/'geometry/depth.npy', depth)
    np.save(e/'geometry/valid.npy', np.ones(depth.shape, bool))
    normals, _ = normals_from_z(depth, K)
    np.save(e/'geometry/normals_cv.npy', normals)
    masks = {'wall': np.ones(depth.shape, bool), 'tiny': np.zeros(depth.shape, bool),
             'chair': np.ones(depth.shape, bool)}
    masks['tiny'][20:23, 30:40] = True
    index = {}
    for oid, mask in masks.items():
        np.save(e/f'masks/{oid}.npy', mask)
        index[oid] = {'available': True, 'path': f'masks/{oid}.npy'}
    write(e/'masks/index.json', index)
    write(e/'inventory.json', {'objects': [{'id': 'wall', 'role': 'wall'},
          {'id': 'tiny', 'role': 'wall'}, {'id': 'chair', 'role': 'object'},
          {'id': 'missing', 'role': 'floor'}]})
    return e, camera, depth, masks


def test_surface_contract_hides_unreliable_planes_and_handles_translation(plane_evidence):
    e, camera, depth, _ = plane_evidence
    evidence = build_surface_geometry(e)
    assert evidence['contract'] == CONTRACT
    assert not evidence['gravity']['available']
    wall = evidence['surfaces']['wall']
    assert wall['reliable']
    assert wall['plane']['point_world'] == pytest.approx([2., 3., 0.], abs=1e-6)
    assert wall['plane']['normal_world'] == pytest.approx([0., -1., 0.], abs=1e-6)
    for oid in ['tiny', 'chair', 'missing']:
        row = evidence['surfaces'][oid]
        assert not row['reliable'] and row['plane'] is None and row['reason']
    json.dumps(evidence, allow_nan=False)
    assert points_world(depth, camera).shape == (60, 80, 3)


def test_tilted_floor_produces_consistent_world_up(plane_evidence):
    e, camera, depth, _ = plane_evidence
    from lightagent.coordinates import rays
    # Camera views a sloped support: plane n.p=2 in camera coordinates.
    n = np.array([0., .8, .6])
    depth = (2/(rays(np.array(camera['K']), depth.shape) @ n)).astype(np.float32)
    np.save(e/'geometry/depth.npy', depth)
    write(e/'inventory.json', {'objects': [{'id': 'wall', 'role': 'floor'}]})
    evidence = build_surface_geometry(e)
    expected = np.array([0., -.6, .8])
    assert evidence['gravity']['available']
    assert evidence['gravity']['up_world'] == pytest.approx(expected, abs=1e-5)
    assert evidence['surfaces']['wall']['plane']['normal_world'] == pytest.approx(expected, abs=1e-5)
    # The input camera was not secretly rotated or translated.
    assert read(e/'camera.json') == camera


def test_missing_and_rotated_surfaces_cannot_score_as_perfect(plane_evidence):
    e, camera, depth, masks = plane_evidence
    reference = build_surface_geometry(e)
    valid = np.ones(depth.shape, bool)
    same = compare_surface_geometry(reference, depth, valid, camera, masks)
    assert same['value'] == pytest.approx(0, abs=1e-5)
    assert same['selection_component'] == pytest.approx(0, abs=1e-7)
    absent = compare_surface_geometry(reference, depth, valid, camera, {})
    assert absent['expected_surfaces'] == 1 and absent['matched_surfaces'] == 0
    assert absent['selection_component'] == 1
    assert not absent['available'] and absent['value'] is None
    from lightagent.coordinates import rays
    r = rays(np.array(camera['K']), depth.shape)
    tilted = 3/(r @ np.array([.5, 0, np.sqrt(.75)]))
    changed = compare_surface_geometry(reference, tilted, valid, camera, masks)
    assert changed['value'] == pytest.approx(30, abs=1e-5)
    assert changed['selection_component'] == pytest.approx(1/3, abs=1e-6)


def test_request_images_do_not_include_dense_normals(tmp_path):
    from PIL import Image
    from lightagent.pipeline import evidence_images
    from lightagent.llm import Client
    from lightagent.prompts import BUILD, REVIEW
    e = tmp_path/'evidence'
    for name in ['reference.png', 'overview.png', 'intrinsics/albedo.png',
                 'intrinsics/shading.png', 'geometry/depth.png', 'geometry/normals.png']:
        p = e/name
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (4, 4)).save(p)
    for role, prompt in [('build', BUILD), ('review', REVIEW)]:
        path = tmp_path/role
        Client({'api': 'responses', 'model': 'test'}).ask(path, prompt,
            {'surface_geometry': {'contract': CONTRACT}}, evidence_images(e), replay={'ok': True})
        assert all('normals.png' not in row['path'] for row in read(path/'images.json'))
        assert read(path/'context.json')['surface_geometry']['contract'] == CONTRACT
        assert 'world +Z is image-up, NOT automatically gravity' in (path/'system.txt').read_text()


def test_legacy_run_is_rejected_before_model_call(tmp_path, monkeypatch):
    from lightagent.pipeline import _iterate_locked
    from lightagent.llm import Client
    e = tmp_path/'evidence'; e.mkdir()
    write(e/'manifest.json', {'schema_version': 1, 'files': {}})
    def forbidden(*args, **kwargs):
        raise AssertionError('No LLM request is permitted for incompatible evidence')
    monkeypatch.setattr(Client, 'ask', forbidden)
    with pytest.raises(ValueError, match='fork_surface_run'):
        _iterate_locked(tmp_path, {}, 1, None)
