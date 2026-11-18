"""Regression for mixed CUDA/OPTIX entries returned by real Cycles preferences."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def configure():
    # Exercise the real host helper without requiring bpy in the host interpreter.
    path = Path(__file__).resolve().parents[1] / 'lightagent/blender_host.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'configure_cycles_gpu')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['configure_cycles_gpu']


def devices(kinds):
    return SimpleNamespace(devices=[SimpleNamespace(type=k, name='same physical GPU', id=f'{k}_gpu', use=True) for k in kinds],
                           get_devices=lambda: None)


def test_mixed_backends_select_only_requested_representation():
    prefs = devices(['CUDA', 'CPU', 'OPTIX'])
    selected = configure()(prefs, 'OPTIX')
    assert [d.id for d in prefs.devices if d.use] == ['OPTIX_gpu']
    assert selected == [{'id': 'OPTIX_gpu', 'name': 'same physical GPU', 'type': 'OPTIX'}]


def test_other_available_backend_does_not_mask_missing_requested_backend():
    prefs = devices(['CUDA', 'CPU'])
    with pytest.raises(RuntimeError, match='OPTIX'):
        configure()(prefs, 'OPTIX')
    assert not any(d.use for d in prefs.devices)
