from copy import deepcopy
from pathlib import Path
from threading import Barrier

import pytest

from lightagent import batch, evidence, pipeline
from lightagent.io import read, write


@pytest.fixture
def setup_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, 'ROOT', tmp_path)
    config = {'gpu': '5', 'rounds': 2, 'sam3': {}, 'vggt': {}, 'marigold': {}, 'lumen': {'mode': 'disabled'}}
    for name in ('a.png', 'b.png'):
        (tmp_path / name).write_bytes(b'fake image')
    spec = tmp_path / 'images.json'
    write(spec, {'images': [{'id': 'a', 'image': 'a.png'}, {'id': 'b', 'image': 'b.png'}],
                 'model_gpus': {'vggt': [2, 3]}, 'parallelism': 2})
    return config, spec, tmp_path / 'runs'


def test_concurrent_batch_isolates_failure_and_resumes(setup_batch, monkeypatch):
    config, spec, root = setup_batch
    barrier = Barrier(2)
    prepared, iterated = [], []
    def prepare(image, run, cfg, inventory=None):
        prepared.append(run)
        write(run / 'config.json', cfg)
        write(run / 'evidence/manifest.json', {})
        write(run / 'initial_state.json', {})
        barrier.wait(timeout=5)  # Sequential execution cannot pass this assertion.
    def iterate(run, rounds):
        iterated.append(run)
        if run.parent.name == 'a':
            raise RuntimeError('one scene failed')
    monkeypatch.setattr(evidence, 'prepare', prepare)
    monkeypatch.setattr(pipeline, 'iterate', iterate)
    assert not batch.run_batch(spec, root, config)
    summary = read(root / 'summary.json')
    assert [r['status'] for r in summary['images']] == ['failed', 'completed']
    assert len(prepared) == 2 and len(iterated) == 2
    assert 'gpu_pool' not in config['vggt']  # Caller config is not mutated.
    monkeypatch.setattr(pipeline, 'iterate', lambda run, rounds: iterated.append(run))
    assert batch.run_batch(spec, root, config, resume=True, rounds=3)
    assert len(prepared) == 2 and len(iterated) == 4
    assert read(root / 'summary.json')['rounds'] == 3
    with pytest.raises(FileExistsError):
        batch.run_batch(spec, root, config)


def test_incomplete_preparation_gets_new_attempt(setup_batch, monkeypatch):
    config, spec, root = setup_batch
    write(spec, {'images': ['a.png']})
    def failed(image, run, cfg, inventory=None):
        write(run / 'partial.json', {'keep': True})
        raise RuntimeError('disconnected')
    monkeypatch.setattr(evidence, 'prepare', failed)
    assert not batch.run_batch(spec, root, config)
    def success(image, run, cfg, inventory=None):
        write(run / 'config.json', cfg)
        write(run / 'evidence/manifest.json', {})
        write(run / 'initial_state.json', {})
    monkeypatch.setattr(evidence, 'prepare', success)
    monkeypatch.setattr(pipeline, 'iterate', lambda *args: None)
    assert batch.run_batch(spec, root, config, resume=True)
    assert (root / 'image_000/attempt_000/partial.json').exists()
    assert read(root / 'summary.json')['images'][0]['run'].endswith('attempt_001')


def test_validation_before_starting_any_work(setup_batch):
    config, spec, root = setup_batch
    for images in ([], ['missing.png'], [{'id': '../escape', 'image': 'a.png'}],
                   [{'id': 'same', 'image': 'a.png'}, {'id': 'same', 'image': 'b.png'}]):
        write(spec, {'images': images})
        with pytest.raises((ValueError, FileNotFoundError)):
            batch.run_batch(spec, root, config)
        assert not root.exists()


def test_resume_rejects_changed_image(setup_batch, monkeypatch):
    config, spec, root = setup_batch
    monkeypatch.setattr(batch, 'run_scene', lambda item, *args: {'id': item['id'], 'status': 'completed'})
    assert batch.run_batch(spec, root, config)
    (spec.parent / 'a.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='unchanged images'):
        batch.run_batch(spec, root, config, resume=True)
