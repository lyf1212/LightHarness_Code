"""Two concurrent scenes with real Blender and explicitly synthetic model evidence."""
from copy import deepcopy
from pathlib import Path
import sys
from threading import Barrier
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightagent import batch, demo, evidence, pipeline
from lightagent.config import load_config, default_config
from lightagent.io import ROOT, inside, read, write


def main():
    root = inside(ROOT, sys.argv[1] if len(sys.argv) > 1 else ROOT / 'validation/batch_parallel_smoke')
    root.mkdir(parents=True, exist_ok=False)
    config = deepcopy(load_config(default_config()))
    config.update(max_side=160, rounds=2, max_repairs=0)
    config['render'].update(samples=16, threads=4, device='CPU')
    config['lumen']['mode'] = 'disabled'
    # These two inputs are placeholders; the demo renders its own reference.
    for name in ('a.png', 'b.png'):
        demo.preview(root / name, np.zeros((4, 4, 3)))
    write(root / 'images.json', {'images': ['a.png', 'b.png'], 'parallelism': 2, 'rounds': 2})
    barrier = Barrier(2)
    real_iterate = pipeline.iterate
    def synthetic_prepare(image, run, cfg, inventory=None):
        barrier.wait(timeout=10)
        demo.make_demo(run, cfg)
    def replay_iterate(run, rounds):
        return real_iterate(run, rounds, Path(run) / 'replay')
    with patch.object(demo, 'iterate', lambda *a: None), \
            patch.object(evidence, 'prepare', synthetic_prepare), \
            patch.object(pipeline, 'iterate', replay_iterate):
        assert batch.run_batch(root / 'images.json', root / 'runs', config)
        assert batch.run_batch(root / 'images.json', root / 'runs', config, resume=True)
    summary = read(root / 'runs/summary.json')
    for item in summary['images']:
        assert (Path(item['final']) / 'scene.blend').is_file()
        assert (Path(item['run']) / 'round_001/done.json').is_file()
    write(root / 'report.json', {'passed': True, 'scenes': 2, 'rounds_per_scene': 2,
                                'resume_passed': True, 'real_blender': True,
                                'evidence': 'synthetic fixtures', 'llm': 'replay',
                                'external_api_called': False, 'gpu_models_loaded': False})
    print(root / 'report.json')


if __name__ == '__main__':
    main()
