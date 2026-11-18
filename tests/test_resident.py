"""Exercise real sockets/processes with lightweight models; no GPU allocation."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from lightagent import resident, workers


@pytest.fixture
def service(tmp_path):
    # Stand in for a model-specific Python environment. The actual daemon, client,
    # cache, logs, process isolation and serialization are all exercised unchanged.
    python = tmp_path / "model-python"
    python.write_text(f"#!{sys.executable}\n" + '''
import os
import sys
import time
from pathlib import Path
from lightagent import resident, workers
from lightagent.io import write
loads = 0
calls = 0
def execute(kind, job):
    global loads, calls
    def load():
        global loads
        loads += 1
        return object()
    model = workers.cached_model(kind, job["options"], load)
    calls += 1
    print("job", calls, flush=True)
    if job.get("fail"):
        raise ValueError("intentional inference failure")
    time.sleep(job.get("sleep", 0))
    write(Path(job["output"]) / "result.json", {"pid": os.getpid(), "loads": loads,
          "calls": calls, "model": id(model), "seed": job["seed"]})
resident.execute = execute
resident.serve(Path(sys.argv[sys.argv.index("--directory") + 1]),
               sys.argv[sys.argv.index("--kind") + 1])
''')
    python.chmod(0o700)
    job = {"output": str(tmp_path), "seed": 42, "options": {
        "checkpoint": str(tmp_path / "weights"), "device": "cuda"}}
    directory = resident.service_dir("sam3", job, str(python), "5")
    yield job, str(python), directory
    try:
        with resident.request_lock(directory, time.monotonic() + 5):
            with resident.connect(directory, 5) as conn, conn.makefile("rwb") as stream:
                resident.receive(stream)
                resident.send(stream, {"command": "stop"})
                resident.receive(stream)
    except (OSError, RuntimeError):
        pass


def call(service, job=None, timeout=10):
    original, python, _ = service
    job = original if job is None else job
    resident.run_resident("sam3", job, Path(job["output"]) / "worker.log",
                          python=python, gpu="5", timeout=timeout)
    return json.loads((Path(job["output"]) / "result.json").read_text())


def test_reuse_across_independent_clients(service):
    job, python, directory = service
    code = '''import json, sys
from pathlib import Path
from lightagent.resident import run_resident
job = json.loads(sys.argv[1])
run_resident("sam3", job, Path(job["output"]) / "worker.log", python=sys.argv[2], gpu="5", timeout=10)
'''
    results = []
    for seed in (42, 123):
        job["seed"] = seed
        subprocess.run([sys.executable, "-c", code, json.dumps(job), python],
                       cwd=resident.ROOT, check=True, timeout=20)
        results.append(json.loads((Path(job["output"]) / "result.json").read_text()))
    assert results[0]["pid"] == results[1]["pid"]
    assert results[0]["model"] == results[1]["model"]
    assert results[1]["loads"] == 1
    assert results[1]["calls"] == 2
    assert results[1]["seed"] == 123
    assert "job 2" in (Path(job["output"]) / "worker.log").read_text()
    assert directory.joinpath("service.json").exists()


def test_error_does_not_discard_loaded_model(service):
    first = call(service)
    with pytest.raises(RuntimeError, match="intentional inference failure"):
        call(service, dict(service[0], fail=True))
    log = Path(service[0]["output"]) / "worker.log"
    assert "Traceback" in log.read_text()
    assert json.loads(log.with_suffix(".process.json").read_text())["returncode"] == 1
    after = call(service)
    assert after["pid"] == first["pid"]
    assert after["loads"] == 1
    assert after["calls"] == 3


def test_dead_service_restarts(service):
    first = call(service)
    os.kill(first["pid"], signal.SIGKILL)
    time.sleep(.1)
    second = call(service)
    assert first["pid"] != second["pid"]
    assert second["loads"] == 1


def test_timeout_stops_inference_and_next_call_restarts(service):
    first = call(service)
    with pytest.raises(TimeoutError, match="inference timed out"):
        call(service, dict(service[0], sleep=10), timeout=.2)
    time.sleep(.1)
    second = call(service)
    assert second["pid"] != first["pid"]
    assert second["calls"] == 1


def test_identity_ignores_inference_options_but_separates_models_and_gpus(tmp_path, monkeypatch):
    job = {"options": {"checkpoint": "weights", "device": "cuda", "resolution": 512}}
    first = resident.service_dir("marigold", job, sys.executable, "5")
    job["options"]["resolution"] = 768
    assert resident.service_dir("marigold", job, sys.executable, "5") == first
    assert resident.service_dir("marigold", job, sys.executable, "4") != first
    job["options"]["checkpoint"] = "other"
    assert resident.service_dir("marigold", job, sys.executable, "5") != first
    job["options"]["checkpoint"] = "weights"
    monkeypatch.setattr(resident, "ROOT", tmp_path)
    assert resident.service_dir("marigold", job, sys.executable, "5") != first


def test_model_cache_retries_failed_load_and_ignores_threshold(monkeypatch):
    monkeypatch.setattr(workers, "_MODELS", {})
    opt = {"checkpoint": "weights", "device": "cuda", "threshold": .3}
    with pytest.raises(ValueError):
        workers.cached_model("sam3", opt, lambda: (_ for _ in ()).throw(ValueError()))
    model = workers.cached_model("sam3", opt, object)
    opt["threshold"] = .6
    assert workers.cached_model("sam3", opt, lambda: pytest.fail("reloaded model")) is model
    opt["checkpoint"] = "other"
    assert workers.cached_model("sam3", opt, object) is not model


def test_lumen_keeps_pipelines_but_refreshes_camera(tmp_path, monkeypatch):
    import contextlib
    from types import SimpleNamespace
    import numpy as np
    from PIL import Image
    from lightagent import lumen_worker
    from lightagent.io import write, read

    # Exercise the actual LUMEN loader and inference adapter using fake upstream
    # components, so a second job must reuse FLUX, LoRA and Redux allocations.
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "inference_single.py").write_text('''
from pathlib import Path
from types import SimpleNamespace
import numpy as np

def record(name):
    with (Path(__file__).parent / "loads.txt").open("a") as f:
        f.write(name + "\\n")

class FluxTransformer2DModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        record("transformer")
        return cls()
    def to(self, *args, **kwargs):
        return self

class CustomFluxTransformer2DModel:
    def __init__(self, base):
        self.base = base
    def add_adapter(self, *args, **kwargs):
        record("adapter")
    @staticmethod
    def lora_state_dict(path, **kwargs):
        assert kwargs["local_files_only"]
        record("lora")
        return {"transformer.test.lora_A.weight": object()}

class DiffusionPipeline:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        record("pipeline")
        pipe = cls()
        pipe.transformer = kwargs["transformer"]
        return pipe
    def to(self, device):
        self.device = device
        return self
    def __call__(self, **kwargs):
        assert self.device == "cuda"
        return SimpleNamespace(images=[np.full((kwargs["height"], kwargs["width"], 3), .25, np.float32)])

class FluxPriorReduxPipeline(DiffusionPipeline):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        record("redux")
        return cls()
    def __call__(self, images):
        assert self.device == "cuda"
        return {"prompt_embeds": object(), "pooled_prompt_embeds": object()}

LoraConfig = lambda **kwargs: kwargs
set_peft_model_state_dict = lambda *args, **kwargs: None
''')
    class Generator:
        def __init__(self, device):
            self.device = device
        def manual_seed(self, seed):
            return self
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        bfloat16="bfloat16", manual_seed=lambda seed: None, Generator=Generator,
        inference_mode=contextlib.nullcontext, __version__="fake", version=SimpleNamespace(cuda="fake")))
    lumen_worker.load_pipelines.cache_clear()
    image_path = tmp_path / "reference.png"
    Image.new("RGB", (20, 10), "white").save(image_path)
    camera = tmp_path / "camera.json"
    lora = tmp_path / "lora.safetensors"
    lora.touch()
    job = {"image": str(image_path), "output": str(tmp_path), "camera": str(camera), "seed": 42,
           "options": {"repo": str(repo), "flux_checkpoint": "flux", "lora_checkpoint": str(lora),
                       "redux_checkpoint": "redux", "height": 8, "width": 16}}
    try:
        for focal in (10, 20):
            K = [[focal, 0, 10], [0, focal, 5], [0, 0, 1]]
            write(camera, {"K": K})
            lumen_worker.main(job)
            assert read(tmp_path / "projection.json")["K"] == K
        assert (repo / "loads.txt").read_text().splitlines() == [
            "transformer", "pipeline", "adapter", "lora", "redux"]
        assert lumen_worker.load_pipelines.cache_info().hits == 1
        assert (tmp_path / "reference.exr").exists()
    finally:
        lumen_worker.load_pipelines.cache_clear()


def test_simultaneous_clients_share_one_service(service, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    original, python, _ = service
    barrier = Barrier(2)
    def run(index):
        output = tmp_path / f"parallel_{index}"
        output.mkdir()
        job = dict(original, output=str(output), sleep=.1)
        barrier.wait(timeout=5)
        resident.run_resident("sam3", job, output / "worker.log", python=python, gpu="5", timeout=10)
        return json.loads((output / "result.json").read_text())
    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = list(executor.map(run, (0, 1)))
    assert first["pid"] == second["pid"]
    assert first["model"] == second["model"]
    assert first["loads"] == second["loads"] == 1
    assert {first["calls"], second["calls"]} == {1, 2}


def test_concurrent_images_share_model_and_tcp_port(service):
    from concurrent.futures import ThreadPoolExecutor
    job, _, directory = service
    jobs = [dict(job, output=str(Path(job['output']) / str(i)), sleep=.1) for i in range(3)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda j: call(service, j), jobs))
    assert len({r['pid'] for r in results}) == 1
    assert all(r['loads'] == 1 for r in results)
    assert sorted(r['calls'] for r in results) == [1, 2, 3]
    info = json.loads((directory / 'service.json').read_text())
    assert info['host'] == '127.0.0.1' and 0 < info['port'] < 65536
    call(service)
    assert json.loads((directory / 'service.json').read_text())['port'] == info['port']


def test_dropped_connection_replays_receipt_without_repeating_inference(service, monkeypatch):
    original = resident.receive
    reads = 0
    def disconnect(stream):
        nonlocal reads
        reads += 1
        if reads == 2:
            stream.close()
            raise ConnectionResetError('simulated lost TCP connection')
        return original(stream)
    monkeypatch.setattr(resident, 'receive', disconnect)
    result = call(service, dict(service[0], sleep=.5))
    assert result['calls'] == 1
    assert result['loads'] == 1
    record = json.loads((Path(service[0]['output']) / 'worker.process.json').read_text())
    assert record['reconnects'] == 1
    assert call(service)['calls'] == 2


def test_pool_uses_an_idle_second_gpu_and_keeps_both_resident(service):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    job, python, _ = service
    barrier = Barrier(2)
    def invoke(i):
        out = Path(job['output']) / f'pool_{i}'
        barrier.wait(timeout=5)
        resident.run_resident('sam3', dict(job, output=str(out), sleep=.5), out / 'worker.log',
                              python=python, gpu=['5', '4'], timeout=10)
        return json.loads((out / 'worker.process.json').read_text())
    second = resident.service_dir('sam3', job, python, '4')
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(invoke, range(2)))
        assert {r['gpu'] for r in results} == {'4', '5'}
        assert len({r['pid'] for r in results}) == 2
        assert second.joinpath('service.json').exists()
    finally:
        if second.joinpath('service.json').exists():
            with resident.connect(second, 5) as conn, conn.makefile('rwb') as stream:
                resident.receive(stream)
                resident.send(stream, {'command': 'stop'})
                resident.receive(stream)


def test_gpu_targets():
    assert resident.gpu_targets({'gpu': '5', 'sam3': {}}, 'sam3') == ['5']
    assert resident.gpu_targets({'gpu': '5', 'sam3': {'gpu': 4}}, 'sam3') == ['4']
    assert resident.gpu_targets({'sam3': {'gpu_pool': [2, 3]}}, 'sam3') == ['2', '3']
    for targets in ([], ['2', '2'], ['2,3']):
        with pytest.raises(ValueError):
            resident.gpu_targets({'sam3': {'gpu_pool': targets}}, 'sam3')
