"""Local, serialized model services that survive the calling CLI process."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import uuid
import socket
import subprocess
import sys
import tempfile
import time
import traceback

from .io import ROOT, read, write

KINDS = ("sam3", "vggt", "marigold", "lumen")
MODEL_OPTIONS = ("repo", "checkpoint", "device", "flux_checkpoint", "lora_checkpoint", "redux_checkpoint")


def service_dir(kind, job, python, gpu):
    identity = {"root": str(ROOT), "kind": kind, "python": str(python), "gpu": str(gpu),
                "models": {k: job["options"].get(k) for k in MODEL_OPTIONS},
                "vggt_repo": job.get("vggt_repo"), "version": 2}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    base = Path(tempfile.gettempdir()) / f"lightagent-workers-{os.getuid()}"
    base.mkdir(mode=0o700, exist_ok=True)
    directory = base / key
    directory.mkdir(mode=0o700, exist_ok=True)
    return directory


@contextlib.contextmanager
def request_lock(directory, deadline):
    with (directory / "request.lock").open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for the resident worker's current job")
                time.sleep(.05)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def connect(directory, timeout):
    # Metadata and its authentication token live in a user-private directory.
    info = read(directory / "service.json")
    legacy = "port" not in info
    conn = socket.socket(socket.AF_UNIX if legacy else socket.AF_INET, socket.SOCK_STREAM)
    conn.settimeout(max(.001, timeout))
    try:
        if legacy:
            conn.connect(str(directory / "worker.sock"))
        else:
            conn.connect(("127.0.0.1", info["port"]))
            conn.sendall((json.dumps({"token": info["token"]}) + "\n").encode())
        return conn
    except BaseException:
        conn.close()
        raise


def gpu_targets(config, kind):
    opt = config[kind]
    targets = opt.get("gpu_pool", [opt.get("gpu", config.get("gpu", "0"))])
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"{kind}.gpu_pool must be a nonempty list of GPU IDs")
    if any((not isinstance(g, (str, int)) or isinstance(g, bool)) or not str(g).strip() or "," in str(g) for g in targets):
        raise ValueError(f"{kind}: each GPU target must name one device; gpu_pool creates replicas")
    targets = [str(g) for g in targets]
    if len(set(targets)) != len(targets):
        raise ValueError(f"{kind}.gpu_pool contains duplicate devices")
    return targets


@contextlib.contextmanager
def pool_slot(kind, job, python, targets, deadline):
    # flock also works across independent batch/CLI processes. Prefer an already
    # running idle replica, then start another configured replica if all are busy.
    directories = [(gpu, service_dir(kind, job, python, gpu)) for gpu in targets]
    while True:
        ordered = sorted(directories, key=lambda item: not (item[1] / "service.json").exists())
        for gpu, directory in ordered:
            lock = (directory / "request.lock").open("a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            try:
                yield gpu, directory
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for a free {kind} replica")
        time.sleep(.05)


def run_model(kind, job, log, config):
    return run_resident(kind, job, log, python=config[kind]["python"],
                        gpu=gpu_targets(config, kind),
                        timeout=config.get("worker_timeout", 1800),
                        queue_timeout=config.get("worker_queue_timeout", 86400),
                        reconnect_attempts=config.get("worker_reconnect_attempts", 3))


def ensure_service(directory, kind, python, gpu, deadline):
    try:
        return connect(directory, deadline - time.monotonic())
    except (FileNotFoundError, ConnectionRefusedError):
        pass
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               OPENCV_IO_ENABLE_OPENEXR="1")
    # Prevent an inherited PYTHONPATH from selecting the other experiment's package.
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    with (directory / "service.log").open("a") as log:
        child = subprocess.Popen([str(python), "-u", "-m", "lightagent.resident", "serve",
                                  "--kind", kind, "--directory", str(directory)],
                                 cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"Resident worker failed to start; see {directory / 'service.log'}")
        try:
            return connect(directory, deadline - time.monotonic())
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(.05)
    child.terminate()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
    raise TimeoutError(f"Resident worker startup timed out; see {directory / 'service.log'}")


def send(stream, message):
    stream.write(json.dumps(message).encode() + b"\n")
    stream.flush()


def receive(stream):
    line = stream.readline()
    if not line:
        raise ConnectionError("Resident worker disconnected; inspect the job and service logs")
    return json.loads(line)


def run_resident(kind, job, log, *, python, gpu, timeout=1800, reconnect_attempts=3, queue_timeout=None):
    if kind not in KINDS:
        raise ValueError(f"Unknown model: {kind}")
    if timeout <= 0 or not isinstance(reconnect_attempts, int) or reconnect_attempts < 0:
        raise ValueError("worker timeout must be positive and reconnect attempts nonnegative")
    job = dict(job)
    for key in ("image", "output", "inventory", "camera"):
        if key in job:
            path = Path(job[key])
            job[key] = str(path if path.is_absolute() else ROOT / path)
    log = Path(log).resolve()
    log.parent.mkdir(parents=True, exist_ok=True)
    targets = gpu_targets({kind: {"gpu_pool": gpu if isinstance(gpu, list) else [gpu]}}, kind)
    queue_timeout = timeout if queue_timeout is None else queue_timeout
    if queue_timeout <= 0:
        raise ValueError("worker queue timeout must be positive")
    started, deadline = time.time(), time.monotonic() + queue_timeout
    request_id = uuid.uuid4().hex
    receipt = log.with_suffix(".receipt.json")
    pid, error, directory, retries = None, None, None, 0
    try:
        with pool_slot(kind, job, python, targets, deadline) as (gpu, directory):
            deadline = time.monotonic() + timeout
            write(log.with_suffix(".command.json"), {"mode": "resident", "kind": kind,
                  "python": str(python), "gpu": str(gpu), "cwd": str(ROOT), "timeout": timeout,
                  "service_dir": str(directory), "request_id": request_id, "queue_timeout": queue_timeout})
            while True:
                try:
                    with ensure_service(directory, kind, python, gpu, deadline) as conn:
                        with conn.makefile("rwb") as stream:
                            pid = receive(stream)["pid"]
                            conn.settimeout(max(.001, deadline - time.monotonic()))
                            send(stream, {"job": job, "log": str(log), "request_id": request_id})
                            response = receive(stream)
                    break
                except TimeoutError:
                    # Also cover a reconnect waiting for an earlier inference.
                    # PID comes from an authenticated live greeting, never metadata.
                    if pid is not None:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        raise TimeoutError(f"{kind} inference timed out; worker stopped; see {log}")
                    raise
                except (ConnectionError, OSError) as exc:
                    # A lost reply does not require repeating successful inference.
                    if receipt.exists():
                        saved = read(receipt)
                        if saved.get("request_id") == request_id:
                            response = saved["response"]
                            break
                    if retries >= reconnect_attempts or time.monotonic() >= deadline:
                        raise ConnectionError(f"{kind} reconnect exhausted; see {log}") from exc
                    retries += 1
                    time.sleep(min(.2 * 2 ** (retries - 1), 2, max(0, deadline - time.monotonic())))
            if not response["ok"]:
                raise RuntimeError(f"{kind} inference failed: {response['error']}; see {log}")
    except BaseException as exc:
        error = str(exc)
        raise
    finally:
        write(log.with_suffix(".process.json"), {"returncode": 1 if error is not None else 0,
              "seconds": time.time() - started, "pid": pid, "resident": True,
              "service_dir": str(directory) if directory else None, "gpu": gpu,
              "reconnects": retries, "error": error})


@contextlib.contextmanager
def job_logging(path):
    # Redirect file descriptors too, so native libraries' output stays with the job.
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    try:
        with open(path, "w", buffering=1) as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        for fd, backup in zip((1, 2), saved):
            os.dup2(backup, fd)
            os.close(backup)


def execute(kind, job):
    if kind == "lumen":
        from .lumen_worker import main
        main(job)
    else:
        from .workers import run_job
        run_job(kind, job)


def serve(directory, kind):
    directory = Path(directory)
    with (directory / "server.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        token = uuid.uuid4().hex
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind(("127.0.0.1", 0))
            server.listen(32)
            info = {"root": str(ROOT), "kind": kind, "pid": os.getpid(),
                    "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "python": sys.executable,
                    "host": "127.0.0.1", "port": server.getsockname()[1], "token": token}
            write(directory / "service.json", info)
            try:
                while True:
                    conn, _ = server.accept()
                    with conn, conn.makefile("rwb") as stream:
                        try:
                            conn.settimeout(30)
                            if receive(stream).get("token") != token:
                                continue
                            send(stream, {"pid": os.getpid()})
                            request = receive(stream)
                            if request.get("command") == "stop":
                                send(stream, {"ok": True})
                                return
                            if request.get("command") == "status":
                                send(stream, {"ok": True})
                                continue
                            conn.settimeout(None)
                            receipt = Path(request["log"]).with_suffix(".receipt.json")
                            previous = read(receipt) if receipt.exists() else {}
                            if request.get("request_id") and previous.get("request_id") == request["request_id"]:
                                send(stream, previous["response"])
                                continue
                            try:
                                with job_logging(request["log"]):
                                    print(f"Resident {kind} worker PID {os.getpid()}", flush=True)
                                    try:
                                        execute(kind, request["job"])
                                    except Exception:
                                        traceback.print_exc()
                                        raise
                                response = {"ok": True}
                            except Exception as exc:
                                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                            write(receipt, {"request_id": request.get("request_id"), "response": response})
                            send(stream, response)
                        except (OSError, RuntimeError, ValueError):
                            traceback.print_exc()
            finally:
                (directory / "service.json").unlink(missing_ok=True)


def manage(action, kind=None):
    base = Path(tempfile.gettempdir()) / f"lightagent-workers-{os.getuid()}"
    results = []
    for path in sorted(base.glob("*/service.json")):
        try:
            info = read(path)
        except FileNotFoundError:
            continue
        if info["root"] != str(ROOT) or (kind and info["kind"] != kind):
            continue
        try:
            with request_lock(path.parent, time.monotonic() + (1800 if action == "stop" else .1)):
                with connect(path.parent, 5) as conn, conn.makefile("rwb") as stream:
                    info["pid"] = receive(stream)["pid"]
                    send(stream, {"command": action})
                    receive(stream)
                    info["status"] = "stopped" if action == "stop" else "running"
        except TimeoutError:
            info["status"] = "busy"
        except (OSError, RuntimeError):
            info["status"] = "not running"
        info.pop("token", None)
        results.append(info)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("serve", "status", "stop"))
    parser.add_argument("--kind", choices=KINDS)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.action == "serve":
        if args.directory is None or args.kind is None:
            parser.error("serve requires --directory and --kind")
        serve(args.directory, args.kind)
    else:
        print(json.dumps(manage(args.action, args.kind), indent=2))
