from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inside(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Output must stay inside {root}: {path}")
    return path


def run_process(argv, log, *, cwd=ROOT, timeout=1800, env=None):
    """No shell; terminate the complete child group on timeout."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    child_env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENCV_IO_ENABLE_OPENEXR="1")
    child_env.update(env or {})
    started = time.time()
    write(log.with_suffix(".command.json"), {"argv": list(map(str, argv)), "cwd": str(cwd), "timeout": timeout})
    with log.open("w") as f:
        p = subprocess.Popen(list(map(str, argv)), cwd=cwd, env=child_env, stdout=f,
                             stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = p.wait(timeout=timeout)
        except BaseException:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            raise
    write(log.with_suffix(".process.json"), {"returncode": code, "seconds": time.time() - started})
    if code:
        raise RuntimeError(f"Process failed ({code}); see {log}")


def srgb_to_linear(x):
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.maximum(np.asarray(x, dtype=np.float32), 0)
    return np.where(x <= .0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - .055)


def preview(path, x, kind="linear"):
    x = np.nan_to_num(np.asarray(x), nan=0, posinf=0, neginf=0)
    if kind == "normal":
        x = x * .5 + .5
    elif kind in {"depth", "scalar"}:
        valid = x[np.isfinite(x) & (x > 0)]
        lo, hi = np.percentile(valid, [2, 98]) if valid.size else (0, 1)
        x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    elif kind == "hdr":
        x = linear_to_srgb(x / (1 + x))
    elif kind == "linear":
        x = linear_to_srgb(x)
    Image.fromarray(np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)).save(path)


def load_hdr(path):
    if Path(path).suffix == ".npy":
        x = np.load(path, allow_pickle=False)
    elif Path(path).suffix.lower() == ".exr":
        x = read_exr(path)
    else:
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        import cv2
        x = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if x is None:
            raise ValueError(f"Cannot read HDR: {path}")
        x = x[..., :3][..., ::-1]
    if x.ndim != 3 or x.shape[2] != 3 or x.dtype.kind != "f" or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("HDR must be finite nonnegative linear HWC RGB floating point")
    return x.astype(np.float32)


def save_hdr(path, x):
    import OpenEXR
    import Imath
    x = np.asarray(x, np.float32)
    header = OpenEXR.Header(x.shape[1], x.shape[0])
    header["channels"] = {c: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for c in "RGB"}
    f = OpenEXR.OutputFile(str(path), header)
    try:
        f.writePixels({c: x[..., i].tobytes() for i, c in enumerate("RGB")})
    finally:
        f.close()


def read_exr(path):
    import OpenEXR
    import Imath
    f = OpenEXR.InputFile(str(path))
    try:
        header = f.header()
        window = header["dataWindow"]
        shape = (window.max.y-window.min.y+1, window.max.x-window.min.x+1)
        names = header["channels"]
        keys = [c for c in "RGB" if c in names]
        if len(keys) != 3:
            keys = [c for c in "XYZ" if c in names]
        if len(keys) != 3 and len(names) == 1:
            keys = list(names)
        if len(keys) not in {1, 3}:
            raise ValueError(f"Expected RGB/vector/scalar EXR channels in {path}: {list(names)}")
        return np.stack([np.frombuffer(f.channel(c, Imath.PixelType(Imath.PixelType.FLOAT)), np.float32).reshape(shape)
                         for c in keys], -1)
    finally:
        f.close()
