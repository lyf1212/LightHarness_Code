"""Resolve portable configuration paths before saving a run snapshot."""
from __future__ import annotations

import os
from pathlib import Path
import shutil

from .io import ROOT, read

PATH_KEYS = {"python", "repo", "checkpoint", "flux_checkpoint", "lora_checkpoint",
             "redux_checkpoint", "hdr_path"}


def load_config(path):
    config = read(path)
    def resolve(value, executable=False):
        if value is None:
            return None
        value = os.path.expandvars(os.path.expanduser(str(value)))
        if "$" in value:
            raise ValueError(f"Unresolved environment variable in configuration path: {value}")
        if executable and "/" not in value:
            return shutil.which(value) or value
        p = Path(value)
        return str(p.resolve() if p.is_absolute() else (ROOT / p).resolve())
    config["blender"] = resolve(config["blender"], executable=True)
    for name in ("sam3", "vggt", "marigold", "lumen"):
        for key in PATH_KEYS & config.get(name, {}).keys():
            config[name][key] = resolve(config[name][key], executable=key == "python")
    return config


def default_config():
    local = ROOT / "config.local.json"
    return local if local.exists() else ROOT / "configs/default.json"
