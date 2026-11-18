# Standalone entry: blender -b --python scene.py
from pathlib import Path
import json
import runpy
root = Path(__file__).resolve().parent
job = json.loads((root / "settings.json").read_text())
job.update(code=str(root / "geometry.py"), state=str(root / "state.json"),
           camera=str(root / "camera.json"), priors=str(root / "object_priors.json"),
           hdr=str(root / "environment.exr") if (root / "environment.exr").exists() else None,
           output=str(root / "rerender"))
runtime = runpy.run_path(str(root / "runtime.py"))
runtime["main"](job)
