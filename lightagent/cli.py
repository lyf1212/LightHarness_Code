from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess

from .io import ROOT, read, write, inside
from .llm import load_env
from .config import default_config, load_config


def doctor(config):
    checks = {"blender": {"available": Path(config["blender"]).is_file()},
              "llm": {"base_url_configured": bool(config["llm"].get("base_url") or os.getenv("OPENAI_BASE_URL")),
                      "model_configured": bool(config["llm"].get("model") or os.getenv("SCENERIG_MODEL") or os.getenv("OPENAI_MODEL")),
                      "key_configured": bool(os.getenv(config["llm"].get("api_key_env", "OPENAI_API_KEY")))}}
    for name in ("sam3", "vggt", "marigold", "lumen"):
        opt = config[name]
        checks[name] = {k: {"path": v, "exists": bool(v and Path(v).exists())}
                        for k, v in opt.items() if k in {"python", "repo", "checkpoint", "flux_checkpoint", "redux_checkpoint", "lora_checkpoint"}}
        if opt.get("python") and Path(opt["python"]).exists():
            packages = {"sam3": ["torch", "numpy", "PIL", "sam3"], "vggt": ["torch", "numpy", "PIL", "cv2", "vggt"],
                        "marigold": ["torch", "diffusers", "transformers"], "lumen": ["torch", "diffusers", "peft", "einops", "cv2", "OpenEXR"]}[name]
            script = "import sys,importlib.util,json;sys.path.insert(0," + repr(opt.get("repo", "")) + ");print(json.dumps({m:importlib.util.find_spec(m) is not None for m in " + repr(packages) + "}))"
            p = subprocess.run([opt["python"], "-c", script], capture_output=True, text=True, timeout=60,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            checks[name]["imports"] = json.loads(p.stdout) if p.returncode == 0 else {"error": p.stderr[-1000:]}
    return checks


def main():
    load_env(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Lighting-aware image-to-Blender agent framework")
    parser.add_argument("--config", default=str(default_config()))
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor"); p.add_argument("--output")
    p = sub.add_parser("prepare"); p.add_argument("--image", required=True); p.add_argument("--run", required=True); p.add_argument("--inventory")
    p = sub.add_parser("run"); p.add_argument("--run", required=True); p.add_argument("--rounds", type=int); p.add_argument("--replay")
    p = sub.add_parser("reconstruct"); p.add_argument("--image", required=True); p.add_argument("--run", required=True); p.add_argument("--rounds", type=int)
    p = sub.add_parser("demo"); p.add_argument("--run", default=str(ROOT / "runs/demo"))
    p = sub.add_parser("batch", help="Reconstruct images from a JSON list concurrently")
    p.add_argument("--input", required=True, help="JSON with images, parallelism and optional model_gpus")
    p.add_argument("--run", required=True)
    p.add_argument("--parallelism", type=int)
    p.add_argument("--rounds", type=int)
    p.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "doctor":
        result = doctor(config)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.output:
            write(inside(ROOT, args.output), result)
        return
    if args.command == "batch":
        from .batch import run_batch
        ok = run_batch(args.input, args.run, config, parallelism=args.parallelism,
                       rounds=args.rounds, resume=args.resume)
        raise SystemExit(0 if ok else 1)
    run = inside(ROOT, args.run)
    if args.command in {"prepare", "reconstruct"}:
        # Fail before an expensive API call when a required model is absent.
        if config["lumen"]["mode"] == "official":
            for key in ("python", "repo", "flux_checkpoint", "lora_checkpoint", "redux_checkpoint"):
                if not Path(config["lumen"][key]).exists():
                    parser.error(f"LUMEN missing {key}: configure it, import a calibrated HDRI, or explicitly use lumen.mode=disabled")
        from .evidence import prepare
        prepare(args.image, run, config, getattr(args, "inventory", None))
    if args.command in {"run", "reconstruct"}:
        from .pipeline import iterate
        iterate(run, args.rounds, getattr(args, "replay", None))
    if args.command == "demo":
        from .demo import make_demo
        make_demo(run, config)


if __name__ == "__main__":
    main()
