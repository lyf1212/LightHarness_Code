"""Vision inference with models retained on their configured device between jobs."""
from __future__ import annotations
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .coordinates import resize_affine
from .io import read, write


_MODELS = {}


def cached_model(kind, opt, load):
    # Inference settings (seed, resolution, prompts, threshold) do not reload weights.
    key = (kind, opt.get("repo"), opt["checkpoint"], opt["device"])
    if key not in _MODELS:
        _MODELS[key] = load()
    return _MODELS[key]


def amp(torch, device):
    return (torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
            if device.startswith("cuda") else contextlib.nullcontext())


def vggt(job):
    import torch
    import cv2
    opt, out = job["options"], Path(job["output"])
    if opt.get("repo"):
        sys.path.insert(0, opt["repo"])
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    im = Image.open(job["image"]).convert("RGB")
    w, h = im.size
    n = 518
    nw = max(14, round(w * n / max(w, h) / 14) * 14)
    nh = max(14, round(h * n / max(w, h) / 14) * 14)
    left, top = (n - nw) // 2, (n - nh) // 2
    canvas = Image.new("RGB", (n, n), "white")
    canvas.paste(im.resize((nw, nh), Image.Resampling.BICUBIC), (left, top))
    canvas.save(out / "vggt_input.png")
    A = resize_affine(w, h, nw, nh, left, top)
    images = torch.from_numpy(np.array(canvas).astype(np.float32) / 255).permute(2, 0, 1)[None].to(opt["device"])
    def load():
        model = VGGT()
        weights = torch.load(opt["checkpoint"], map_location="cpu", weights_only=True)
        model.load_state_dict(weights.get("model", weights))
        return model.to(opt["device"]).eval()
    model = cached_model("vggt", opt, load)
    with torch.inference_mode(), amp(torch, opt["device"]):
        prediction = model(images)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(prediction["pose_enc"], images.shape[-2:])
    depth = prediction["depth"][0, 0, ..., 0].float().cpu().numpy()
    conf = prediction["depth_conf"][0, 0].float().cpu().numpy()
    K = np.linalg.inv(A) @ intrinsic[0, 0].float().cpu().numpy()
    E = extrinsic[0, 0].float().cpu().numpy()
    v, u = np.mgrid[:h, :w]
    xy = np.stack([u, v, np.ones_like(u)], -1) @ A.T
    def restore(x):
        return cv2.remap(x, xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    np.save(out / "depth_raw.npy", restore(depth))
    np.save(out / "depth_confidence.npy", restore(conf))
    write(out / "vggt.json", {"K": K.tolist(), "extrinsic": E.tolist(), "canonical_to_model": A.tolist(),
                              "model_size": [n, n], "resized_size": [nw, nh], "padding": [left, top],
                              "depth_type": "camera_z", "scale": "unknown", "confidence": "raw score, not probability"})


def marigold(job):
    import torch
    from diffusers import MarigoldIntrinsicsPipeline
    opt, out = job["options"], Path(job["output"])
    dtype = torch.float16 if opt["device"].startswith("cuda") else torch.float32
    pipe = cached_model("marigold", opt, lambda: MarigoldIntrinsicsPipeline.from_pretrained(
        opt["checkpoint"], torch_dtype=dtype, local_files_only=True,
        use_safetensors=True).to(opt["device"]))
    props = pipe.target_properties
    names = props["target_names"]
    for name in ("albedo", "shading", "residual"):
        if name not in names or props[name]["prediction_space"] != "linear":
            raise ValueError("Use marigold-iid-lighting-v1-1, with linear albedo/shading/residual")
    with torch.inference_mode():
        result = pipe(Image.open(job["image"]).convert("RGB"), num_inference_steps=opt.get("steps", 4),
                      ensemble_size=1, processing_resolution=opt.get("resolution", 768),
                      match_input_resolution=True, output_type="np",
                      generator=torch.Generator(device=opt["device"]).manual_seed(job["seed"]))
    pred = np.asarray(result.prediction)
    if pred.ndim == 5 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.ndim != 4 or pred.shape[0] != len(names):
        raise ValueError(f"Unexpected Marigold output {pred.shape}")
    for name in names:
        np.save(out / f"{name}.npy", pred[names.index(name)].astype(np.float32))
    write(out / "marigold.json", {"target_properties": props, "input_resolution_preserved": True,
                                  "color_space": "linear", "formula": "I = A*S + R"})


def sam3(job):
    import torch
    opt, out = job["options"], Path(job["output"])
    sys.path.insert(0, opt["repo"])
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = cached_model("sam3", opt, lambda: build_sam3_image_model(
        checkpoint_path=opt["checkpoint"], device=opt["device"], load_from_HF=False))
    processor = Sam3Processor(model, device=opt["device"], confidence_threshold=opt.get("threshold", .3))
    image = Image.open(job["image"]).convert("RGB")
    w, h = image.size
    inventory = read(job["inventory"])
    records, used = {}, []
    with torch.inference_mode(), amp(torch, opt["device"]):
        state = processor.set_image(image)
        for obj in inventory["objects"]:
            processor.reset_all_prompts(state)
            result = processor.set_text_prompt(state=state, prompt=obj["prompt"])
            candidates = result["masks"]
            scores = result["scores"]
            record = {"id": obj["id"], "prompt": obj["prompt"], "available": False}
            if candidates is not None and len(candidates):
                masks = candidates.detach().cpu().numpy().reshape(-1, h, w) > .5
                scores = scores.detach().float().cpu().numpy().reshape(-1)
                x0, y0, x1, y1 = obj["bbox"]
                box = np.zeros((h, w), bool)
                box[int(y0*h):max(int(y1*h), int(y0*h)+1), int(x0*w):max(int(x1*w), int(x0*w)+1)] = True
                ranking = []
                for i, mask in enumerate(masks):
                    overlap = (mask & box).sum() / max((mask | box).sum(), 1)
                    duplicate = any((mask & m).sum() / max((mask | m).sum(), 1) > .95 for m in used)
                    ranking.append(-1 if duplicate else float(scores[i]) * overlap)
                index = int(np.argmax(ranking))
                if ranking[index] > opt.get("min_match_score", .01):
                    mask = masks[index]
                    used.append(mask)
                    np.save(out / f"{obj['id']}.npy", mask)
                    Image.fromarray(mask.astype(np.uint8)*255).save(out / f"{obj['id']}.png")
                    record.update(available=True, score=float(scores[index]), match_score=ranking[index],
                                  path=f"masks/{obj['id']}.npy", pixels=int(mask.sum()))
                # Retain all candidates and selection metadata for ambiguous/repeated objects.
                np.savez_compressed(out / f"{obj['id']}_candidates.npz", masks=masks, scores=scores)
                record["candidate_ranking"] = ranking
            records[obj["id"]] = record
    write(out / "index.json", records)


def run_job(kind, job):
    import os
    import random
    import torch
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    write(Path(job["output"]) / "runtime.json", {"python": sys.version, "torch": torch.__version__,
          "cuda": torch.version.cuda, "seed": job["seed"], "pid": os.getpid(), "resident": True})
    {"sam3": sam3, "vggt": vggt, "marigold": marigold}[kind](job)


if __name__ == "__main__":
    # Keep the historical CLI while routing it through the resident service.
    import os
    from .resident import run_resident
    kind, job_path = sys.argv[1:3]
    job = read(job_path)
    run_resident(kind, job, Path(job["output"]) / "worker.log",
                 python=sys.executable, gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
