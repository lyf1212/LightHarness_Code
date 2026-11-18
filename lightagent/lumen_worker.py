"""GPU-resident official LUMEN pipelines with calibrated camera projection."""
from __future__ import annotations
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from .io import read, write, save_hdr


def project_condition(image, K, height, width):
    import cv2
    v, u = np.mgrid[:height, :width]
    lon = ((u + .5) / width - .5) * (2*np.pi)
    lat = (.5 - (v + .5) / height) * np.pi
    # CV ray for the declared equirectangular basis.
    d = np.stack([np.cos(lat)*np.sin(lon), -np.sin(lat), np.cos(lat)*np.cos(lon)], -1)
    p = d @ np.asarray(K).T
    z = p[..., 2]
    x = p[..., 0] / np.maximum(z, 1e-8)
    y = p[..., 1] / np.maximum(z, 1e-8)
    im = np.asarray(image.convert("RGB"))
    valid = (z > 0) & (x >= -.5) & (x < im.shape[1]-.5) & (y >= -.5) & (y < im.shape[0]-.5)
    projected = cv2.remap(im, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
    projected[~valid] = 0
    return Image.fromarray(projected)


@lru_cache(maxsize=None)
def load_pipelines(repo_path, flux_path, lora_path, redux_path, vggt_repo, device):
    import torch
    repo = Path(repo_path)
    sys.path.insert(0, str(repo))
    if vggt_repo:
        sys.path.insert(0, vggt_repo)
    spec = importlib.util.spec_from_file_location("lumen_upstream", repo / "inference_single.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Use the same components, adapters and precision as official inference_single.
    # Build once; retain both pipelines on the GPU without CPU offloading.
    FluxTransformer2DModel = module.FluxTransformer2DModel
    CustomFluxTransformer2DModel = module.CustomFluxTransformer2DModel
    DiffusionPipeline = module.DiffusionPipeline
    FluxPriorReduxPipeline = module.FluxPriorReduxPipeline
    LoraConfig = module.LoraConfig
    set_peft_model_state_dict = module.set_peft_model_state_dict
    LUMEN_ROOT = str(repo)
    target_modules = [
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
    ]

    pretrained_model_name_or_path = str(Path(flux_path) / 'transformer')
    transformer = FluxTransformer2DModel.from_pretrained(pretrained_model_name_or_path, local_files_only=True)
    transformer = transformer.to(dtype=torch.bfloat16)
    custom_transformer = CustomFluxTransformer2DModel(transformer)
    pipe = DiffusionPipeline.from_pretrained(
        flux_path, # trained on dev2pro, infer on dev
        transformer=custom_transformer,
        text_encoder=None,
        text_encoder_2=None,
        custom_pipeline=str(Path(LUMEN_ROOT) / 'fluxpaint_single'),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    transformer_lora_config = LoraConfig(
                r=128,
                lora_alpha=128,
                init_lora_weights=True,
                target_modules=target_modules,
                # modules_to_save=modules_to_save,
                lora_bias=False,
    )
    pipe.transformer.add_adapter(transformer_lora_config, adapter_name="lumen")
    lora_ckpt_path = lora_path
    if not Path(lora_ckpt_path).exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_ckpt_path}")
    lora_state_dict = CustomFluxTransformer2DModel.lora_state_dict(
        str(Path(lora_ckpt_path).parent), weight_name=Path(lora_ckpt_path).name, local_files_only=True)
    transformer_lora_state_dict = {
        f"{k.replace('transformer.', '', 1)}": v
        for k, v in lora_state_dict.items()
        if k.startswith("transformer.") and "lora" in k
    }
    if len(transformer_lora_state_dict) == 0:
        raise RuntimeError(f"No LoRA weights found in {lora_ckpt_path}")
    set_peft_model_state_dict(
        pipe.transformer, transformer_lora_state_dict, adapter_name="lumen"
    )
    print(f"[INFO] Loaded {len(transformer_lora_state_dict)} LoRA tensors from {lora_ckpt_path}")


    repo_redux = redux_path
    pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(repo_redux, torch_dtype=torch.bfloat16, local_files_only=True)

    pipe = pipe.to(device)
    pipe_prior_redux.to(device)

    return pipe, pipe_prior_redux


def main(job):
    import os
    import random
    import torch
    opt, out = job["options"], Path(job["output"])
    device = opt.get("device", "cuda")
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    pipe, prior = load_pipelines(opt["repo"], opt["flux_checkpoint"], opt["lora_checkpoint"],
                                 opt["redux_checkpoint"], job.get("vggt_repo"), device)
    # All image/camera state is request-local, including the calibrated projection.
    K = read(job["camera"])["K"]
    image = Image.open(job["image"]).convert("RGB")
    height, width = opt.get("height", 512), opt.get("width", 1024)
    panorama = project_condition(image, K, height, width)
    stem = Path(job["image"]).stem
    panorama.save(out / f"{stem}_panorama.png")
    with torch.inference_mode():
        embeddings = prior([image])
        output = pipe(prompt=None, height=height, width=width, num_inference_steps=30,
                      guidance_scale=3.0, generator=torch.Generator(device=device).manual_seed(job["seed"]),
                      prompt_embeds=embeddings["prompt_embeds"],
                      pooled_prompt_embeds=embeddings["pooled_prompt_embeds"], output_type="np",
                      num_in_batch=1, panorama=[[panorama]], rgb_imgs=[[image]]).images
    ldr = np.clip(output[0], 0, .999)
    hdr = (ldr / (1.0 - ldr)).astype(np.float32)
    try:
        save_hdr(out / f"{stem}.exr", hdr)
    except ImportError:
        np.save(out / f"{stem}.npy", hdr)
    write(out / "runtime.json", {"python": sys.version, "torch": torch.__version__,
          "cuda": torch.version.cuda, "seed": job["seed"], "pid": os.getpid(), "resident": True})
    write(out / "projection.json", {"forward_world": [0, 1, 0], "up_world": [0, 0, 1],
         "projection": "equirectangular", "longitude_sign": -1, "K": K, "patched_projection": "calibrated camera pixel centers",
         "hdr_scale": "unobserved; official inverse-Reinhard output", "probe_position_world": [0, 0, 0]})


if __name__ == "__main__":
    import os
    from .resident import run_resident
    job = read(sys.argv[1])
    run_resident("lumen", job, Path(job["output"]) / "lumen.log", python=sys.executable,
                 gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
