# Model environments

Run these commands from the repository root. The host environment runs orchestration and metrics; each vision model has a separate interpreter. The checked-in configuration uses `.venv-sam3`, `.venv-vggt`, `.venv-marigold`, and `.venv-lumen`.

## Source code

```bash
python scripts/fetch_sources.py
```

The source revisions are recorded in [sources.json](../third_party/sources.json). The experiment used SAM3 commit `2345a4a`, VGGT commit `a288dd0`, and LUMEN commit `dd97e9b`. The fetch script refuses to replace an existing checkout at a different revision.

## Python packages

Create the environments first:

```bash
for model in sam3 vggt marigold lumen; do
  python3.11 -m venv ".venv-$model"
done
```

The original setup instructions used PyTorch 2.7.0 / Torchvision 0.22.0 for LUMEN. The inspected VGGT environment uses 2.7.1 / 0.22.1; SAM3 uses 2.11.0 / 0.26.0 through a shared environment. The Marigold and LUMEN interpreters currently inherit system PyTorch 2.14.0 / Torchvision 0.29.0. These are observations of the existing environments, not a complete historical lock of every run.

The following setup keeps the original documented LUMEN pair and the inspected SAM3/VGGT pairs. It must be validated on the target GPU before a full benchmark; this release's clean-environment validation covers the host and CPU rendering, not a fresh installation of all GPU models.

```bash
.venv-sam3/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
.venv-vggt/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
.venv-marigold/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
.venv-lumen/bin/python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

for model in sam3 vggt marigold lumen; do
  ".venv-$model/bin/python" -m pip install -r "requirements/$model.txt"
done
.venv-sam3/bin/python -m pip install -e third_party/sam3
```

VGGT is imported directly from its checkout. LUMEN also uses that checkout, passed by the worker. The single-image adapter does not use LUMEN's multi-view PyTorch3D path. If your CUDA environment requires different wheel builds, record the resolved package versions alongside your results.

## Weights

The expected layout is:

```text
models/
  sam3/sam3.pt
  vggt/model.pt
  marigold-iid-lighting-v1-1/
  FLUX.1-dev/                       # Diffusers layout, including transformer/
  FLUX.1-Redux-dev/
  lumen/pytorch_lora_weights.safetensors
```

Download from the model publishers using an account with access where required. Run `hf auth login` in the relevant environment if needed. Authentication is not stored in this release.

```bash
.venv-sam3/bin/hf download facebook/sam3 sam3.pt --local-dir models/sam3
.venv-vggt/bin/hf download facebook/VGGT-1B model.pt --local-dir models/vggt
.venv-marigold/bin/hf download prs-eth/marigold-iid-lighting-v1-1 \
  --local-dir models/marigold-iid-lighting-v1-1
.venv-lumen/bin/hf download black-forest-labs/FLUX.1-dev \
  --revision 0ef5fff789c832c5c7f4e127f94c8b54bbcced44 \
  --local-dir models/FLUX.1-dev --exclude 'flux1-dev.safetensors' 'ae.safetensors'
.venv-lumen/bin/hf download BackRabb1t/lumen pytorch_lora_weights.safetensors \
  --revision 345f7a2b55abbbcaf68db0f949f2bd8bfebebe15 --local-dir models/lumen
.venv-lumen/bin/hf download black-forest-labs/FLUX.1-Redux-dev \
  --revision c95859fbf7703ca4d6824b4da4407d7cd0434f81 --local-dir models/FLUX.1-Redux-dev
```

The FLUX revision is taken from the run configuration; the LoRA and Redux revisions are from the experiment's download script. SAM3, VGGT and Marigold were referenced through local files without a recorded remote revision. Keep the downloaded snapshot metadata and file hashes for new runs. The archived images and measurements remain the reference for the reported numbers.

## Resources and configuration

All models remain on their assigned devices after first use. The release does not provide a measured minimum VRAM requirement. In particular, LUMEN holds both the FLUX and Redux pipelines on the GPU without CPU offloading. A single-GPU configuration is a path template, not a claim that all models fit on a small card.

Assign `gpu` separately in each model section, or use `gpu_pool` for replicas. Set `render.gpu` independently for Cycles. GPU IDs are physical device IDs; each worker sees its assigned GPU as logical CUDA device 0. A batch can override routing with `model_gpus`.

Run `python -m lightagent doctor` after configuration. The report checks file presence and imports; it does not certify successful inference or sufficient GPU memory. Start with one image and `--rounds 1` before increasing the batch size.

`lumen.mode="disabled"` is an explicit ablation and changes the method. Importing an external HDRI also changes the evidence source; configure its directional basis using `hdr_path`, `forward_world`, `up_world`, and `longitude_sign`. Do not use either setting to claim a full LUMEN rerun.
