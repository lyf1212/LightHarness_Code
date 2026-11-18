# Release notes

The code was copied from `exp_1113_1445_rule` into a separate release directory. The original experiment, credentials, model files, run archives, and running workers were not changed.

The release adds portable configuration loading, public configuration templates, model-source revision records, metric reproduction scripts, result plotting, image hashes, and a standalone exported scene. The reconstruction algorithm, prompts, selection weights, and state-update rules are retained. Local API gateways and machine-specific model paths were removed from the distributed configuration. The original `.env`, virtual environments, caches, request archives, and model weights are excluded.

## Exported examples

The `baid_input_0000` example comes from `exp_1113_1445_rule/runs/batch25_autodl_p25_20261115_150058`. It includes the selected final export and processed reference image. See the [example documentation](../examples/batch25_autodl_p25_20261115_150058/README.md) for render commands, settings, and per-file provenance.

## Code map

| File | Role |
| --- | --- |
| `lightagent/evidence.py`, `workers.py`, `lumen_worker.py` | Fixed visual evidence and model adapters |
| `lightagent/pipeline.py` | Generator–render–verifier loop and final selection |
| `lightagent/state.py`, `light_rules.py` | Persistent source identities, parameters, validated patches |
| `lightagent/object_review.py` | Structured reviews and cross-round object history |
| `lightagent/metrics.py`, `surface_geometry.py` | Fixed selection rubric and surface-plane evidence |
| `lightagent/blender_host.py` | Blender host, rendering and light-source passes |
| `lightagent/batch.py`, `resident.py` | Batch scheduling and persistent workers |
| `lightagent/llm.py` | API requests, retries and replay |

## Third-party material

SAM3, VGGT, LUMEN, Marigold, FLUX, Blender, and evaluation models retain their publishers' terms. Third-party source code and model weights are fetched separately. The original SceneRig notice is preserved under `third_party/notices/`; the surface-plane workflow references that implementation. Benchmark images and manuscript figures retain their existing rights. The optional image archive is kept separate from the code package for this reason.

No new license has been assigned to the authors' code. A repository-wide license choice is still needed before distributing it as an open-source project. This package does not assert a license for third-party images or weights.

## Validation

The release is checked in an isolated Python 3.11 environment using the pinned host requirements. The validation report records tests, real CPU rendering, metric recomputation, archive integrity, and figure checks. It also identifies anything not rerun. A full paid-API/GPU benchmark is not part of release packaging.

See `release_validation.json` for the machine-readable results and `MANIFEST.sha256` for the distributed file hashes. The optional image archive has a separate per-file hash manifest under `results/`.

## Documentation review

The README describes the implementation without claiming a new contribution beyond the manuscript. Its numerical statements are tied to the archived table and per-image records. It reports the SSIM exception, different sample counts, mixed rendering settings, and the later VLM revision. Full-method and ablation reproduction are distinguished. Hardware minima, a unified historical dependency lock, and an exact remote-model rerun are not claimed because the source records do not establish them.
