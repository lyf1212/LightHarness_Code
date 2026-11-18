# Results and reproduction protocol

## What each artifact reproduces

| Artifact / command | Scope |
| --- | --- |
| `scripts/summarize_results.py` | Recomputes four image-metric means from archived per-image values and checks the manuscript CSV |
| `scripts/plot_results.py` | Redraws the main and ablation plots from CSV files |
| `evaluation/evaluate_manifest.py` | Recomputes metrics from the exact archived image pairs |
| `scripts/refit_vlm.py` | Refits the later pooled VLM votes |
| `python -m lightagent demo` | Exercises the reconstruction loop with synthetic evidence, replayed agents, and real rendering |
| `examples/.../scene.py` | Renders a real exported reconstruction without model inference |
| `python -m lightagent ... batch` | Runs evidence extraction and remote-model reconstruction again |

Only the last path invokes the full model stack. Local seeds do not fix remote model responses, provider revisions, or GPU numerical differences.

## Main table snapshot

`results/manuscript_results_source.csv` is the manuscript table snapshot. `results/manuscript_per_image_metrics.json` comes from the matching `all_metrics.json`, not the subsequently updated `comparison.json`. The script checks 30, 32, 32, 14, and 32 evaluated image pairs for 3DCodeVerse, GPT-6 Sol, VIGA, weiSceneRig, and LightHarness respectively. Missing results are excluded rather than assigned zero. The comparison is not on a common valid-image subset across all methods.

The scored inputs are each baseline's supplied reference image. LightHarness uses the prepared `evidence/reference.png`. Images are decoded to display RGB, with EXIF orientation applied and alpha composited on black. When necessary, the reference is resized to the render dimensions using Pillow Lanczos. No geometric registration, crop, exposure fitting, or color fitting is performed.

PSNR uses 8-bit RGB error. SSIM uses an 11×11 Gaussian window with sigma 1.5 and channel averaging. LPIPS uses AlexNet v0.1 with input normalization to [-1, 1]. DreamSim uses the official ensemble and its 224×224 preprocessing. LPIPS and DreamSim weights are external dependencies.

The current optional image archive contains 158 reference/render pairs plus 32 original inputs: 348 files in total. It replaces the old 14 weiSceneRig pairs with all 32 SceneRig pairs. `results/evaluation_pairs.json` describes these current pairs; `results/per_image_metrics.json` also retains the two missing 3DCodeVerse entries. Historical per-image values live separately in `results/manuscript_per_image_metrics.json`. The verification script checks every archive file hash.

## Reconstruction settings

The input list contains 10 Hypersim images, 10 BAID images, 7 FiveK images, 4 LuxRemix images, and one scene labeled `scene_rig_winter`. This last input is preserved as a separate source rather than counted as one of the four named datasets. Identifiers and hashes are in `results/scene_manifest.json`.

All 32 runs completed rounds 0–9. A saved `rounds=5` field in the earlier six-scene configuration is not the completed iteration count; the final selection records show 10 completed rounds. The candidate is selected by the minimum fixed internal loss, not by external LPIPS, DreamSim, or VLM preference.

| Batch | Images | Selected-scene render settings |
| --- | ---: | --- |
| `batch25_autodl_p25_20261115_150058` | 25 | 512 samples, OPTIX, OIDN, adaptive threshold 0.01 |
| `batch_six_10rounds` | 6 | 64 samples, CPU, legacy unfiltered settings |
| `batch65_20261116_000917` | 1 | 512 samples, OPTIX, OIDN, adaptive threshold 0.01 |

The runs use seed 42, maximum input side 768, and `gpt-6-astra` with `xhigh` reasoning. The last one-image batch uses streaming; the other recorded configurations use non-streaming requests. These differences are retained in the scene manifest. The Blender runtime uses the Standard view transform with exposure 0. Model evidence is an estimate, not ground-truth geometry or intrinsic decomposition. The median scene depth of 4 is a scale convention for monocular reconstruction.

## VLM version difference

The manuscript snapshot reports LightHarness BT scores of 1.259, 1.620, and 1.388 for completeness, layout, and appearance. The corresponding complete raw vote snapshot was not recovered during packaging, so those numbers are retained as reported values rather than presented as freshly refitted scores.

The later `vlm_results.json` archive contains additional baseline outputs and judgments. Its pooled scores are 1.387, 1.651, and 1.443 for LightHarness. `results/vlm_latest.json` preserves its pair-win counts; `scripts/refit_vlm.py` independently refits and checks these scores. It adds 0.5 pseudo-win in each direction for each observed method pair and mean-centers the fitted scores. Per-image BT scores must not be averaged to reproduce the pooled estimate.

The executable VLM protocol records a **+15°** additional view. The supplied PDF says **±15°** in its table caption. This release documents the recorded +15° protocol and does not infer an unrecorded second view. Candidate-order-inconsistent judgments are discarded. The historical quantitative and VLM archives use `weiSceneRig`; the current image archive uses `SceneRig`. Old VLM scores, including the other methods' scores, are not valid estimates for the changed candidate set. No replacement VLM judgments were requested.

## Ablations

`results/ablation_results_source.csv` contains the matched 16-image subset. The full model has PSNR 14.242 and DreamSim 0.213. Its SSIM and LPIPS are not the best among the ablations. The plotted bars are reported means, without invented confidence intervals. The ablation plot can be regenerated from this CSV; this release does not bundle the separate ablation implementations or claim to regenerate those 16-image runs from the full-method code.

## Limits of the release

The manuscript includes additional application images and user-study scores. The included figures are manuscript exports; they are not proof that all application pipelines or user-study raw data are packaged here. The full reconstruction rerun requires external model weights, an API account, GPU memory, and the specified data. Baseline generation systems are not vendored; the current comparison images are included in the optional image archive for metric evaluation. The old weiSceneRig image pairs are no longer distributed.

## SceneRig image replacement

The replacement uses the existing paper-comparison mapping: indices 1–11 use the top-level SceneRig renderings, 12–22 use `single_pass_12_22`, and 23–32 use the directories listed in `last10_manifest.json`. All reference images are checked against the original benchmark input hashes. `results/scenerig_replacement.json` records the source-relative render paths and hashes.

`SceneRig` is the supplied directory label. Its own README describes direct assistant-authored Blender reconstruction, not execution of the official SceneRig multi-agent pipeline. This provenance is retained. Historical weiSceneRig metrics are not renamed or copied to these new images.

The replacement images have been evaluated for PSNR and SSIM using the same pixel protocol. The 32-image means are 10.241299 dB and 0.432484; the per-image results and hashes are in `results/scenerig_image_metrics.json`. LPIPS, DreamSim and VLM judgments have not been recomputed for this replacement, so they remain unset rather than inherited from weiSceneRig.
