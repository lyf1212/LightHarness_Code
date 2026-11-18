#!/usr/bin/env python3
"""Independent VLM protocol for source and +30-degree nearby-view renders.

Pairwise mode follows the *reported* SceneRig procedure: anonymous A/B panels,
both candidate orders, order-consistency filtering, half-credit ties and
mean-centered Bradley--Terry scores. The original benchmark prompts and judge
implementations are not public here, so this is an independently specified adapter.

Single-candidate mode is an explicitly supplemental anchored rubric, not a
SceneRig Bradley--Terry score.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


CRITERIA = ("completeness", "layout", "appearance")
PROMPT_VERSION = "scenerig_nearby_vlm_independent_v1"
SYSTEM_PROMPT = (
    "You are a visual evaluator of editable 3D scene reconstructions. "
    "Use the input photograph as evidence for visible content. The nearby views "
    "have no ground-truth photograph: use them to assess whether the reconstructed "
    "objects and arrangement remain plausible when the camera moves. "
    "Do not reward a method just for copying source-view pixels onto flat geometry. "
    "Evaluate only the requested criterion. Return one JSON object and no markdown."
)
CRITERION_TEXT = {
    "completeness": "Which reconstruction includes more of the visible objects and supporting scene structure, with fewer missing, fused, or spurious objects?",
    "layout": "Which reconstruction better preserves the objects' positions, relative sizes, and support relationships seen in the input, with spatially plausible structure in the nearby view?",
    "appearance": "Which reconstruction better preserves object identity, shape, color, material, and texture, including plausible appearance in the nearby view?",
}


@dataclass(frozen=True)
class Candidate:
    name: str
    source: Path
    nearby: Path


def file_record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    with Image.open(path) as opened:
        opened.verify()
    with Image.open(path) as opened:
        size = list(opened.size)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "size_wh": size}


def _font() -> ImageFont.ImageFont:
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(font), 18) if font.is_file() else ImageFont.load_default()


def make_panel(reference: Path, ordered: list[Candidate], width: int, nearby_degrees: int = 30) -> Image.Image:
    """One photo row and one two-view row per anonymized candidate."""
    with Image.open(reference) as opened:
        aspect = opened.height / opened.width
    height = round(width * aspect)
    caption = 33
    panel = Image.new("RGB", (2 * width, (len(ordered) + 1) * (height + caption)), "white")
    draw = ImageDraw.Draw(panel)
    font = _font()

    def tile(path: Path, x: int, y: int, label: str) -> None:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image)
        image = background.convert("RGB")
        fitted = ImageOps.contain(image, (width, height), method=Image.Resampling.LANCZOS)
        panel.paste(fitted, (x + (width - fitted.width) // 2, y + caption + (height - fitted.height) // 2))
        draw.rectangle((x, y, x + width - 1, y + caption - 1), fill=(235, 238, 242))
        draw.text((x + 8, y + 5), label, fill=(0, 0, 0), font=font)

    tile(reference, width // 2, 0, "Input photograph")
    for index, candidate in enumerate(ordered):
        y = (index + 1) * (height + caption)
        letter = chr(ord("A") + index)
        tile(candidate.source, 0, y, f"{letter}: source view")
        tile(candidate.nearby, width, y, f"{letter}: {nearby_degrees:+d} deg nearby view")
    return panel


def image_data_url(panel: Image.Image) -> str:
    buffer = io.BytesIO()
    panel.save(buffer, format="JPEG", quality=90, subsampling=0)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_json_reply(value: str) -> dict:
    decoder = json.JSONDecoder()
    for index, character in enumerate(value):
        if character == "{":
            try:
                result, _ = decoder.raw_decode(value[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(result, dict):
                return result
    raise ValueError("VLM did not return a JSON object")


def parse_pair_reply(value: str) -> dict:
    payload = parse_json_reply(value)
    winner = str(payload.get("winner", "")).strip().upper()
    if winner not in {"A", "B", "TIE"}:
        raise ValueError("VLM winner must be A, B, or tie")
    return {"winner": winner, "reason": str(payload.get("reason", ""))[:2000]}


def parse_single_reply(value: str) -> dict:
    payload = parse_json_reply(value)
    score = payload.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 5:
        raise ValueError("VLM score must be a finite number from 0 to 5")
    return {"score": float(score), "reason": str(payload.get("reason", ""))[:2000]}


def _content_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text")
    raise ValueError("VLM response has no text content")


def call_vlm(base_url: str, api_key: str, model: str, panel: Image.Image, prompt: str,
             timeout: float, retries: int) -> tuple[str, dict]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": model,
        "max_completion_tokens": 512,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url(panel)}},
            ]},
        ],
    }).encode("utf-8")
    for attempt in range(retries + 1):
        request = urllib.request.Request(endpoint, data=body, method="POST", headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        })
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            choice = payload["choices"][0]
            return _content_text(choice["message"]), {
                "response_model": payload.get("model"),
                "usage": payload.get("usage"),
                "finish_reason": choice.get("finish_reason"),
            }
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == retries:
                raise RuntimeError(f"VLM HTTP {exc.code} from {endpoint}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f"VLM transport failed after {retries + 1} attempts: {type(exc).__name__}") from exc
        except ValueError as exc:
            if attempt == retries:
                raise RuntimeError(f"VLM response could not be read after {retries + 1} attempts: {exc}") from exc
        time.sleep(min(2 ** attempt, 10))
    raise AssertionError("unreachable")


def pair_prompt(criterion: str, nearby_degrees: int = 30) -> str:
    return (
        f"Compare the anonymous reconstructions A and B for {criterion}. "
        + CRITERION_TEXT[criterion]
        + f" Treat the input photograph as the source-view reference; the {nearby_degrees:+d} degree views "
        "have no measured reference. If evidence is indistinguishable, choose tie. "
        'Return a JSON object with keys "winner" and "reason". '
        'The winner value must be "A", "B", or "tie". '
        "Do not infer the method identity from image order."
    )


def single_prompt(criterion: str) -> str:
    return (
        f"Assess this one reconstruction for {criterion}. " + CRITERION_TEXT[criterion]
        + " Give an anchored 0 to 5 score: 0=severe failure, 1=mostly wrong, "
        "2=major omissions or distortions, 3=recognizable with substantial errors, "
        "4=mostly faithful with minor errors, 5=highly faithful. "
        "Use the nearby view only as evidence of 3D plausibility, because it has no real reference. "
        'Return exactly {"score":number,"reason":"brief evidence"}.'
    )


def canonical_vote(winner: str, order: tuple[int, int]) -> int | None:
    if winner == "TIE":
        return None
    return order[0] if winner == "A" else order[1]


def fit_bradley_terry(names: list[str], pair_counts: dict[tuple[int, int], tuple[float, float]]) -> dict[str, float] | None:
    """Maximum-likelihood fit, with 0.5 pseudo-win in both directions per pair."""
    n = len(names)
    if not pair_counts:
        return None
    linked = {0}
    while True:
        extended = linked | {j for i, j in pair_counts if i in linked} | {i for i, j in pair_counts if j in linked}
        if extended == linked:
            break
        linked = extended
    if len(linked) != n:
        return None
    score = np.zeros(n, dtype=np.float64)
    for _ in range(100):
        gradient = np.zeros(n, dtype=np.float64)
        hessian = np.zeros((n, n), dtype=np.float64)
        for (i, j), (wins_i, wins_j) in pair_counts.items():
            a, b = wins_i + 0.5, wins_j + 0.5
            delta = float(np.clip(score[i] - score[j], -40, 40))
            probability = 1 / (1 + math.exp(-delta))
            g = (a + b) * probability - a
            h = (a + b) * probability * (1 - probability)
            gradient[i] += g
            gradient[j] -= g
            hessian[i, i] += h
            hessian[j, j] += h
            hessian[i, j] -= h
            hessian[j, i] -= h
        step = np.linalg.solve(hessian[:-1, :-1] + np.eye(n - 1) * 1e-10, gradient[:-1])
        step = np.clip(step, -2, 2)
        score[:-1] -= step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    score -= score.mean()
    return {name: float(value) for name, value in zip(names, score)}


def summarize_pairs(names: list[str], judgments: dict[str, dict], models: list[str]) -> dict:
    summary = {}
    for criterion in CRITERIA:
        counts: dict[tuple[int, int], list[float]] = {}
        retained = discarded = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                for model in models:
                    key_ab = f"{model}|{criterion}|{i},{j}|ab"
                    key_ba = f"{model}|{criterion}|{i},{j}|ba"
                    if key_ab not in judgments or key_ba not in judgments:
                        continue
                    first = canonical_vote(judgments[key_ab]["parsed"]["winner"], (i, j))
                    second = canonical_vote(judgments[key_ba]["parsed"]["winner"], (j, i))
                    if first != second:
                        discarded += 1
                        continue
                    wins = counts.setdefault((i, j), [0.0, 0.0])
                    if first is None:
                        wins[0] += 0.5
                        wins[1] += 0.5
                    elif first == i:
                        wins[0] += 1
                    else:
                        wins[1] += 1
                    retained += 1
        score = fit_bradley_terry(names, {k: tuple(v) for k, v in counts.items()})
        summary[criterion] = {
            "mean_centered_bt_scores": score,
            "retained_pair_judgments": retained,
            "discarded_order_inconsistent_pair_judgments": discarded,
            "pair_win_counts_before_pseudowins": {f"{names[i]} vs {names[j]}": v for (i, j), v in counts.items()},
            "pseudowin_per_direction_per_observed_pair": 0.5,
        }
    return summary


def _write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def run(args: argparse.Namespace) -> dict:
    reference = args.reference.resolve()
    candidates = [Candidate(name, Path(source).resolve(), Path(nearby).resolve()) for name, source, nearby in args.candidate]
    names = [c.name for c in candidates]
    if len(names) != len(set(names)) or any(not name.strip() or "|" in name for name in names):
        raise ValueError("Candidate names must be unique, nonempty, and must not contain '|'")
    if not candidates:
        raise ValueError("At least one --candidate NAME SOURCE_RENDER NEAR_RENDER is required")
    if not 128 <= args.tile_width <= 1024:
        raise ValueError("--tile-width must be between 128 and 1024")
    models = list(dict.fromkeys(args.model or ["gpt-6-astra"]))
    nearby_degrees = getattr(args, "nearby_degrees", 30)
    inputs = {"reference": file_record(reference), "candidates": [
        {"name": c.name, "source": file_record(c.source), "nearby_view": file_record(c.nearby)} for c in candidates
    ]}
    manifest = {"prompt_version": PROMPT_VERSION, "protocol": "single_candidate_supplemental_rubric" if len(candidates) == 1 else "SceneRig_style_pairwise_VLM_independent_adapter",
                "models": models, "base_url": args.base_url.rstrip("/"), "tile_width": args.tile_width,
                "max_completion_tokens": 512,
                "rgba_composite_background_rgb": [255, 255, 255],
                "inputs": inputs, "nearby_camera_orbit_degrees": nearby_degrees,
                "nearby_view_pose_verified_by_script": False}
    if args.output.exists() and not args.overwrite:
        report = json.loads(args.output.read_text())
        if report.get("manifest") != manifest:
            raise ValueError("Existing output has a different manifest; use a new --output or --overwrite")
    else:
        report = {"manifest": manifest, "judgments": {}, "status": "in_progress"}
        _write(args.output, report)
    panels = args.output.parent / (args.output.stem + "_panels")
    panels.mkdir(parents=True, exist_ok=True)
    if args.preview_only:
        if len(candidates) == 1:
            make_panel(reference, candidates, args.tile_width, nearby_degrees).save(panels / "single.png")
        else:
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    make_panel(reference, [candidates[i], candidates[j]], args.tile_width, nearby_degrees).save(panels / f"{i}_{j}_ab.png")
                    make_panel(reference, [candidates[j], candidates[i]], args.tile_width, nearby_degrees).save(panels / f"{i}_{j}_ba.png")
        report["status"] = "preview_only"
        _write(args.output, report)
        return report
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {args.api_key_env} is required for VLM evaluation")

    def evaluate(key: str, model: str, criterion: str, panel: Image.Image, prompt: str, parser, panel_path: Path) -> None:
        if key in report["judgments"]:
            return
        panel.save(panel_path)
        last_error = None
        for _ in range(args.parse_retries + 1):
            raw, meta = call_vlm(args.base_url, api_key, model, panel, prompt, args.timeout, args.http_retries)
            try:
                parsed = parser(raw)
            except ValueError as exc:
                last_error = exc
                report.setdefault("invalid_responses", []).append({"key": key, "model": model,
                    "criterion": criterion, "raw_text": raw, "api": meta, "error": str(exc)})
                _write(args.output, report)
                continue
            report["judgments"][key] = {"model": model, "criterion": criterion, "panel": str(panel_path),
                                        "parsed": parsed, "raw_text": raw, "api": meta}
            _write(args.output, report)
            return
        raise ValueError(f"Invalid VLM response for {key}: {last_error}")

    if len(candidates) == 1:
        panel = make_panel(reference, candidates, args.tile_width, nearby_degrees)
        for model in models:
            for criterion in CRITERIA:
                key = f"{model}|{criterion}|single"
                evaluate(key, model, criterion, panel, single_prompt(criterion), parse_single_reply,
                         panels / "single.png")
        report["scores"] = {criterion: {
            "mean_0_to_5": float(np.mean([report["judgments"][f"{model}|{criterion}|single"]["parsed"]["score"] for model in models])),
            "per_model_0_to_5": {model: report["judgments"][f"{model}|{criterion}|single"]["parsed"]["score"] for model in models},
        } for criterion in CRITERIA}
    else:
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                for suffix, order in (("ab", (i, j)), ("ba", (j, i))):
                    panel = make_panel(reference, [candidates[k] for k in order], args.tile_width, nearby_degrees)
                    for model in models:
                        for criterion in CRITERIA:
                            key = f"{model}|{criterion}|{i},{j}|{suffix}"
                            evaluate(key, model, criterion, panel, pair_prompt(criterion, nearby_degrees), parse_pair_reply,
                                     panels / f"{i}_{j}_{suffix}.png")
        report["scores"] = summarize_pairs(names, report["judgments"], models)
    report["status"] = "complete"
    _write(args.output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Input photograph")
    parser.add_argument("--candidate", action="append", nargs=3, metavar=("NAME", "SOURCE_RENDER", "NEAR_RENDER"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", action="append", help="VLM model ID; repeat to pool judges (default: gpt-6-astra)")
    parser.add_argument("--nearby-degrees", type=int, default=30, help="Orbit angle of supplied nearby renders")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Name of credential environment variable; value is never written")
    parser.add_argument("--tile-width", type=int, default=384)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--http-retries", type=int, default=2)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--preview-only", action="store_true", help="Save labeled panels without calling the VLM")
    parser.add_argument("--overwrite", action="store_true", help="Discard prior judgments at --output")
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception as exc:
        print(f"VLM evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "protocol": report["manifest"]["protocol"],
                      "scores": report.get("scores"), "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
