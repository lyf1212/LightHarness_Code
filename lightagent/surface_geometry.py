"""SceneRig-style surface evidence, using this project's existing VGGT depth.

RANSAC/SVD fitting and reliability thresholds follow SceneRig's scene_graph.py.
All public vectors use the existing Blender world frame; it is NOT gravity aligned.
No model import, external checkout import, or inference is needed here.
Adapted geometric routines: SceneRig, Copyright (c) 2025 Fugtemypt123, MIT;
see third_party/notices/SceneRig-LICENSE.txt.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .coordinates import rays
from .io import read, write

CONTRACT = "surface_planes_v1"
FIT_THRESHOLD = .01  # current scene gauge, not observed physical meters
MIN_INLIER = .6
MIN_MASK_FRAC = .02
MIN_PIXELS = 10
SURFACE_ROLES = {"floor", "ground", "wall", "ceiling", "table", "tabletop",
                 "counter", "countertop", "support", "support_surface"}
SUPPORT_ROLES = {"floor", "ground", "table", "tabletop", "counter", "countertop",
                 "support", "support_surface"}


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    length = np.linalg.norm(v)
    return v / length if np.isfinite(length) and length > 1e-12 else None


def fit_plane(points, *, seed=0, threshold=FIT_THRESHOLD):
    """Bounded deterministic RANSAC, followed by SVD refitting of its inliers."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < MIN_PIXELS:
        return None
    rng = np.random.default_rng(seed)
    if len(pts) > 5000:
        pts = pts[rng.choice(len(pts), 5000, replace=False)]
    best, count = None, 0
    for _ in range(200):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = unit(np.cross(b-a, c-a))
        if n is None:
            continue
        selected = np.abs((pts-a) @ n) < threshold
        if selected.sum() > count:
            best, count = selected, int(selected.sum())
    if best is None or count < MIN_PIXELS:
        return None
    center = pts[best].mean(axis=0)
    _, singular, vh = np.linalg.svd(pts[best]-center, full_matrices=False)
    # A near-line cannot constrain a plane orientation, even with perfect inliers.
    if singular[1] < max(1e-8, singular[0]*1e-3):
        return None
    n = unit(vh[-1])
    residual = np.abs((pts-center) @ n)
    inliers = residual < threshold
    return {"point": center.tolist(), "normal": n.tolist(),
            "inlier_frac": float(inliers.mean()), "inliers": int(inliers.sum()),
            "sampled_pixels": len(pts),
            "median_residual": float(np.median(residual[inliers]))}


def robust_normal(normals):
    """SceneRig's two-step 25-degree rejection, without its world-axis conversion."""
    n = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    n = n[np.isfinite(n).all(axis=1)]
    lengths = np.linalg.norm(n, axis=1)
    n = n[lengths > 1e-6] / lengths[lengths > 1e-6, None]
    if len(n) < MIN_PIXELS:
        return None
    mean = unit(n.mean(axis=0))
    for _ in range(2):
        if mean is None:
            return None
        keep = n @ mean > np.cos(np.radians(25.))
        if keep.sum() < MIN_PIXELS:
            break
        mean = unit(n[keep].mean(axis=0))
    return mean


def points_world(depth, camera):
    rotation = np.asarray(camera["cv_to_blender_world_rotation"])
    translation = np.asarray(camera["matrix_world"])[:3, 3]
    return (rays(np.asarray(camera["K"]), depth.shape)*depth[..., None]) @ rotation.T + translation


def plumb_normal(normal, up):
    """SceneRig's .7/.3 bands, generalized to an explicitly estimated world up."""
    n = unit(normal)
    cosine = float(n @ up)
    if abs(cosine) >= .7:
        return up.copy(), "support"
    if abs(cosine) <= .3:
        return unit(n-cosine*up), "wall"
    return n, "ambiguous"


def build_surface_geometry(evidence, *, seed=0, source="VGGT depth"):
    evidence = Path(evidence)
    camera = read(evidence / "camera.json")
    depth = np.load(evidence / "geometry/depth.npy")
    valid = np.load(evidence / "geometry/valid.npy").astype(bool)
    valid &= np.isfinite(depth) & (depth > 0)
    points = points_world(depth, camera)
    masks = read(evidence / "masks/index.json")
    inv = read(evidence / "inventory.json")
    camera_position = np.asarray(camera["matrix_world"])[:3, 3]
    image_up = np.asarray(camera["cv_to_blender_world_rotation"]) @ [0., -1., 0.]
    surfaces, fits, support_masks = {}, {}, {}
    for obj in inv["objects"]:
        oid, role = obj["id"], obj.get("role", "object").lower()
        row = {"id": oid, "label": obj.get("label", oid), "role": role,
               "reliable": False, "plane": None}
        surfaces[oid] = row
        if role not in SURFACE_ROLES:
            row["reason"] = "not a planar root surface; do not flatten this object to a plane"
            continue
        item = masks.get(oid, {})
        if not item.get("available"):
            row["reason"] = "reference segmentation unavailable"
            continue
        mask = np.load(evidence / item["path"]).astype(bool)
        if mask.shape != depth.shape:
            raise ValueError(f"Surface mask grid mismatch: {oid}")
        selected = mask & valid
        row.update(mask_pixels=int(mask.sum()), mask_frac=float(mask.mean()),
                   valid_pixels=int(selected.sum()),
                   valid_fraction_of_mask=float(selected.sum()/max(mask.sum(), 1)))
        if role in SUPPORT_ROLES and row["mask_frac"] >= MIN_MASK_FRAC:
            support_masks[oid] = selected
        fit = fit_plane(points[selected], seed=seed)
        if fit is not None:
            row["fit_quality"] = {k: v for k, v in fit.items() if k not in {"point", "normal"}}
            fits[oid] = fit
        if row["mask_frac"] < MIN_MASK_FRAC:
            row["reason"] = "mask covers less than 2% of the frame; orientation is not an anchor"
        elif fit is None:
            row["reason"] = "insufficient or near-collinear trusted points"
        elif fit["inlier_frac"] < MIN_INLIER:
            row["reason"] = "plane inlier fraction below 0.6; mixed or non-planar evidence"
        else:
            row["reliable"] = True

    # Prefer a visible floor; otherwise a support. This is an estimate, never an
    # unannounced rotation of the calibrated camera, lights or HDRI.
    gravity_candidates = []
    normals_path = evidence / "geometry/normals_cv.npy"
    normals = np.load(normals_path) if normals_path.exists() else None
    rotation = np.asarray(camera["cv_to_blender_world_rotation"])
    for oid, selected in support_masks.items():
        fit = fits.get(oid)
        g, method = None, "masked_point_plane"
        if fit and fit["inlier_frac"] >= .5:
            g = unit(fit["normal"])
        elif normals is not None:
            g = robust_normal(normals[selected] @ rotation.T)
            method = "depth_normal_average_2_iterations_25_degrees"
        if g is None:
            continue
        if g @ image_up < 0:
            g = -g
        if g @ image_up < .3:
            continue  # semantic support conflicts with an almost upright plane
        row = surfaces[oid]
        gravity_candidates.append((row["role"] in {"floor", "ground"}, row["valid_pixels"], oid, g, method))
    gravity = {"available": False, "reason": "no usable masked supporting surface; world +Z is image-up, not measured gravity"}
    up = None
    if gravity_candidates:
        _, _, oid, up, method = max(gravity_candidates, key=lambda x: x[:2])
        gravity = {"available": True, "up_world": up.tolist(), "surface_id": oid,
                   "source": method, "is_estimate": True,
                   "world_frame_changed": False}
    for oid, row in surfaces.items():
        if not row["reliable"]:
            continue
        fit = fits[oid]
        point, measured = np.asarray(fit["point"]), np.asarray(fit["normal"])
        if measured @ (camera_position-point) < 0:
            measured = -measured
        normal, kind = (measured.copy(), "unclassified") if up is None else plumb_normal(measured, up)
        # A semantic wall cannot turn into a floor merely because the fit is bad.
        conflict = up is not None and (kind == "ambiguous" or
            (row["role"] == "wall" and kind != "wall") or
            (row["role"] in SUPPORT_ROLES | {"ceiling"} and kind != "support"))
        if conflict:
            row.update(reliable=False, reason="measured orientation is ambiguous or conflicts with surface role")
            continue
        row["plane"] = {"point_world": point.tolist(), "measured_normal_world": measured.tolist(),
                        "normal_world": normal.tolist(), "kind": kind,
                        "normal_sign": "plane orientation is sign-invariant; not a light emission direction",
                        "plumbed_to_estimated_up": up is not None}
        if kind == "wall":
            row["plane"]["horizontal_basis_world"] = unit(np.cross(normal, up)).tolist()
        row["usage"] = "position/orientation anchor only; infer extent, openings and boundaries from reference and mask"
    return {"contract": CONTRACT, "source": source + "; masked RANSAC/SVD planes, no added model",
            "frame": "blender_world", "world_axes": camera["world_axes"],
            "units": "existing scene scale gauge; not measured physical meters",
            "thresholds": {"plane_distance": FIT_THRESHOLD, "min_inlier_frac": MIN_INLIER,
                           "min_mask_frac": MIN_MASK_FRAC, "min_points": MIN_PIXELS,
                           "ransac_iterations": 200, "max_fit_points": 5000},
            "gravity": gravity, "surfaces": surfaces,
            "instructions": ["Use only reliable=true plane records as position/orientation anchors.",
                "No plane for an object does not mean no geometry. Use its description and the reference image.",
                "A plane point lies on the visible surface, not at the object's volume center.",
                "Fit quality is a geometric diagnostic, not a calibrated probability.",
                "All vectors are already in the current Blender world frame; do not rotate the camera or lights.",
                "No dense normal image is supplied; compare surfaces by stable object IDs and numeric plane evidence."]}


def save_surface_geometry(evidence, *, seed=0, source="VGGT depth"):
    result = build_surface_geometry(evidence, seed=seed, source=source)
    write(Path(evidence) / "surface_geometry.json", result)
    return result


def compare_surface_geometry(reference, depth, valid, camera, masks):
    """Sign-invariant plane-angle feedback; every missing expected surface is penalized."""
    points = points_world(depth, camera)
    valid = valid & np.isfinite(depth) & (depth > 0)
    rows, losses, angles = {}, [], []
    for oid, ref in reference["surfaces"].items():
        if not ref["reliable"]:
            continue
        mask = masks.get(oid, np.zeros(depth.shape, bool))
        selected = mask & valid
        fit = fit_plane(points[selected])
        row = {"available": False, "reference_plane": ref["plane"], "rendered_pixels": int(selected.sum())}
        if fit is None or fit["inlier_frac"] < MIN_INLIER:
            row["reason"] = "rendered surface absent or has no reliable plane"
            losses.append(1.)
        else:
            n = np.asarray(fit["normal"])
            nr = np.asarray(ref["plane"]["measured_normal_world"])
            if n @ nr < 0:
                n = -n
            angle = float(np.degrees(np.arccos(np.clip(n @ nr, -1, 1))))
            point = np.asarray(fit["point"])
            distance = float(abs((point-np.asarray(ref["plane"]["point_world"])) @ nr))
            row.update(available=True, value=angle, rendered_point_world=point.tolist(),
                       rendered_normal_world=n.tolist(), point_to_reference_plane_distance=distance,
                       inlier_frac=fit["inlier_frac"])
            angles.append(angle)
            losses.append(angle/90.)
        rows[oid] = row
    return {"available": bool(angles), "value": float(np.mean(angles)) if angles else None,
            "frame": "blender_world", "unit": "degrees", "per_surface": rows,
            "expected_surfaces": len(rows), "matched_surfaces": len(angles),
            "selection_component": float(np.mean(losses)) if losses else 1.,
            "missing_surface_penalty": 1.,
            "comparison": "unsigned angle to measured plane; no depth scale fitting; distance is diagnostic only",
            "reason": None if angles else "no matched reliable reference/render planes"}
