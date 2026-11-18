"""Pixel centers: integer (u,v), top-left origin. CV: +X right,+Y down,+Z forward."""
from __future__ import annotations
import numpy as np

# A right-handed world tied to the reference camera, not an inferred gravity frame.
CV_TO_WORLD = np.array([[1., 0, 0], [0, 0, 1], [0, -1, 0]])
GL_TO_CV = np.diag([1., -1., -1.])


def resize_affine(width, height, new_width, new_height, left=0, top=0):
    sx, sy = new_width / width, new_height / height
    return np.array([[sx, 0, left + (sx - 1) / 2],
                     [0, sy, top + (sy - 1) / 2], [0, 0, 1.]])


def rays(K, shape):
    h, w = shape
    v, u = np.mgrid[:h, :w]
    p = np.stack([u, v, np.ones_like(u)], -1)
    r = p @ np.linalg.inv(K).T
    return r / r[..., 2:3]


def range_to_z(distance, K):
    return distance / np.linalg.norm(rays(K, distance.shape), axis=-1)


def z_to_range(depth, K):
    return depth * np.linalg.norm(rays(K, depth.shape), axis=-1)


def normals_from_z(depth, K, valid=None):
    points = rays(K, depth.shape) * depth[..., None]
    dy, dx = np.gradient(points, axis=(0, 1))
    n = np.cross(dy, dx)  # toward camera for a front-facing plane
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
    valid = np.isfinite(depth) & (depth > 0) if valid is None else valid.copy()
    valid[[0, -1], :] = False
    valid[:, [0, -1]] = False
    gy, gx = np.gradient(depth)
    valid &= (np.abs(gx) + np.abs(gy)) / np.maximum(depth, 1e-6) < .08
    n[~valid] = 0
    return n.astype(np.float32), valid


def camera_contract(K, native_extrinsic, shape, depth_scale):
    E = np.eye(4)
    E[:3] = np.asarray(native_extrinsic)[:3]
    world_from_native = np.eye(4)
    world_from_native[:3, :3] = CV_TO_WORLD * depth_scale
    world_from_native = world_from_native @ E
    matrix = np.eye(4)
    matrix[:3, :3] = CV_TO_WORLD @ GL_TO_CV
    return {"K": np.asarray(K).tolist(), "width": shape[1], "height": shape[0],
            "matrix_world": matrix.tolist(), "vggt_world_to_camera_cv": E.tolist(),
            "blender_world_from_vggt_world": world_from_native.tolist(),
            "cv_to_blender_world_rotation": CV_TO_WORLD.tolist(),
            "depth_type": "camera_z", "depth_scale_to_scene_units": float(depth_scale),
            "scene_unit": "meter_gauge", "metric_scale_observed": False,
            "world_axes": "+X image-right, +Y camera-forward, +Z image-up; gravity is not observed",
            "pixel_coordinates": "integer centers; top-left origin; EXIF orientation applied"}


def canonicalize_environment(hdr, forward, up, longitude_sign=1):
    """Resample an explicitly declared equirectangular basis to Blender's native basis.

    Input center looks along forward; increasing longitude points toward
    longitude_sign*cross(up,forward); north pole points up.
    Blender uses u=-atan2(y,x)/(2pi)+.5, v=acos(z)/pi (top-origin):
    center +X, north +Z, increasing u toward -Y. Verified by an axis-color render.
    """
    import cv2
    f, up = np.asarray(forward, float), np.asarray(up, float)
    if any(v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) < 1e-8 for v in (f, up)):
        raise ValueError("Environment forward and up must be finite nonzero 3-vectors")
    f /= np.linalg.norm(f)
    up /= np.linalg.norm(up)
    if abs(f @ up) > 1e-5:
        raise ValueError("Environment forward and up must be orthogonal")
    if longitude_sign not in {-1, 1}:
        raise ValueError("longitude_sign must be +1 or -1")
    right = longitude_sign * np.cross(up, f)
    h, w = hdr.shape[:2]
    v, u = np.mgrid[:h, :w]
    lon, lat = ((u + .5) / w - .5) * 2 * np.pi, (.5 - (v + .5) / h) * np.pi
    d = np.stack([np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
    x = ((np.arctan2(d @ right, d @ f) / (2 * np.pi) + .5) * w - .5).astype(np.float32)
    y = ((.5 - np.arcsin(np.clip(d @ up, -1, 1)) / np.pi) * h - .5).astype(np.float32)
    # Longitude wraps; latitude clamps. Pad only the horizontal seam.
    padded = np.concatenate([hdr[:, -1:], hdr, hdr[:, :1]], axis=1)
    return cv2.remap(padded, (x % w) + 1, np.clip(y, 0, h - 1), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
