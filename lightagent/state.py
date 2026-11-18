"""Transactional persistent identities. The event history is host-owned."""
from __future__ import annotations
import copy
import math
import re
from .light_rules import (SOURCE_TYPES, SUN_ANGLE, GROUP_FIELDS, source_type,
                          validate_light_sources, require_new_light_links)

ID = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")
LIGHT_FIELDS = {"label", "visibility", "status", "confidence", "evidence", "elements",
                "occluded_by", "image_bbox", "notes", "environment_relation", "source_type",
                "illumination_group_id", "approximation"}


def check_id(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError(f"Invalid stable ID: {value!r}")


def finite_vector(x, n):
    if not isinstance(x, list) or len(x) != n or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in x):
        raise ValueError(f"Expected finite {n}-vector: {x}")


def validate(state):
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported state schema")
    owners = set()
    for oid, obj in state["objects"].items():
        check_id(oid)
        if obj["id"] != oid:
            raise ValueError("Object ID/key mismatch")
    for lid, light in state["lights"].items():
        check_id(lid)
        if lid == "world":
            raise ValueError("world is reserved for the environment light group")
        if lid != light["id"]:
            raise ValueError("Light ID/key mismatch")
        if light["visibility"] not in {"visible", "partial", "occluded", "off_frame"}:
            raise ValueError("Invalid light visibility")
        if light["status"] not in {"active", "inactive", "hypothesis", "retired"}:
            raise ValueError("Invalid light status")
        if not 0 <= light["confidence"] <= 1 or not light.get("evidence"):
            raise ValueError("Light requires evidence and confidence")
        if light.get("environment_relation") not in {"independent", "possibly_in_hdri", "represented_in_hdri"}:
            raise ValueError("Declare light/environment relationship")
        if set(light.get("occluded_by", [])) - set(state["objects"]):
            raise ValueError("Unknown occluder object ID")
        local_ids = set()
        for e in light["elements"]:
            check_id(e["id"])
            if e["id"] in local_ids:
                raise ValueError("Duplicate element ID")
            local_ids.add(e["id"])
            if not isinstance(e.get("enabled", True), bool):
                raise ValueError("Element enabled must be boolean")
            if e["kind"] not in {"AREA", "POINT", "SUN", "SPOT", "EMISSIVE_MESH", "GEOMETRY"}:
                raise ValueError("Unknown Blender light element type")
            if e["kind"] in {"GEOMETRY", "EMISSIVE_MESH"}:
                name = e["object_name"]
                if name in owners:
                    raise ValueError("One object cannot be owned by multiple light elements")
                owners.add(name)
            else:
                if e["kind"] != "SUN" or "position" in e:
                    finite_vector(e["position"], 3)
                finite_vector(e["direction"], 3)
                if sum(v*v for v in e["direction"]) < 1e-10:
                    raise ValueError("Light direction cannot be zero")
            if e["kind"] != "GEOMETRY":
                finite_vector(e["color"], 3)
                if min(e["color"]) < 0 or max(e["color"]) > 1:
                    raise ValueError("Light color must be linear RGB in [0,1]")
                if not math.isfinite(e["energy"]) or e["energy"] < 0:
                    raise ValueError("Energy must be finite and nonnegative")
                if e["kind"] == "AREA":
                    finite_vector(e["size"], 2)
                    if min(e["size"]) <= 0:
                        raise ValueError("Area dimensions must be positive")
                if e["kind"] in {"POINT", "SPOT"} and (not math.isfinite(e.get("radius", .05)) or e.get("radius", .05) <= 0):
                    raise ValueError("Radius must be finite and positive")
                if e["kind"] == "SUN" and not 0 <= e.get("angle", SUN_ANGLE) <= math.pi:
                    raise ValueError("Sun angle must be in [0,pi] radians")
                if e["kind"] == "SPOT":
                    if not math.pi/180 <= e.get("spot_size", math.pi/3) <= math.pi or not 0 <= e.get("spot_blend", .15) <= 1:
                        raise ValueError("Invalid spotlight cone size/blend")
        # An identity may retain retired bindings, but its active version is explicit.
    world = state["world"]
    if world["mode"] not in {"evidence_only", "world"}:
        raise ValueError("World mode must be evidence_only or world")
    for key in ("strength", "rotation_z"):
        if not math.isfinite(world[key]):
            raise ValueError("Invalid world parameter")
    if world["strength"] < 0:
        raise ValueError("Invalid world strength")
    validate_light_sources(state)
    return state


def initialize(inventory):
    objects, lights = {}, {}
    for obj in inventory["objects"]:
        check_id(obj["id"])
        if obj["id"] in objects:
            raise ValueError("Duplicate object ID")
        finite_vector(obj["bbox"], 4)
        x0, y0, x1, y1 = obj["bbox"]
        if not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1:
            raise ValueError("Inventory bbox must be normalized xyxy")
        objects[obj["id"]] = copy.deepcopy(obj)
    for light in inventory["lights"]:
        if light.get("source_type") not in SOURCE_TYPES:
            raise ValueError("Every newly initialized light must declare source_type: direct_sun/sky_diffuse/artificial/unknown")
        if light["id"] in lights:
            raise ValueError("Duplicate light ID")
        require_new_light_links(light)
        lights[light["id"]] = copy.deepcopy(light)
    groups = {}
    for group in inventory.get("illumination_groups", []):
        check_id(group["id"])
        if group["id"] in groups:
            raise ValueError("Duplicate illumination group ID")
        groups[group["id"]] = copy.deepcopy(group)
    s = {"schema_version": 1, "revision": 0, "objects": objects, "lights": lights,
         "illumination_groups": groups,
         "world": copy.deepcopy(inventory.get("world", {"mode": "evidence_only", "strength": 0., "rotation_z": 0.})),
         "history": [{"revision": 0, "round": -1, "event": "initialize", "evidence": "prepare/inventory/output.txt"}]}
    return validate(s)


def apply_patch(state, patch, round_index):
    """Entire candidate commits only after validation; rejected candidates leave no changes."""
    if patch.get("base_revision") != state["revision"]:
        raise ValueError("Stale light-state revision")
    if set(patch) - {"base_revision", "upsert_lights", "retire_lights", "world", "reason", "upsert_illumination_groups"}:
        raise ValueError("Unknown state patch keys")
    if not patch.get("reason"):
        raise ValueError("A state change needs an evidence-based reason")
    result = copy.deepcopy(state)
    seen = set()
    for item in patch.get("upsert_lights", []):
        lid = item["id"]
        if lid in seen:
            raise ValueError("Duplicate upsert")
        seen.add(lid)
        if set(item) - LIGHT_FIELDS - {"id"}:
            raise ValueError("Unsupported light fields")
        existing = result["lights"].get(lid, {})
        if "source_type" not in existing and "source_type" not in item:
            raise ValueError(f"{lid}: declare source_type when creating a light or updating a legacy light")
        previous_source = source_type(existing)
        existing.update(copy.deepcopy(item))
        if previous_source == "direct_sun" and source_type(existing) != "direct_sun":
            raise ValueError(f"{lid}: cannot relabel a direct_sun to bypass its SUN representation; "
                             "retire this identity if evidence disproves the sun hypothesis")
        require_new_light_links(existing)
        result["lights"][lid] = existing
    group_ids = set()
    for item in patch.get("upsert_illumination_groups", []):
        gid = item["id"]
        check_id(gid)
        if gid in group_ids or set(item) - GROUP_FIELDS:
            raise ValueError("Duplicate illumination group or unsupported group fields")
        group_ids.add(gid)
        existing_group = result.setdefault("illumination_groups", {}).get(gid, {})
        existing_group.update(copy.deepcopy(item))
        result["illumination_groups"][gid] = existing_group
    for lid in patch.get("retire_lights", []):
        if lid in seen:
            raise ValueError("Cannot upsert and retire the same light")
        result["lights"][lid]["status"] = "retired"
    if "world" in patch:
        if set(patch["world"]) - {"mode", "strength", "rotation_z"}:
            raise ValueError("World file/orientation basis is immutable evidence")
        result["world"].update(patch["world"])
    result["revision"] += 1
    result["history"].append({"revision": result["revision"], "round": round_index,
                              "event": "patch", "patch": copy.deepcopy(patch)})
    return validate(result)
