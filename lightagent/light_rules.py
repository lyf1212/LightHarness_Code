"""Pure-Python physical-source rules shared by the host and standalone exports."""
import math
import re

SOURCE_TYPES = {"direct_sun", "sky_diffuse", "artificial", "unknown"}
SUN_ANGLE = .00918  # radians, initial solar angular diameter (~0.526 degrees)
GROUP_FIELDS = {"id", "type", "label", "sun_light_id", "gain", "enabled"}


def require_new_light_links(light):
    """New/edited sky AREA proxies must explicitly state their editing association."""
    if source_type(light) == "sky_diffuse" and any(e.get("kind") == "AREA" for e in light.get("elements", [])):
        if not light.get("illumination_group_id") or not light.get("approximation"):
            raise ValueError(f"{light['id']}: sky_diffuse AREA requires illumination_group_id and "
                             "approximation={kind: sky_diffuse_area, via_objects: [...], note: ...}")


def group_gain(state, light):
    gid = light.get("illumination_group_id")
    if gid is None:
        return 1.
    group = state["illumination_groups"][gid]
    return float(group["gain"]) if group["enabled"] else 0.


def effective_energy(state, light, element):
    if light["status"] != "active" or not element.get("enabled", True):
        return 0.
    return float(element.get("energy", 0.))*group_gain(state, light)


def illumination_group_metadata(state):
    return {gid: {**group, "members": [lid for lid, light in state["lights"].items()
                  if light.get("illumination_group_id") == gid], "world_included": False}
            for gid, group in state.get("illumination_groups", {}).items()}


def validate_illumination_groups(state):
    groups = state.get("illumination_groups", {})
    if not isinstance(groups, dict):
        raise ValueError("illumination_groups must be a dictionary keyed by stable group ID")
    for gid, group in groups.items():
        if not isinstance(gid, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,62}", gid):
            raise ValueError("Invalid illumination group ID")
        if not isinstance(group, dict) or set(group) - GROUP_FIELDS:
            raise ValueError(f"{gid}: unknown illumination group fields")
        if group.get("id") != gid or group.get("type") != "daylight":
            raise ValueError(f"{gid}: group requires matching id and type=daylight")
        gain = group.get("gain")
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or gain < 0:
            raise ValueError(f"{gid}: gain must be finite and nonnegative")
        if not isinstance(group.get("enabled"), bool):
            raise ValueError(f"{gid}: enabled must be boolean")
        if not isinstance(group.get("label"), str) or not group["label"].strip():
            raise ValueError(f"{gid}: label is required")
        if "sun_light_id" not in group:
            raise ValueError(f"{gid}: declare sun_light_id or null when direct sun is unobserved")
        sun_id = group["sun_light_id"]
        if sun_id is not None:
            if not isinstance(sun_id, str) or sun_id not in state["lights"]:
                raise ValueError(f"{gid}: sun_light_id references an unknown light")
            sun = state["lights"][sun_id]
            if source_type(sun) != "direct_sun" or sun.get("illumination_group_id") != gid:
                raise ValueError(f"{gid}: associated sun must be direct_sun and belong to this same group")
    for lid, light in state["lights"].items():
        gid = light.get("illumination_group_id")
        if gid is not None:
            if not isinstance(gid, str) or gid not in groups:
                raise ValueError(f"{lid}: unknown illumination_group_id")
            source = source_type(light)
            if source not in {"direct_sun", "sky_diffuse"}:
                raise ValueError(f"{lid}: daylight groups only contain direct_sun and sky_diffuse")
            if source == "direct_sun" and groups[gid]["sun_light_id"] != lid:
                raise ValueError(f"{lid}: must be the daylight group's declared sun_light_id")
        approximation = light.get("approximation")
        if approximation is not None:
            if not isinstance(approximation, dict) or set(approximation) != {"kind", "via_objects", "note"}:
                raise ValueError(f"{lid}: approximation requires kind, via_objects and note")
            if (source_type(light) != "sky_diffuse" or approximation["kind"] != "sky_diffuse_area"
                    or gid is None or not light.get("elements")
                    or any(e.get("kind") != "AREA" for e in light["elements"])):
                raise ValueError(f"{lid}: sky_diffuse_area approximation requires grouped sky_diffuse AREA elements only")
            objects = approximation["via_objects"]
            if (not isinstance(objects, list) or any(not isinstance(oid, str) for oid in objects)
                    or len(set(objects)) != len(objects) or any(oid not in state["objects"] for oid in objects)):
                raise ValueError(f"{lid}: via_objects must reference distinct existing object IDs")
            if not isinstance(approximation["note"], str) or not approximation["note"].strip():
                raise ValueError(f"{lid}: explain the diffuse approximation; it is not a physical emitting window")


def source_type(light):
    # Read historical SUN records without relabeling their saved JSON. Legacy
    # AREA/mesh emitters cannot be physically identified from their shape alone.
    return light.get("source_type", "direct_sun" if any(
        e.get("kind") == "SUN" for e in light.get("elements", [])) else "unknown")


def validate_light_sources(state):
    validate_illumination_groups(state)
    emitting_suns = []
    for lid, light in state["lights"].items():
        source = source_type(light)
        if source not in SOURCE_TYPES:
            raise ValueError(f"{lid}: source_type must be direct_sun/sky_diffuse/artificial/unknown")
        elements = light.get("elements", [])
        if any(e.get("kind") == "SUN" for e in elements) and source != "direct_sun":
            raise ValueError(f"{lid}: a SUN element must be classified as direct_sun")
        if source != "direct_sun":
            continue
        if len(elements) != 1 or elements[0].get("kind") != "SUN":
            raise ValueError(f"{lid}: direct_sun requires exactly one SUN element; "
                             "AREA/POINT/SPOT/EMISSIVE_MESH/GEOMETRY cannot substitute for the sun. "
                             "Keep the light ID and replace its elements with one SUN; "
                             "model diffuse sky separately.")
        sun = elements[0]
        forbidden = set(sun) & {"size", "radius", "spot_size", "spot_blend", "object_name"}
        if forbidden:
            raise ValueError(f"{lid}: SUN uses direction/energy/color/angle, not {sorted(forbidden)}")
        angle = sun.get("angle", SUN_ANGLE)
        if not isinstance(angle, (int, float)) or not math.isfinite(angle) or not 0 <= angle <= math.pi:
            raise ValueError(f"{lid}: SUN angle must be finite radians in [0, pi]")
        if effective_energy(state, light, sun) > 0:
            emitting_suns.append(lid)
            world = state["world"]
            if (world["mode"] == "world" and world["strength"] > 0
                    and light.get("environment_relation") == "represented_in_hdri"):
                raise ValueError(f"{lid}: explicit SUN duplicates the same sun declared represented_in_hdri; "
                                 "use evidence_only World or disable the explicit duplicate")
    if len(emitting_suns) > 1:
        raise ValueError(f"One physical sun may have only one emitting SUN, not {emitting_suns}; "
                         "multiple windows share the same sun")
