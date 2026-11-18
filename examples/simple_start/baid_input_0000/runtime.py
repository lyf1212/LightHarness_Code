"""Executed only by Blender 4.5: geometry -> authoritative state -> fixed camera -> passes."""
from __future__ import annotations
import json
import copy
import math
import random
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector

# Also works when this file is exported as final/runtime.py beside light_rules.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from light_rules import (SUN_ANGLE, source_type, validate_light_sources, group_gain,
                         effective_energy, illumination_group_metadata)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))


class Context:
    def __init__(self, state, camera, priors):
        # Builders can inspect snapshots but cannot change the authoritative patch/camera
        # by mutating ctx.state or ctx.camera while constructing geometry.
        self.state, self.camera, self.priors = copy.deepcopy(state), copy.deepcopy(camera), copy.deepcopy(priors)
        self.indices = {oid: i+1 for i, oid in enumerate(sorted(state["objects"]))}

    def register(self, obj, object_id):
        if object_id not in self.indices:
            raise ValueError(f"Unknown semantic object ID {object_id}")
        obj["stable_object_id"] = object_id
        obj.pass_index = self.indices[object_id]
        return obj

    @staticmethod
    def material(name, color, roughness=.7, metallic=0.):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (*color[:3], 1.)
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
        return m


def set_camera(scene, camera):
    K = camera["K"]
    w, h = camera["width"], camera["height"]
    data = bpy.data.cameras.new("calibrated_camera")
    cam = bpy.data.objects.new("calibrated_camera", data)
    scene.collection.objects.link(cam)
    cam.matrix_world = Matrix(camera["matrix_world"])
    data.type = "PERSP"
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36.
    data.lens = K[0][0] * 36. / w
    aspect = K[0][0] / K[1][1]
    # Blender clamps both components to >=1; preserve their ratio when fx<fy.
    scene.render.pixel_aspect_x = max(1., 1./aspect)
    scene.render.pixel_aspect_y = max(1., aspect)
    data.shift_x = (w/2 - (K[0][2]+.5)) / w
    data.shift_y = ((K[1][2]+.5) - h/2) * aspect / w
    data.clip_start = .01
    data.clip_end = 10000
    scene.camera = cam
    return cam


def validate_camera_projection(scene, camera):
    from bpy_extras.object_utils import world_to_camera_view
    bpy.context.view_layer.update()
    K = Matrix(camera["K"])
    R = Matrix(camera["cv_to_blender_world_rotation"])
    w, h = camera["width"], camera["height"]
    error = 0.
    for u, v in [(0., 0.), (w-1., h-1.), (w*.3, h*.7), (w*.5, h*.5)]:
        p = R @ (K.inverted() @ Vector((u, v, 1.))) * 4
        projected = world_to_camera_view(scene, scene.camera, p)
        error = max(error, abs(projected.x*w-.5-u), abs((1-projected.y)*h-.5-v))
    if error > .001:
        raise ValueError(f"Calibrated camera pixel projection mismatch: {error}")
    return error


def set_world(scene, state, hdr):
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    scene.world = world
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Color"].default_value = (0, 0, 0, 1)
    background.inputs["Strength"].default_value = 0
    links.new(background.outputs["Background"], output.inputs["Surface"])
    if state["world"]["mode"] == "world":
        if not hdr or not Path(hdr).exists():
            raise ValueError("World illumination requested but no calibrated HDRI exists")
        tex = nodes.new("ShaderNodeTexEnvironment")
        tex.image = bpy.data.images.load(hdr, check_existing=True)
        tex.image.colorspace_settings.name = "Linear Rec.709"
        coord = nodes.new("ShaderNodeTexCoord")
        mapping = nodes.new("ShaderNodeVectorRotate")
        mapping.rotation_type = "AXIS_ANGLE"
        mapping.inputs["Axis"].default_value = (0, 0, 1)
        # Rotate sampling rays inversely to rotate the physical environment positively.
        mapping.inputs["Angle"].default_value = -state["world"]["rotation_z"]
        links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        links.new(tex.outputs["Color"], background.inputs["Color"])
        background.inputs["Strength"].default_value = state["world"]["strength"]
    world.lightgroup = "world"
    return world


def apply_lights(scene, state):
    validate_light_sources(state)
    bindings = {}
    for lid, light in state["lights"].items():
        bindings[lid] = []
        gid = light.get("illumination_group_id")
        group = state.get("illumination_groups", {}).get(gid, {})
        gain = group_gain(state, light)
        for element in light["elements"]:
            kind = element["kind"]
            active = light["status"] == "active" and element.get("enabled", True) and gain > 0
            energy = effective_energy(state, light, element)
            if kind in {"GEOMETRY", "EMISSIVE_MESH"}:
                obj = bpy.data.objects.get(element["object_name"])
                if obj is None:
                    # Retired bindings are historical; active and hypothesized fixtures must exist.
                    if light["status"] == "retired":
                        continue
                    raise ValueError(f"Missing bound object {element['object_name']} for {lid}")
                if kind == "EMISSIVE_MESH":
                    if obj.type != "MESH":
                        raise ValueError(f"EMISSIVE_MESH must bind a mesh: {obj.name}")
                    # Geometry instances may share mesh data; isolate the emitter's slots.
                    obj.data = obj.data.copy()
                    # Preserve the fixture's BSDF, including in its switched-off state.
                    originals = list(obj.data.materials) or [None]
                    materials = []
                    for original in originals:
                        material = original.copy() if original else bpy.data.materials.new(f"{lid}_{element['id']}_emission")
                        material.use_nodes = True
                        nodes, links = material.node_tree.nodes, material.node_tree.links
                        output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None)
                        if output is None:
                            output = nodes.new("ShaderNodeOutputMaterial")
                        existing = output.inputs["Surface"].links
                        previous = existing[0].from_socket if existing else None
                        emission = nodes.new("ShaderNodeEmission")
                        emission.inputs["Color"].default_value = (*element["color"], 1)
                        emission.inputs["Strength"].default_value = energy
                        if previous:
                            add = nodes.new("ShaderNodeAddShader")
                            links.new(previous, add.inputs[0])
                            links.new(emission.outputs["Emission"], add.inputs[1])
                            links.new(add.outputs[0], output.inputs["Surface"])
                        else:
                            links.new(emission.outputs["Emission"], output.inputs["Surface"])
                        materials.append(material)
                    obj.data.materials.clear()
                    for material in materials:
                        obj.data.materials.append(material)
                    obj.lightgroup = lid
            else:
                data = bpy.data.lights.new(f"{lid}_{element['id']}", kind)
                obj = bpy.data.objects.new(data.name, data)
                scene.collection.objects.link(obj)
                obj.location = (0, 0, 0) if kind == "SUN" else element["position"]
                obj.rotation_euler = Vector(element["direction"]).to_track_quat('-Z', 'Y').to_euler()
                data.color = element["color"]
                data.energy = energy
                if kind == "AREA":
                    data.shape = "RECTANGLE"
                    data.size, data.size_y = element["size"]
                if kind in {"POINT", "SPOT"}:
                    data.shadow_soft_size = element.get("radius", .05)
                if kind == "SUN":
                    data.angle = element.get("angle", SUN_ANGLE)
                if kind == "SPOT":
                    data.spot_size = element.get("spot_size", math.pi/3)
                    data.spot_blend = element.get("spot_blend", .15)
                obj.lightgroup = lid
            obj["stable_light_id"] = lid
            obj["light_element_id"] = element["id"]
            obj["source_type"] = source_type(light)
            obj["illumination_group_id"] = gid or ""
            obj["associated_sun_light_id"] = group.get("sun_light_id") or ""
            obj["approximation_json"] = json.dumps(light.get("approximation"), ensure_ascii=False)
            obj["base_energy"] = element.get("energy", 0.)
            obj["illumination_group_gain"] = gain
            bpy.context.view_layer.update()
            bindings[lid].append({"element_id": element["id"], "object_name": obj.name, "kind": kind, "active": active,
                                  "source_type": source_type(light),
                                  "illumination_group_id": gid,
                                  "associated_sun_light_id": group.get("sun_light_id"),
                                  "approximation": light.get("approximation"),
                                  "base_energy": element.get("energy", 0.), "group_gain": gain,
                                  "direction_world": element.get("direction"),
                                  "energy": energy,
                                  "color": element.get("color"),
                                  "sun_angle": element.get("angle", SUN_ANGLE) if kind == "SUN" else None,
                                  "position_affects_illumination": kind != "SUN",
                                  "position_world": list(obj.matrix_world.translation), "dimensions": list(obj.dimensions),
                                  "matrix_world": [list(row) for row in obj.matrix_world]})
    return bindings


def audit_geometry(ctx):
    vertices = 0
    for obj in bpy.context.scene.objects:
        if obj.type in {"LIGHT", "CAMERA"}:
            raise ValueError("Generated code must not create lights or cameras")
        if obj.type in {"MESH", "CURVE", "SURFACE", "META", "FONT"}:
            if obj.get("stable_object_id") not in ctx.indices:
                raise ValueError(f"Unregistered object {obj.name}")
            # Reapply authoritative pass indices after generated code has finished.
            obj.pass_index = ctx.indices[obj["stable_object_id"]]
            if obj.type == "MESH":
                vertices += len(obj.data.vertices)
            for slot in obj.material_slots:
                m = slot.material
                if not m or not m.use_nodes:
                    continue
                for node in m.node_tree.nodes:
                    if node.type in {"EMISSION", "TEX_IMAGE", "GROUP"}:
                        raise ValueError("Host owns emission; generated image textures/node groups are not supported")
                    if node.type == "BSDF_PRINCIPLED":
                        strength = node.inputs.get("Emission Strength")
                        if strength and (strength.is_linked or strength.default_value > 0):
                            raise ValueError("Host owns emission; use an EMISSIVE_MESH state element")
    if vertices > 500000:
        raise ValueError("Geometry vertex budget exceeded")


def configure_passes(scene, state, out):
    view = bpy.context.view_layer
    view.name = "ViewLayer"
    view.cycles.use_denoising = scene.cycles.use_denoising
    view.cycles.denoising_store_passes = scene.cycles.use_denoising
    for attr in ("use_pass_z", "use_pass_normal", "use_pass_object_index", "use_pass_diffuse_color",
                 "use_pass_diffuse_direct", "use_pass_diffuse_indirect", "use_pass_emit"):
        setattr(view, attr, True)
    for name in ["world", *state["lights"]]:
        view.lightgroups.add(name=name)
    view.update_render_passes()
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    layers = tree.nodes.new("CompositorNodeRLayers")
    layers.layer = view.name
    # Light groups remain raw: compare their sum to the unfiltered Combined pass.
    specs = {"rgb_linear": "Noisy Image" if scene.cycles.use_denoising else "Image",
             "depth_native": "Depth", "normals_world": "Normal",
             "object_index": "IndexOB", "albedo": "DiffCol", "diffuse_direct": "DiffDir",
             "diffuse_indirect": "DiffInd", "emission": "Emit"}
    available = [socket.name for socket in layers.outputs]
    for name in ["world", *state["lights"]]:
        socket = f"Combined_{name}"
        if socket not in available:
            raise ValueError(f"Missing Cycles light-group pass {socket}; available={available}")
        specs["light_" + name] = socket
    for name, socket in specs.items():
        node = tree.nodes.new("CompositorNodeOutputFile")
        node.base_path = str(out)
        node.format.file_format = 'OPEN_EXR'
        node.format.color_mode = 'RGB'
        node.format.color_depth = '32'
        node.format.exr_codec = 'ZIP'
        node.file_slots[0].path = name + "_"
        tree.links.new(layers.outputs[socket], node.inputs[0])
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(layers.outputs["Image"], composite.inputs[0])
    return specs


def configure_cycles_gpu(prefs, backend):
    """Select one backend; Cycles lists CUDA and OPTIX entries for the same GPU."""
    prefs.compute_device_type = backend
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == backend
    selected = [{"id": device.id, "name": device.name, "type": device.type}
                for device in prefs.devices if device.use]
    if not selected:
        raise RuntimeError(f"Requested Cycles GPU backend unavailable: {backend}")
    return selected


def main(job):
    out = Path(job["output"])
    out.mkdir(parents=True, exist_ok=True)
    state, camera, priors = read(job["state"]), read(job["camera"]), read(job["priors"])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    ctx = Context(state, camera, priors)
    random.seed(job["seed"])
    scope = {"bpy": bpy}
    exec(compile(Path(job["code"]).read_text(), job["code"], 'exec'), scope)
    scope["build_scene"](ctx)
    ctx.indices = {oid: i+1 for i, oid in enumerate(sorted(state["objects"]))}
    audit_geometry(ctx)
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = job.get("samples", 512)
    scene.cycles.use_denoising = job.get("use_denoising", False)
    scene.cycles.denoiser = job.get("denoiser", "OPENIMAGEDENOISE")
    scene.cycles.seed = job["seed"]
    scene.cycles.use_adaptive_sampling = job.get("use_adaptive_sampling", False)
    scene.cycles.adaptive_threshold = job.get("adaptive_threshold", .01)
    scene.cycles.adaptive_min_samples = job.get("adaptive_min_samples", 0)
    scene.cycles.device = "CPU"
    selected_devices = []
    if job.get("device", "CPU") != "CPU":
        prefs = bpy.context.preferences.addons['cycles'].preferences
        selected_devices = configure_cycles_gpu(prefs, job["device"])
        scene.cycles.device = "GPU"
    scene.render.resolution_x, scene.render.resolution_y = camera["width"], camera["height"]
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = job.get("exposure", 0.)
    scene.view_settings.gamma = 1.
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.
    scene.frame_set(1)
    set_camera(scene, camera)
    camera_projection_error = validate_camera_projection(scene, camera)
    set_world(scene, state, job.get("hdr"))
    bindings = apply_lights(scene, state)
    specs = configure_passes(scene, state, out)
    bpy.ops.file.pack_all()
    scene.render.filepath = str(out / "rgb.png")
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "scene.blend"))
    bpy.ops.render.render(write_still=True)
    objects = {o.name: {"object_id": o.get("stable_object_id"), "pass_index": o.pass_index,
                        "light_id": o.get("stable_light_id")} for o in scene.objects}
    write(out / "metadata.json", {"blender_version": bpy.app.version_string, "objects": objects,
          "render_device": scene.cycles.device, "selected_devices": selected_devices,
          "object_indices": ctx.indices, "light_bindings": bindings,
          "illumination_groups": illumination_group_metadata(state),
          "passes": {name: name+"_0001.exr" for name in specs}, "camera": camera,
          "state_revision": state["revision"], "color_management": {"view": "Standard", "exposure": scene.view_settings.exposure},
          "camera_projection_error_pixels": camera_projection_error,
          "depth_native": job.get("depth_type", "camera_z"), "normal_native": "world", "samples": scene.cycles.samples,
          "denoising": scene.cycles.use_denoising, "use_denoising": scene.cycles.use_denoising,
          "denoiser": scene.cycles.denoiser, "view_layer_denoising": bpy.context.view_layer.cycles.use_denoising,
          "use_adaptive_sampling": scene.cycles.use_adaptive_sampling,
          "adaptive_threshold": scene.cycles.adaptive_threshold,
          "adaptive_min_samples": scene.cycles.adaptive_min_samples,
          "rgb_output": "denoised" if scene.cycles.use_denoising else "raw",
          "rgb_linear_source": specs["rgb_linear"], "seed": scene.cycles.seed})


if __name__ == "__main__":
    main(read(sys.argv[sys.argv.index("--")+1]))
