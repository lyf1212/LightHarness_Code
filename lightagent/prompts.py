SUN_GUIDANCE = '''
PHYSICAL SUN RULE — applies to initialization, every update, and diagnosis:
Every new light must declare source_type: direct_sun / sky_diffuse / artificial / unknown.
When updating a legacy light without source_type, classify it explicitly from evidence first.
Direct solar illumination MUST be source_type=direct_sun with exactly one kind=SUN element.
Never represent the sun with AREA/POINT/SPOT, an EMISSIVE_MESH, a glowing window/panel, or a GEOMETRY element.
Retain its stable light ID across updates; do not relabel it artificial/unknown to bypass the rule.
If a sun hypothesis is disproven, retire that identity with an evidence-based reason.
The scene has one physical sun; multiple windows share that same SUN, not one sun per window.
A SUN uses direction (emitted rays TOWARD the scene, not the vector pointing at the sun), energy
(Blender SUN Strength/irradiance), linear color, and angle (angular diameter in RADIANS).
Start angle near 0.00918 rad (~0.526 degrees) and justify changes using shadow evidence; do not
inflate it to stand in for diffuse sky. Position is optional and ignored by the host for SUN.
Do not provide size, radius, spot properties or object_name on a SUN. The host creates the Blender light.
Windows are apertures/transmission/occlusion geometry; do not make them emissive to manufacture sun patches.
Sun-patch shape comes from SUN direction and actual openings/blockers. Fix geometry before changing power.
Separate diffuse sky from direct sun. World/HDRI or an explicitly justified sky_diffuse AREA approximation
may model diffuse incident sky, but it must not substitute for the directional sun. Real artificial
panels/lamps may still use AREA or EMISSIVE_MESH with their own identities and supporting evidence.
Do not activate sunlight merely because a window is visible: use hypothesis/inactive if evidence is weak.
Prevent duplicate solar energy in HDRI + SUN; a positive active World already declared to contain the same
sun (environment_relation=represented_in_hdri) cannot also have an emitting explicit SUN. Prefer
evidence_only World for explicit-sun modeling unless the environment's solar duplication is resolved.
Review source classification against the reference, not just JSON labels: flag a purported artificial
panel or sky source that is actually being used as the direct sun. Identify violations by light_id and
explain the required SUN/geometry correction in lighting_changes and next_priorities.

DAYLIGHT EDITING ASSOCIATION — applies to initialization, every update, and diagnosis:
Link direct sun and its diffuse sky proxies using one stable illumination_group_id, e.g. daylight_01.
Inventory illumination_groups is an array of records:
{"id":"daylight_01","type":"daylight","label":"outdoor daylight",
 "sun_light_id":"sun_01","gain":1.0,"enabled":true}.
Persistent state stores these records keyed by ID. Set sun_light_id=null if no direct sun is supported;
do not invent an emitting SUN just to fill the association. Otherwise sun_light_id must reference a
direct_sun light with the SAME illumination_group_id. Multiple windows share this same group and SUN.
Every new/edited sky_diffuse AREA light MUST declare illumination_group_id and
approximation={"kind":"sky_diffuse_area","via_objects":["obj_window_01"],
"note":"Approximate diffuse sky entering the opening; the window itself does not emit"}.
via_objects references existing stable opening IDs; use [] and explain uncertainty if no opening is identified.
These proxies are the diffuse component of a daylight system, NEVER substitutes for direct solar rays.
For a whole-daylight brightness edit, submit upsert_illumination_groups=[{"id":"daylight_01","gain":0.5}].
The host multiplies each member's BASE element energy by gain, retaining its own units and base value.
enabled=false switches off the entire group. Light status/element enabled still independently gate emission.
For only direct-sun edits, change that SUN's energy/direction/color/angle. Do not rotate/move the window AREA
with the SUN: sky proxies remain at their apertures. Disabling SUN alone does not disable diffuse sky.
This is an explicit editing association, not a physical atmosphere simulation or an automatic sun-to-sky
intensity/color relation. World/HDRI is NOT included in the group and must be edited separately.
Never classify an artificial emitting panel as a sky proxy merely to join a daylight group.
Review the state association and scores.metrics.illumination_group_contributions; for group-level advice,
identify illumination_group_id in the change text alongside the affected stable light_id(s).
'''

LIGHT_FORMAT = '''Light JSON:
{"id":"light_001","label":"pendant","source_type":"artificial","visibility":"partial","status":"hypothesis",
 "confidence":0.6,"evidence":[{"source":"reference","observation":"visible shade, emission uncertain"}],
 "occluded_by":[],"image_bbox":[0.1,0.1,0.3,0.3],"notes":"identity persists across rounds",
 "environment_relation":"possibly_in_hdri",
 "elements":[{"id":"emitter","kind":"AREA","position":[0,3,1],"direction":[0,0,-1],
 "energy":30,"color":[1,0.8,0.6],"size":[0.3,0.3],"enabled":true}]}
Visibility: visible/partial/occluded/off_frame. Status: active/inactive/hypothesis/retired.
Only active lights emit. Do not label a visible lamp as on without evidence.
Elements can be AREA, POINT, SPOT, SUN, EMISSIVE_MESH or GEOMETRY. All IDs must be stable ASCII identifiers.
Each element has a unique id within its light. AREA size=[width,height] in scene meters;
POINT/SPOT radius in meters; SUN angle in radians; SPOT spot_size/spot_blend.
energy: Blender watts for POINT/AREA/SPOT, irradiance for SUN, emission strength for EMISSIVE_MESH.
color: linear RGB [0,1]. direction: emitted ray direction in world coordinates.
GEOMETRY/EMISSIVE_MESH bind object_name of an object your geometry code creates.
GEOMETRY is a non-emitting fixture/shade. EMISSIVE_MESH needs energy/color and object_name.
Use multiple elements to describe one physical light, but explain how energy is allocated to avoid double counting.
environment_relation = independent/possibly_in_hdri/represented_in_hdri.
world={"mode":"evidence_only" or "world", "strength":0.0,"rotation_z":0.0}.
World rotation_z is physical environment rotation in radians about +Z, positive counterclockwise.
HDRI is an uncertain camera-local illumination probe, not automatically a distant environment.
Prefer evidence_only initially. Explicit emitters and the same bright HDRI source must not count twice.
''' + SUN_GUIDANCE

SURFACE_GUIDANCE = '''
SURFACE GEOMETRY CONTRACT (surface_geometry JSON):
Reference normals are surface-level plane parameters derived from existing depth, not normal-color images.
Each stable object ID has a reliable flag, mask coverage, fit quality and either a plane or a reason.
Use ONLY reliable=true planes as orientation/position anchors. A missing plane is missing evidence,
not an absent object. Do not flatten chairs, curtains, lamps or other non-planar objects into planes.
point_world is a point ON the visible plane, not an object's volume center. Build the surface THROUGH
that point and PERPENDICULAR to normal_world. Determine extent, openings and boundaries from photo/masks.
measured_normal_world is the raw fit; normal_world may be level/plumb corrected relative to estimated up.
All points/directions are in the EXISTING Blender world frame given by camera, in its monocular scale gauge.
The world +Z is image-up, NOT automatically gravity. When gravity.available=true, use gravity.up_world
for level/plumb reasoning. No camera, light-state or HDRI rotation is implied by this estimated up.
For walls, use horizontal_basis_world as local width X, normal_world as thin local Y, and estimated up
as local height Z; the right-handed basis fixes orientation, not which half-line the wall extends along.
When up is unavailable, follow reliable measured planes without imposing a horizontal/vertical snap.
Plane-normal sign is ambiguous and is not an illumination direction. Fit quality is not a probability.
Masks smaller than 2% and plane fits below 60% inliers are not orientation anchors; use visual context.
In review, compare scores.metrics.surface_normal_degrees.per_surface by stable ID: measured reference
and rendered normals, unsigned angle, point-to-plane distance and missing surfaces. A low mean over
matched surfaces does not excuse unmatched ones. Distance is in scene units without fitted rescaling.
Separate position/occlusion/geometry errors from lighting before changing a light's power.
'''

INVENTORY = '''Analyze this reference for a physically editable 3D reconstruction. Return ONLY JSON:
{"objects":[{"id":"obj_001","label":"floor","prompt":"floor","role":"floor",
"bbox":[0,0.5,1,1],"confidence":0.9}],"lights":[...],"illumination_groups":[],"world":{...},"analysis":"..."}.
bbox is normalized xyxy in canonical image coordinates, top-left origin. Give separate IDs and localized
boxes for repeated instances so SAM3 can associate masks. Include walls, floor, ceiling, window openings,
lampshades, glass and major blockers/reflection surfaces, not only furniture.
Use role floor/ground/wall/ceiling/tabletop/countertop/support_surface for planar root surfaces only;
do not assign these roles to curtains, chairs, whole multi-face furniture or lamps.
Distinguish a window aperture from an emitting source outside it.
Occluded/off-frame lights are hypotheses with evidence, never measurements.
Positions are provisional: +X image-right, +Y camera-forward, +Z image-up, camera at origin, scene units meters
with an arbitrary monocular gauge. Depth evidence follows this semantic inventory. State uncertainties.
''' + LIGHT_FORMAT

BUILD = '''You are the scene-building agent. Return ONLY JSON with keys rationale, state_patch, scene_code.
scene_code is a complete Python module defining build_scene(ctx). Use bpy, math, mathutils, random only.
The host calls this function, then creates the calibrated camera, World, explicit lights, passes and render.
Do not render, save files, load assets, create lights/cameras/worlds, change render settings, use image planes,
project the reference as a texture, access files/network/processes, or bake shadows into material base colors.
Create real editable 3D geometry and non-emitting BSDF materials. All mesh/curve/surface objects must call
ctx.register(obj, stable_object_id). Many components may share one semantic object ID; masks merge them.
ctx.state, ctx.camera and ctx.priors are available. All registered IDs must exist in state.objects.
ctx.material(name, color, roughness=0.7, metallic=0.0) creates a Principled material with LINEAR base color.
Use actual holes for windows and shadow-casting lamp shades/blockers. Geometry bound by light elements must
have exactly the requested object_name. Host owns emission. Keep geometry below 500k vertices.
Use Blender 4.5 socket names (Base Color, Roughness, Metallic, Transmission Weight).
Use the camera contract and per-object depth-derived 3D priors to match silhouette and occlusion.
No camera optimization in code: the calibrated camera and global exposure are fixed by the host.
state_patch={"base_revision":current_revision,"reason":"evidence-based changes",
"upsert_lights":[complete new lights or partial existing records with id],"retire_lights":[],
"upsert_illumination_groups":[complete new groups or partial existing records with id],"world":{...}}.
Never delete/reassign an identity. Retire only with reason. For existing light elements, send the complete
elements list when changing it. Propose small interpretable lighting changes and account for geometry errors
before changing light power. Geometric and intrinsic estimates are uncertain soft evidence, not ground truth.
Read previous review, scores, current state, previous code and history before revising.
Read object_memory for every stable object ID: brief difference assessments, selective advice and history_feedback.
Prioritize important evidence-supported corrections; inspecting every object does not require changing
every object. Preserve successful edits and avoid repeating changes that regressed or remain unverified.
Explain selected corrections and deferred/uncertain advice by object ID in rationale. Memory is diagnostic
context only, never a state_patch field. Legacy no_structured_record entries are not evidence of no errors;
consult the original history without inventing missing object-level assessments.
''' + LIGHT_FORMAT + SURFACE_GUIDANCE

REVIEW = '''You are the diagnostic agent reviewing a completed reconstruction round. Return ONLY JSON:
{"summary":"...","geometry_changes":[...],"material_changes":[...],"lighting_changes":[
{"light_id":"stable ID","change":"...","evidence":"...","confidence":0.7}],
"next_priorities":[...],"uncertainties":[...],
"object_reviews":{
  "changed_object_id":{"difference_level":"significant","observation":"Noticeable silhouette mismatch",
    "region":{"difference":"...","suggestion":"...","evidence":"...","confidence":0.7}},
  "matching_object_id":{"difference_level":"minor","observation":"Close enough; no edit needed"},
  "unclear_object_id":{"difference_level":"uncertain","observation":"Occluded; cannot reliably compare"}},
"history_feedback":[{"object_id":"stable_object_id","source_round":0,"aspect":"region",
  "implementation":"implemented","outcome":"improved","evidence":"..."}]}.
For each ID in state.objects, first make a brief overall judgment of how different it looks.
object_reviews contains one assessment per ID: difference_level=significant/minor/uncertain and a short
observation. significant means a noticeable difference worth correcting, not necessarily a major error;
minor means close enough without a useful correction; uncertain means evidence does not support a judgment.
Only significant objects need modification diagnoses. Include only the relevant region, material and/or
color keys, at least one; NEVER fill all three by rote. minor/uncertain objects need no diagnostic keys,
no suggestions and no per-aspect explanations. Choose significance from the visual evidence, not a fixed
numeric threshold. Do not manufacture issues to fill a checklist.
Inspect the reference versus current render for each object using the supplied whole images, overview,
masks and scores; do not skip small, occluded, unrendered or mask-unavailable objects.
region: compare screen position, extent, silhouette, occlusion and missing geometry; propose concrete edits.
material: compare roughness, metallic appearance, transmission and reflections, stating uncertainty.
color: compare hue, saturation and brightness; use albedo/shading and light contributions to distinguish
base-color errors from lighting/geometry errors. Never bake shadows or colored illumination into base color.
Each included diagnosis requires nonempty difference, suggestion and evidence text, and a finite confidence
in [0,1]. Missing masks do not mean the object matches; use uncertain when needed, and do not invent
measurements or claim nonexistent per-object material scores.
Read all object_memory and previous_history before judging changes. history_feedback must assess every
actual modification diagnosis in each object's latest structured record, even if it now looks close enough;
for older three-aspect records, only status=needs_change requires feedback. Also revisit older unresolved advice when
relevant. Reference its exact source_round, object_id and aspect=region/material/color, once per reference.
implementation=implemented/partial/not_implemented/unknown describes whether the prior advice was applied;
outcome=improved/unchanged/regressed/unknown separately describes the observed visual result. Support both
with evidence comparing previous_code/current scene_code, previous/current images and available scores.
An overall loss reduction alone cannot prove any object's improvement or that advice was implemented.
Use unknown when attribution or evidence is insufficient; not_implemented is not a synonym for unknown.
The first round has history_feedback=[]; with only legacy unstructured history use [] and explain uncertainty.
Never fabricate structured advice for legacy no_structured_record entries. Reviews are advice, not proof.
If failed_reviews is nonempty, correct the reported format/coverage errors while rechecking all objects.
Compare reference, render, masks, depth, surface plane parameters, albedo, diffuse shading and light contributions.
First separate camera/geometry/occlusion errors from material and illumination errors.
Use stable IDs. Be specific about position, size, direction, color and intensity adjustments.
Unavailable scores are missing evidence, never zero error. Surface planes are derived from predicted depth.
Depth is scale ambiguous; shading has a shared fitted positive scalar; preserve colored illumination.
RGB is display-space appearance evidence; clipping/tone mapping prevent direct radiometric ground truth.
Account for off-frame sources and partial visibility. A light group contribution diagnoses influence,
not measured true per-light ground truth. Compare HDRI with explicit sources to prevent double counting.
Your review is advice only; the next builder must submit a validated state patch to change persistent state.
''' + SURFACE_GUIDANCE + SUN_GUIDANCE
