import bpy, sys, json, math
from pathlib import Path
import numpy as np
from mathutils import Matrix, Vector

args = sys.argv[sys.argv.index("--") + 1 :]
root = Path(args[0])
render_output = Path(args[1])
settings = json.loads(Path(args[2]).read_text())
info = json.loads((root / "scene.json").read_text())
z = np.load(root / "assets.npz")
kind = info["kind"]
N = info["frames"]
start = info["start"]
w, h = info["width"], info["height"]
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.samples = settings["samples"]
sc.cycles.use_denoising = True
sc.cycles.device = settings["device"].upper()
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "CUDA" if settings["device"] == "gpu" else "NONE"
prefs.get_devices()
for d in prefs.devices:
    d.use = d.type == "CUDA" if settings["device"] == "gpu" else d.type == "CPU"
print("RENDER_DEVICES", [(d.name, d.type, d.use) for d in prefs.devices], flush=True)
sc.render.resolution_x = w * settings["scale"]
sc.render.resolution_y = h * settings["scale"]
sc.render.resolution_percentage = 100
sc.render.image_settings.file_format = "PNG"
sc.render.film_transparent = False
sc.render.fps = round(info["fps"])
sc.view_settings.view_transform = "Standard"
sc.view_settings.look = "None"
sc.view_settings.exposure = 0
sc.view_settings.gamma = 1
sc.render.threads_mode = "FIXED"
sc.render.threads = 4
sc.render.use_persistent_data = True
sc.world = bpy.data.worlds.new("Studio")
sc.world.use_nodes = True
sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.5, 0.5, 0.5, 1)
sc.world.node_tree.nodes["Background"].inputs[1].default_value = 0.45


def linear(c):
    c = np.array(c[:3], float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).tolist()


def material(name, color, metal=0, rough=0.42, alpha=1, emission=False):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes.get("Principled BSDF")
    p.inputs["Base Color"].default_value = (*linear(color), 1)
    p.inputs["Metallic"].default_value = metal
    p.inputs["Roughness"].default_value = rough
    if emission:
        p.inputs["Emission Color"].default_value = (*linear(color), 1)
        p.inputs["Emission Strength"].default_value = 1
        p.inputs["Base Color"].default_value = (0, 0, 0, 1)
    if alpha < 1:
        out = nt.nodes.get("Material Output")
        transparent = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        mix.inputs[0].default_value = alpha
        nt.links.new(transparent.outputs[0], mix.inputs[1])
        nt.links.new(p.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return m


def meshob(name, v, f, mat):
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.asarray(v).tolist(), [], np.asarray(f).tolist())
    me.update()
    ob = bpy.data.objects.new(name, me)
    sc.collection.objects.link(ob)
    if mat:
        me.materials.append(mat)
    return ob


def curve(name, points, color, radius):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.resolution_u = 1
    cu.bevel_depth = radius
    cu.bevel_resolution = 3
    s = cu.splines.new("POLY")
    s.points.add(len(points) - 1)
    for p, co in zip(s.points, points):
        p.co = (*co, 1)
    ob = bpy.data.objects.new(name, cu)
    sc.collection.objects.link(ob)
    cu.materials.append(material(name, color, rough=0.4))
    return ob


def pointcurve_update(ob, points):
    s = ob.data.splines[0]
    for p, co in zip(s.points, points):
        p.co = (*co, 1)


def light(name, loc, energy, size, target):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    ob = bpy.data.objects.new(name, data)
    sc.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = (
        (Vector(target) - ob.location).to_track_quat("-Z", "Y").to_euler()
    )


camdat = bpy.data.cameras.new(
    "Estimated camera"
    if kind in ["cloth", "softbody"]
    else "Fitted pixel-space orthographic camera"
)
cam = bpy.data.objects.new("Camera", camdat)
sc.collection.objects.link(cam)
sc.camera = cam
camdat.clip_start = 0.001
camdat.clip_end = 100
if kind in ["cloth", "softbody"]:
    K = z["K"][0]
    camdat.type = "PERSP"
    camdat.sensor_fit = "HORIZONTAL"
    camdat.sensor_width = 36
    camdat.lens = float(K[0, 0]) * 36 / w
    sc.render.pixel_aspect_x = 1
    sc.render.pixel_aspect_y = float(K[0, 0] / K[1, 1])
    camdat.shift_x = float((w / 2 - K[0, 2]) / w)
    camdat.shift_y = float((K[1, 2] - h / 2) / w * sc.render.pixel_aspect_y)
    cam.matrix_world = Matrix(z["camera_to_world"][0]) @ Matrix.Diagonal((1, -1, -1, 1))
    P = z["camera_to_world"][0]
    cp = P[:3, 3]
    forward = P[:3, 2]
    target = cp + forward * 0.9
    light("Large softbox", cp + P[:3, 0] * -0.4 + P[:3, 1] * -0.4, 22, 1.2, target)
    light("Fill", cp + P[:3, 0] * 0.4, 18, 1.1, target)
else:
    aspect = info["aspect"]
    camdat.type = "ORTHO"
    camdat.ortho_scale = aspect
    cam.location = (0, 0, 4)
    light("Softbox", (-1, 1, 2), 70, 3, (0, 0, 0))
    light("Fill", (1, -0.2, 2), 25, 2, (0, 0, 0))


def imageplane(path, vertices):
    ob = meshob("Temporal mean background", vertices, [[0, 1, 2, 3]], None)
    uv = ob.data.uv_layers.new()
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for poly in ob.data.polygons:
        for li in poly.loop_indices:
            uv.data[li].uv = coords[ob.data.loops[li].vertex_index]
    mat = material("Background pixels", [1, 1, 1], emission=True)
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(path))
    p = mat.node_tree.nodes.get("Principled BSDF")
    mat.node_tree.links.new(tex.outputs["Color"], p.inputs["Emission Color"])
    ob.data.materials.append(mat)
    return ob


if kind == "cloth":
    pts = z["background_points"]
    colors = z["background_colors"][:, :3].astype(float)
    colors = colors / 255 if colors.max() > 1 else colors
    P = z["camera_to_world"][0]
    W = np.linalg.inv(P)
    depth = (pts @ W[:3, :3].T + W[:3, 3])[:, 2]
    radius = depth / float(z["K"][0, 0, 0]) * 0.72
    dx = P[:3, 0][None] * radius[:, None]
    dy = P[:3, 1][None] * radius[:, None]
    verts = np.stack(
        [pts - dx - dy, pts + dx - dy, pts + dx + dy, pts - dx + dy], 1
    ).reshape(-1, 3)
    faces = np.arange(len(verts)).reshape(-1, 4)
    ob = meshob("Retained static dense point cloud", verts, faces, None)
    attr = ob.data.color_attributes.new(name="RGB", type="FLOAT_COLOR", domain="POINT")
    c = np.array([linear(x) for x in colors])
    rgba = np.concatenate(
        [np.repeat(c, 4, axis=0), np.ones((len(verts), 1))], 1
    ).ravel()
    attr.data.foreach_set("color", rgba)
    m = material("Point cloud RGB", [1, 1, 1], emission=True)
    a = m.node_tree.nodes.new("ShaderNodeVertexColor")
    a.layer_name = "RGB"
    m.node_tree.links.new(
        a.outputs["Color"],
        m.node_tree.nodes["Principled BSDF"].inputs["Emission Color"],
    )
    ob.data.materials.append(m)
elif kind == "softbody":
    P = z["camera_to_world"][0]
    K = z["K"][0]
    dep = (
        float(
            np.max(
                (
                    z["background_points"] @ np.linalg.inv(P)[:3, :3].T
                    + np.linalg.inv(P)[:3, 3]
                )[:, 2]
            )
        )
        + 0.05
    )
    uv = np.array([[0, h], [w, h], [w, 0], [0, 0]])
    xyz = np.column_stack(
        [
            (uv[:, 0] - K[0, 2]) * dep / K[0, 0],
            (uv[:, 1] - K[1, 2]) * dep / K[1, 1],
            np.full(4, dep),
        ]
    )
    imageplane(root / "background.png", xyz @ P[:3, :3].T + P[:3, 3])
elif kind == "fluid":
    imageplane(
        root / "background.png",
        [
            [-aspect / 2, -0.5, -0.015],
            [aspect / 2, -0.5, -0.015],
            [aspect / 2, 0.5, -0.015],
            [-aspect / 2, 0.5, -0.015],
        ],
    )
else:
    meshob(
        "Mean background",
        [
            [-aspect, -1, -0.04],
            [aspect, -1, -0.04],
            [aspect, 1, -0.04],
            [-aspect, 1, -0.04],
        ],
        [[0, 1, 2, 3]],
        material("Background RGB mean", info["background_mean"], emission=True),
    )

contact_asset = (
    np.load(render_output / "cloth-display.npz")
    if (render_output / "cloth-display.npz").exists()
    else z
)
rendered = {}
for item in info["objects"]:
    name = item["name"]
    key = item["key"]
    typ = item["kind"]
    mat = material(
        name + " RGB",
        item["color"],
        metal=item.get("metallic", 0),
        rough=(
            (0.38 if item.get("metallic", 0) > 0.3 else 0.72) if typ != "cloth" else 0.8
        ),
        alpha=0.55 if typ == "cloth" else 1,
    )
    if typ == "mesh":
        ob = meshob(name, z[key + "_vertices"], z[key + "_faces"], mat)
        for p in ob.data.polygons:
            p.use_smooth = len(ob.data.polygons) > 20 and len(p.vertices) == 3
        rendered[key] = ob
    elif typ == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1)
        ob = bpy.context.object
        ob.name = name
        ob.data.materials.append(mat)
        for p in ob.data.polygons:
            p.use_smooth = True
        rendered[key] = ob
    elif typ == "cloth":
        base = contact_asset[key + "_vertices_frames"][start]
        faces = contact_asset[key + "_faces"]
        ob = meshob(name, base, faces, mat)
        nt = mat.node_tree
        wire = nt.nodes.new("ShaderNodeWireframe")
        wire.use_pixel_size = True
        wire.inputs["Size"].default_value = 0.65
        mix = nt.nodes.new("ShaderNodeMixRGB")
        mix.inputs[1].default_value = (*linear(item["color"]), 1)
        mix.inputs[2].default_value = (*linear(np.array(item["color"]) * 0.36), 1)
        nt.links.new(wire.outputs[0], mix.inputs[0])
        nt.links.new(
            mix.outputs[0], nt.nodes.get("Principled BSDF").inputs["Base Color"]
        )
        for poly in ob.data.polygons:
            poly.use_smooth = True
        rendered[key] = ob
    elif typ == "rope":
        line = z[key + "_line"][start]
        v = np.column_stack(
            [line[:, 0] - aspect / 2, 0.5 - line[:, 1], np.full(len(line), 0.07)]
        )
        rendered[key] = curve(name, v, item["color"], item["radius"])
    else:
        empty = bpy.data.objects.new(name, None)
        sc.collection.objects.link(empty)
        r = item["radius"]
        e = item.get("half_extent", [r, r])
        children = []
        if typ == "box":
            bpy.ops.mesh.primitive_cube_add(size=2)
            ob = bpy.context.object
            ob.scale = (e[0], e[1], min(e) / 2)
            bevel = ob.modifiers.new("Small edge highlights", "BEVEL")
            bevel.width = min(e) * 0.06
            bevel.segments = 3
            children.append(ob)
        elif typ == "sphere2d":
            bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=r)
            children.append(bpy.context.object)
        else:
            bpy.ops.mesh.primitive_torus_add(
                major_segments=48,
                minor_segments=12,
                major_radius=r * 0.82,
                minor_radius=r * 0.18,
            )
            children.append(bpy.context.object)
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=32, radius=r * 0.25, depth=r * 0.45
            )
            children.append(bpy.context.object)
            if item.get("wheel_style", "disc") == "disc":
                bpy.ops.mesh.primitive_cylinder_add(
                    vertices=64, radius=r * 0.84, depth=r * 0.23
                )
                children.append(bpy.context.object)
            for a in range(3) if item.get("wheel_style", "disc") == "spokes" else []:
                theta = 2 * math.pi * a / 3
                bpy.ops.mesh.primitive_cube_add(size=2)
                ob = bpy.context.object
                ob.scale = (r * 0.38, r * 0.09, r * 0.12)
                ob.location = (
                    math.cos(theta) * r * 0.48,
                    math.sin(theta) * r * 0.48,
                    0,
                )
                ob.rotation_euler.z = theta
                children.append(ob)
        for ob in children:
            ob.parent = empty
            ob.data.materials.append(mat)
            for poly in ob.data.polygons:
                poly.use_smooth = typ != "box"
        rendered[key] = empty
fluid = []
if kind == "fluid":
    for j, color in enumerate(info["colors"]):
        ids = np.flatnonzero(z["particle_fluid"] == j)
        ob = meshob("Fluid " + str(j), np.zeros((len(ids), 3)), [], None)
        mat = material("Fluid sampled RGB " + str(j), color, rough=0.24)
        group = bpy.data.node_groups.new(
            "Particle spheres " + str(j), "GeometryNodeTree"
        )
        group.interface.new_socket(
            name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
        )
        group.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )
        nodes = group.nodes
        links = group.links
        inp = nodes.new("NodeGroupInput")
        out = nodes.new("NodeGroupOutput")
        ico = nodes.new("GeometryNodeMeshIcoSphere")
        ico.inputs["Radius"].default_value = info["particle_radius_pixels"] / h
        ico.inputs["Subdivisions"].default_value = 2
        sm = nodes.new("GeometryNodeSetMaterial")
        sm.inputs["Material"].default_value = mat
        links.new(ico.outputs["Mesh"], sm.inputs["Geometry"])
        inst = nodes.new("GeometryNodeInstanceOnPoints")
        links.new(inp.outputs["Geometry"], inst.inputs["Points"])
        links.new(sm.outputs["Geometry"], inst.inputs["Instance"])
        links.new(inst.outputs["Instances"], out.inputs["Geometry"])
        mod = ob.modifiers.new("Saved particles", "NODES")
        mod.node_group = group
        fluid.append((ids, ob))


def update(f):
    if kind in ["cloth", "softbody"]:
        cam.matrix_world = Matrix(z["camera_to_world"][f]) @ Matrix.Diagonal(
            (1, -1, -1, 1)
        )
    for item in info["objects"]:
        key = item["key"]
        typ = item["kind"]
        ob = rendered[key]
        if typ in ["mesh", "sphere"]:
            pose = z[key + "_poses"][f]
            ob.hide_render = not np.isfinite(pose).all()
            if not ob.hide_render:
                ob.matrix_world = Matrix(pose)
        elif typ == "cloth":
            points = contact_asset[key + "_vertices_frames"][f]
            ob.hide_render = not np.isfinite(points).all()
            points = np.nan_to_num(points)
            ob.data.vertices.foreach_set("co", points.astype(np.float32).ravel())
            ob.data.update()
        elif typ == "rope":
            line = z[key + "_line"][f]
            ob.hide_render = not np.isfinite(line).all()
            line = np.nan_to_num(line)
            pointcurve_update(
                ob,
                np.column_stack(
                    [
                        line[:, 0] - aspect / 2,
                        0.5 - line[:, 1],
                        np.full(len(line), 0.07),
                    ]
                ),
            )
        else:
            c = z[key + "_centers"][f]
            ob.location = (c[0] - aspect / 2, 0.5 - c[1], 0.045)
    for ids, ob in fluid:
        xy = z["predicted_xy"][f, ids]
        active = (
            (z["particle_birth"][ids] <= f) & np.isfinite(xy).all(1) & (xy >= 0).all(1)
        )
        v = np.column_stack(
            [xy[:, 0] / h - aspect / 2, 0.5 - xy[:, 1] / h, np.full(len(ids), 0.015)]
        )
        v[~active] = [100, 100, -100]
        ob.data.vertices.foreach_set("co", v.astype(np.float32).ravel())
        ob.data.update()
    bpy.context.view_layer.update()


frames = settings["frames"]
for f in frames:
    update(f)
    sc.render.filepath = str(render_output / f"{f:05d}.png")
    bpy.ops.render.render(write_still=True)
    print("FRAME_DONE", kind, f, flush=True)
