"""
creatures.py — 生物模块
  海龟（主角，前后肢划水动画）/ 64 条小鱼的鱼群（整体转向 + 散开）/ 3 丛随水流摆动的海草
运动轨迹来自 paths.py，这里负责建模、材质和逐帧烘焙关键帧。
"""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

import paths as P
from util import (GN, add_empty, add_gn_modifier, add_object, add_point_attr,
                  assign, bake, collection, eulers_from_matrices, new_material,
                  principled)

COLL = "Creatures"


def _bm_to_mesh(bm, name, smooth=True):
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    try:
        me.shade_smooth() if smooth else None
    except AttributeError:
        me.polygons.foreach_set("use_smooth", [smooth] * len(me.polygons))
    return me


def _uvsphere(u=32, v=16):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=1.0)
    return bm


def _subsurf(ob, levels=2):
    m = ob.modifiers.new("Subdiv", "SUBSURF")
    m.levels = 1
    m.render_levels = levels
    return m


# ============================================================================
# 海龟材质
# ============================================================================
def mat_shell():
    m, n, out = new_material("TurtleShell")
    co = n.texcoord("Object")
    cm = n.mapping(co, scale=(0.95, 1.25, 1.0))
    cells = n.voronoi(cm, scale=4.3, feature="F1", rand=0.55)
    edges = n.voronoi(cm, scale=4.3, feature="DISTANCE_TO_EDGE", rand=0.55)
    rel = n.vmath("SUBTRACT", cm, n.vmath("DIVIDE", cells.outputs["Position"], (4.3, 4.3, 4.3)))
    streak = n.noise(n.mapping(rel, scale=(14.0, 14.0, 3.0)), scale=2.0, detail=6.0, rough=0.6, dist=1.5)
    col_cell = n.ramp(n.sep(cells.outputs["Color"])[0],
                      [(0.0, (0.14, 0.07, 0.025)), (0.5, (0.26, 0.13, 0.04)), (1.0, (0.40, 0.22, 0.06))])
    col = n.mix(n.math("MULTIPLY", n.math("POWER", streak.outputs["Fac"], 2.0), 1.6), col_cell, (0.72, 0.42, 0.09))
    mott = n.noise(co, scale=18.0, detail=4.0)
    col = n.mix(n.math("MULTIPLY", mott.outputs["Fac"], 0.35), col, (0.08, 0.05, 0.02), "MULTIPLY")
    edge_mask = n.math("LESS_THAN", edges.outputs["Distance"], 0.018)
    col = n.mix(n.math("MULTIPLY", edge_mask, 0.6), col, (0.50, 0.40, 0.22))
    groove = n.math("MINIMUM", n.math("DIVIDE", edges.outputs["Distance"], 0.05), 1.0)
    rings = n.noise(rel, scale=60.0, detail=2.0)
    h = n.math("ADD", groove, n.math("MULTIPLY", rings.outputs["Fac"], 0.15))
    b = principled(n, Base_Color=col, Roughness=0.5, Coat_Weight=0.15, Coat_Roughness=0.25,
                   Normal=n.bump(h, 0.45, 0.01))
    n.link(b.outputs[0], out.inputs["Surface"])
    return m


def mat_plastron():
    m, n, out = new_material("TurtlePlastron")
    co = n.texcoord("Object")
    v = n.voronoi(n.mapping(co, scale=(1.0, 1.6, 1.0)), scale=4.0, feature="DISTANCE_TO_EDGE")
    col = n.mix(n.math("LESS_THAN", v.outputs["Distance"], 0.03), (0.78, 0.68, 0.45), (0.55, 0.45, 0.28))
    b = principled(n, Base_Color=col, Roughness=0.5, Normal=n.bump(v.outputs["Distance"], 0.2, 0.01))
    n.link(b.outputs[0], out.inputs["Surface"])
    return m


def mat_skin(name, scale, dark=(0.15, 0.10, 0.05), light=(0.62, 0.52, 0.32), edge_w=0.09):
    m, n, out = new_material(name)
    co = n.texcoord("Object")
    cells = n.voronoi(co, scale=scale, feature="F1", rand=0.8)
    edges = n.voronoi(co, scale=scale, feature="DISTANCE_TO_EDGE", rand=0.8)
    var = n.sep(cells.outputs["Color"])[1]
    cell_col = n.mix(n.math("MULTIPLY", var, 0.6), dark, (0.30, 0.22, 0.09))
    z = n.sep(co)[2]
    belly = n.math("SUBTRACT", 1.0, n.math("MULTIPLY", n.math("ADD", z, 0.02), 30.0), clamp=True)
    cell_col = n.mix(n.math("MULTIPLY", belly, 0.7), cell_col, (0.62, 0.52, 0.34))
    em = n.math("SUBTRACT", 1.0, n.math("DIVIDE", edges.outputs["Distance"], edge_w), clamp=True)
    col = n.mix(em, cell_col, light)
    h = n.math("MINIMUM", n.math("DIVIDE", edges.outputs["Distance"], edge_w * 1.3), 1.0)
    b = principled(n, Base_Color=col, Roughness=0.42, Coat_Weight=0.35, Coat_Roughness=0.2,
                   Subsurface_Weight=0.08, Subsurface_Radius=(0.02, 0.012, 0.008), Normal=n.bump(h, 0.5, 0.004))
    n.link(b.outputs[0], out.inputs["Surface"])
    return m


def mat_eye():
    m, n, out = new_material("TurtleEye")
    z = n.sep(n.texcoord("Object"))[2]
    col = n.ramp(z, [(0.55, (0.10, 0.05, 0.02)), (0.72, (0.32, 0.16, 0.05)), (0.86, (0.01, 0.01, 0.01))])
    b = principled(n, Base_Color=col, Roughness=0.08, Coat_Weight=1.0, Coat_Roughness=0.02, Specular_IOR_Level=0.7)
    n.link(b.outputs[0], out.inputs["Surface"])
    return m


# ============================================================================
# 海龟网格
# ============================================================================
def shell_mesh():
    bm = _uvsphere(56, 28)
    for v in bm.verts:
        x, y, z = v.co
        X = x * 0.48
        taper = 1.0 - 0.30 * max(0.0, -x) ** 2 - 0.10 * max(0.0, x) ** 2
        Y = y * 0.41 * taper
        if z >= 0:
            Z = z * 0.21 * (1.0 - 0.18 * max(0.0, -x))
            Z += 0.010 * math.exp(-(Y / 0.05) ** 2) * z          # 中脊
        else:
            Z = z * 0.075
        # 边缘略向外翻
        rim = math.exp(-(z / 0.15) ** 2)
        X *= 1.0 + 0.03 * rim
        Y *= 1.0 + 0.04 * rim
        v.co = (X, Y, Z)
    for f in bm.faces:
        f.material_index = 1 if f.calc_center_median().z < -0.012 else 0
    return _bm_to_mesh(bm, "TurtleShell")


def head_mesh():
    bm = _uvsphere(40, 20)
    for v in bm.verts:
        x, y, z = v.co
        f = 1.0 - 0.38 * max(0.0, x) ** 2
        X = x * 0.135 + (0.012 * max(0.0, x) ** 3)
        Y = y * 0.084 * f
        Z = z * 0.072 * f - 0.016 * max(0.0, x) ** 2 + 0.012 * max(0.0, z) * (1 - abs(x))
        v.co = (X, Y, Z)
    return _bm_to_mesh(bm, "TurtleHead")


def neck_mesh():
    bm = _uvsphere(28, 14)
    for v in bm.verts:
        x, y, z = v.co
        v.co = (x * 0.17, y * 0.088 * (1 + 0.1 * x), z * 0.066)
    return _bm_to_mesh(bm, "TurtleNeck")


def flipper_mesh(name, L, W, T, sweep, side, curve=0.03):
    bm = _uvsphere(28, 16)
    for v in bm.verts:
        x, y, z = v.co
        u = (z + 1.0) / 2.0                         # 0 根部 → 1 末端
        w = W * (1.0 - 0.72 * u ** 1.3) * (0.75 + 0.25 * math.sin(math.pi * min(1, u * 1.6)))
        X = x * w - sweep * u * u * L + curve * math.sin(math.pi * u) + 0.25 * w * (x > 0) * x * 0.4
        Y = side * u * L
        th = T * (1.0 - 0.6 * u) * (1.0 + 0.4 * max(0.0, x))   # 前缘略厚
        Z = y * th
        v.co = (X, Y, Z)
    return _bm_to_mesh(bm, name)


def tail_mesh():
    bm = _uvsphere(16, 8)
    for v in bm.verts:
        x, y, z = v.co
        u = (1 - x) / 2
        v.co = (x * 0.09, y * 0.035 * (0.4 + u), z * 0.025 * (0.4 + u))
    return _bm_to_mesh(bm, "TurtleTail")


def build_turtle(tl):
    collection(COLL)
    tt = tl["turtle"]
    shell_m, plast_m = mat_shell(), mat_plastron()
    skin_head = mat_skin("TurtleSkinHead", 24.0, edge_w=0.07)
    skin_body = mat_skin("TurtleSkin", 48.0, edge_w=0.12)
    skin_flip = mat_skin("TurtleSkinFlipper", 30.0, light=(0.70, 0.62, 0.42), edge_w=0.1)

    root = add_empty("Turtle", COLL, size=0.6)

    me = shell_mesh()
    me.materials.append(shell_m)
    me.materials.append(plast_m)
    shell = add_object("TurtleShell", me, COLL, parent=root)
    _subsurf(shell, 1)

    neck_piv = add_empty("TurtleNeckPivot", COLL, location=(0.40, 0, 0.0), parent=root, size=0.1)
    neck = add_object("TurtleNeck", neck_mesh(), COLL, location=(0.08, 0, 0.005), parent=neck_piv)
    assign(neck, skin_body)
    _subsurf(neck, 1)
    head_c = Vector((0.24, 0.0, 0.035))
    head = add_object("TurtleHead", head_mesh(), COLL, location=head_c, parent=neck_piv)
    assign(head, skin_head)
    _subsurf(head, 2)

    # 眼睛：贴在头部表面并略微内陷，局部 +Z 朝外（瞳孔方向）
    eye_mat = mat_eye()
    bm = _uvsphere(24, 12)
    me_eye = _bm_to_mesh(bm, "TurtleEye")
    me_eye.materials.append(eye_mat)
    for side in (1, -1):
        d = Vector((0.40, 0.80 * side, 0.42)).normalized()
        sx, sy, sz = 0.135, 0.084, 0.072
        k = 1.0 / math.sqrt((d.x / sx) ** 2 + (d.y / sy) ** 2 + (d.z / sz) ** 2)
        p = d * k * 0.93
        f = 1.0 - 0.38 * max(0.0, p.x / sx) ** 2
        p = Vector((p.x, p.y * f, p.z * f))
        e = add_object("TurtleEye_" + ("L" if side > 0 else "R"), me_eye, COLL, location=head_c + p, parent=neck_piv)
        e.scale = (0.019, 0.019, 0.019)
        out = Vector((0.35, 1.0 * side, 0.25)).normalized()
        e.rotation_mode = "QUATERNION"
        e.rotation_quaternion = out.to_track_quat("Z", "X")

    # 四肢
    flips = {}
    spec = {
        "FL": dict(loc=(0.24, 0.30, -0.02), L=0.54, W=0.095, T=0.022, sweep=0.20, side=1, rest=(0.0, 0.10, 0.25)),
        "FR": dict(loc=(0.24, -0.30, -0.02), L=0.54, W=0.095, T=0.022, sweep=0.20, side=-1, rest=(0.0, 0.10, -0.25)),
        "RL": dict(loc=(-0.36, 0.17, -0.03), L=0.22, W=0.075, T=0.016, sweep=0.0, side=1, rest=(0.0, 0.05, 0.95)),
        "RR": dict(loc=(-0.36, -0.17, -0.03), L=0.22, W=0.075, T=0.016, sweep=0.0, side=-1, rest=(0.0, 0.05, -0.95)),
    }
    for key, s in spec.items():
        piv = add_empty(f"TurtleFlipperPivot_{key}", COLL, location=s["loc"], parent=root, size=0.08)
        me = flipper_mesh(f"TurtleFlipper_{key}", s["L"], s["W"], s["T"], s["sweep"], s["side"])
        fo = add_object(f"TurtleFlipper_{key}", me, COLL, parent=piv)
        assign(fo, skin_flip)
        _subsurf(fo, 1)
        flips[key] = (piv, s)
    tail = add_object("TurtleTail", tail_mesh(), COLL, location=(-0.47, 0, -0.02), parent=root)
    assign(tail, skin_body)

    # ---------------- 动画烘焙 ----------------
    frames = np.arange(1, P.FRAMES + 1)
    T = tl["T"]
    ph = tt["stroke_phase"]
    bob_z = 0.022 * np.cos(ph)
    bob_pitch = 0.035 * np.sin(ph + 0.6)
    Rb = tt["R"].copy()
    c, s_ = np.cos(-bob_pitch), np.sin(-bob_pitch)
    Ry = np.zeros_like(Rb)
    Ry[:, 0, 0] = c; Ry[:, 0, 2] = s_; Ry[:, 2, 0] = -s_; Ry[:, 2, 2] = c; Ry[:, 1, 1] = 1
    Rb = Rb @ Ry
    root.rotation_mode = "XYZ"
    bake(root, "location", frames, tt["pos"] + np.stack([0 * bob_z, 0 * bob_z, bob_z], 1))
    bake(root, "rotation_euler", frames, eulers_from_matrices(Rb))

    # 前肢：上下扑动(X) + 攻角(Y) + 前后扫(Z)；后肢：舵式小摆动
    for key, (piv, s) in flips.items():
        piv.rotation_mode = "XYZ"
        r0 = np.array(s["rest"])
        sd = s["side"]
        if key[0] == "F":
            flap = 0.52 * np.sin(ph) - 0.06
            feather = 0.45 * np.cos(ph)
            sweep = 0.30 * np.sin(ph - 0.9)
            e = np.stack([r0[0] + sd * flap, r0[1] + feather, r0[2] + sd * sweep], 1)
        else:
            e = np.stack([r0[0] + sd * 0.12 * np.sin(ph * 0.5 + 1.0), r0[1] + 0.10 * np.sin(ph + 2.0),
                          r0[2] + sd * 0.12 * np.sin(ph * 0.5)], 1)
        bake(piv, "rotation_euler", frames, e)

    neck_piv.rotation_mode = "XYZ"
    head_e = np.stack([0.03 * np.sin(0.21 * T),
                       -0.05 * np.sin(ph + 0.4) - 0.04,
                       0.07 * np.sin(0.29 * T + 1.0)], 1)
    bake(neck_piv, "rotation_euler", frames, head_e)
    return root


# ============================================================================
# 鱼群
# ============================================================================
def fish_mesh():
    xs = np.linspace(0.068, -0.044, 15)
    nseg = 12
    bm = bmesh.new()
    rings = []

    def hw(x):
        u = (0.07 - x) / 0.115
        h = 0.024 * max(4 * u * (1 - u), 0.0) ** 0.55 * (1 - 0.5 * u ** 3)
        h = max(h, 0.0055)
        return h, h * 0.42 * (1 - 0.25 * u)

    for x in xs:
        h, w = hw(x)
        ring = []
        for j in range(nseg):
            a = 2 * math.pi * j / nseg
            ring.append(bm.verts.new((x, w * math.cos(a), h * math.sin(a) - 0.002 * (math.sin(a) < 0))))
        rings.append(ring)
    nose = bm.verts.new((0.074, 0, -0.001))
    for j in range(nseg):
        bm.faces.new((nose, rings[0][(j + 1) % nseg], rings[0][j]))
    for i in range(len(rings) - 1):
        for j in range(nseg):
            bm.faces.new((rings[i][j], rings[i][(j + 1) % nseg], rings[i + 1][(j + 1) % nseg], rings[i + 1][j]))
    endc = bm.verts.new((-0.046, 0, 0))
    for j in range(nseg):
        bm.faces.new((rings[-1][j], rings[-1][(j + 1) % nseg], endc))
    # 尾鳍（叉形）
    p0 = bm.verts.new((-0.042, 0, 0.0055))
    p1 = bm.verts.new((-0.042, 0, -0.0055))
    t1 = bm.verts.new((-0.086, 0, 0.031))
    t2 = bm.verts.new((-0.086, 0, -0.031))
    fk = bm.verts.new((-0.071, 0, 0))
    bm.faces.new((p0, t1, fk))
    bm.faces.new((p0, fk, p1))
    bm.faces.new((p1, fk, t2))
    # 背鳍 / 腹鳍
    for (xa, xb, up) in ((0.022, -0.03, 1), (0.0, -0.028, -1)):
        ha, hb = hw(xa)[0], hw(xb)[0]
        a = bm.verts.new((xa, 0, up * ha * 0.95))
        b = bm.verts.new((xb, 0, up * hb * 0.95))
        c = bm.verts.new(((xa + xb) / 2 - 0.006, 0, up * (max(ha, hb) + 0.012)))
        bm.faces.new((a, c, b))
    # 胸鳍
    for sd in (1, -1):
        a = bm.verts.new((0.035, sd * 0.009, -0.006))
        b = bm.verts.new((0.024, sd * 0.009, -0.009))
        c = bm.verts.new((0.012, sd * 0.024, -0.012))
        bm.faces.new((a, c, b))
    me = bpy.data.meshes.new("FishMesh")
    bm.to_mesh(me)
    bm.free()
    co = np.zeros(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    add_point_attr(me, "rest", co.reshape(-1, 3), kind="FLOAT_VECTOR")
    try:
        me.shade_smooth()
    except AttributeError:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    return me


def mat_fish():
    m, n, out = new_material("Fish")
    rest = n.attr("rest")
    x, y, z = n.sep(rest)
    rnd = n.objinfo("Random")
    zn = n.math("ADD", n.math("MULTIPLY", z, 20.0), 0.5)
    body = n.ramp(zn, [(0.25, (0.80, 0.84, 0.86)), (0.5, (0.35, 0.62, 0.78)), (0.7, (0.10, 0.30, 0.55)), (0.9, (0.04, 0.12, 0.28))])
    tail = n.math("SUBTRACT", 1.0, n.math("DIVIDE", n.math("ADD", x, 0.038), 0.012), clamp=True)
    col = n.mix(n.math("MULTIPLY", tail, 0.95), body, (0.98, 0.78, 0.10))
    stripe = n.math("MULTIPLY", n.math("LESS_THAN", n.math("ABSOLUTE", n.math("SUBTRACT", z, 0.004)), 0.0018),
                    n.math("GREATER_THAN", x, -0.035))
    col = n.mix(n.math("MULTIPLY", stripe, 0.7), col, (0.95, 0.85, 0.35))
    eye_p = n.comb(x, n.math("ABSOLUTE", y), z)
    d = n.vmath("DISTANCE", eye_p, (0.056, 0.0085, 0.006))
    col = n.mix(n.math("LESS_THAN", d, 0.0058), col, (0.85, 0.85, 0.8))
    col = n.mix(n.math("LESS_THAN", d, 0.0036), col, (0.0, 0.0, 0.0))
    col = n.mix(n.math("MULTIPLY", rnd, 0.25), col, (0.6, 0.7, 0.75), "OVERLAY")
    b = principled(n, Base_Color=col, Metallic=0.45, Roughness=0.28, Coat_Weight=0.4, Coat_Roughness=0.1)
    n.link(b.outputs[0], out.inputs["Surface"])
    return m


def fish_wiggle_group():
    g = GN("FishWiggle")
    phase = g.param("Phase", 0.0)
    amp = g.param("Amp", 0.012)
    x = g.sep(g.position())[0]
    w = g.math("POWER", g.math("DIVIDE", g.math("SUBTRACT", 0.05, x), 0.135, clamp=True), 2.0)
    s = g.math("SINE", g.math("ADD", phase, g.math("MULTIPLY", x, 41.9)))
    dy = g.math("MULTIPLY", g.math("MULTIPLY", w, s), amp)
    g.geo_out(g.set_position(g.geo_in(), offset=g.comb(0.0, dy, 0.0)))
    return g


def build_fish(tl):
    collection(COLL)
    fs = tl["fish"]
    me = fish_mesh()
    me.materials.append(mat_fish())
    g = fish_wiggle_group()
    frames = np.arange(1, P.FRAMES + 1)
    up = np.array([0, 0, 1.0])
    objs = []
    for i in range(P.FISH_COUNT):
        ob = add_object(f"Fish_{i:02d}", me, COLL)
        s = float(fs["size"][i])
        ob.scale = (s, s, s)
        mod = add_gn_modifier(ob, g, "Wiggle", Amp=0.011 + 0.003 * (i % 3) / 2)
        F = fs["fwd"][i]
        left = np.cross(up, F)
        left /= np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-6)
        upv = np.cross(F, left)
        mats = np.stack([F, left, upv], axis=2)     # 列向量 = 局部 X/Y/Z
        ob.rotation_mode = "XYZ"
        bake(ob, "location", frames, fs["pos"][i])
        bake(ob, "rotation_euler", frames, eulers_from_matrices(mats))
        bake(ob, f'modifiers["Wiggle"]["{g.params["Phase"]}"]', frames, fs["tail_phase"][i])
        objs.append(ob)
    return objs


# ============================================================================
# 海草
# ============================================================================
def mat_seagrass():
    m, n, out = new_material("Seagrass")
    hf = n.attr("hf", "Fac")
    rnd = n.objinfo("Random")
    col = n.ramp(hf, [(0.0, (0.06, 0.14, 0.03)), (0.55, (0.17, 0.36, 0.07)), (0.92, (0.38, 0.42, 0.12)), (1.0, (0.50, 0.42, 0.18))])
    sp = n.noise(n.texcoord("Object"), scale=60.0, detail=2.0)
    col = n.mix(n.math("MULTIPLY", n.math("GREATER_THAN", sp.outputs["Fac"], 0.68), 0.6), col, (0.55, 0.50, 0.38))
    b = principled(n, Base_Color=col, Roughness=0.45, Coat_Weight=0.2)
    tr = n.new("ShaderNodeBsdfTranslucent")
    n.link(col, tr.inputs["Color"])
    mx = n.new("ShaderNodeMixShader")
    mx.inputs[0].default_value = 0.35
    n.link(b.outputs[0], mx.inputs[1])
    n.link(tr.outputs[0], mx.inputs[2])
    n.link(mx.outputs[0], out.inputs["Surface"])
    return m


def seagrass_mesh(name, seed, n_blades=55, radius=0.32):
    rng = np.random.default_rng(seed)
    verts, faces, hf = [], [], []
    nseg = 14
    for b in range(n_blades):
        r = radius * math.sqrt(rng.uniform())
        a = rng.uniform(0, 2 * math.pi)
        base = np.array([r * math.cos(a), r * math.sin(a), -0.05])
        H = rng.uniform(0.9, 1.9)
        w = rng.uniform(0.014, 0.026)
        lean_dir = np.array([math.cos(a), math.sin(a), 0]) * rng.uniform(0.10, 0.40) + rng.normal(size=3) * [0.12, 0.12, 0]
        th0 = rng.uniform(0, math.pi)
        tw = rng.uniform(-1.5, 1.5)
        i0 = len(verts)
        for k in range(nseg + 1):
            f = k / nseg
            c = base + lean_dir * f * f + np.array([0, 0, H * f])
            th = th0 + tw * f
            ww = w * (1.0 - 0.75 * max(0.0, f - 0.9) / 0.1)
            d = np.array([math.cos(th), math.sin(th), 0]) * ww / 2
            verts += [c - d, c + d]
            hf += [f, f]
        for k in range(nseg):
            a0 = i0 + 2 * k
            faces.append((a0, a0 + 1, a0 + 3, a0 + 2))
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.update()
    add_point_attr(me, "hf", hf)
    try:
        me.shade_smooth()
    except AttributeError:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    return me


def build_seagrass(tl, sway):
    collection(COLL)
    mat = mat_seagrass()
    out = []
    for i, (x, y) in enumerate(tl["grass"]):
        me = seagrass_mesh(f"SeagrassMesh{i}", 90 + i, n_blades=60 if i < 2 else 70, radius=0.30 if i < 2 else 0.45)
        me.materials.append(mat)
        z = float(P.seabed_height(x, y))
        ob = add_object(f"Seagrass_{i}", me, COLL, location=(x, y, z))
        add_gn_modifier(ob, sway, "Sway", Height=1.75, Amplitude=0.22, Speed=0.9 + 0.1 * i)
        out.append(ob)
    return out


def build(tl, sway):
    t = build_turtle(tl)
    f = build_fish(tl)
    g = build_seagrass(tl, sway)
    return t, f, g
